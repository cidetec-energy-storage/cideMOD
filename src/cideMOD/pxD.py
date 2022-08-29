#
# Copyright (c) 2021 CIDETEC Energy Storage.
#
# This file is part of cideMOD.
#
# cideMOD is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
from dolfin import *
from multiphenics import *

import json
import os
import sys
import numpy as np
from collections import namedtuple

from cideMOD.simulation_interface.triggers import SolverCrashed, TriggerDetected, TriggerSurpassed
from cideMOD.helpers.config_parser import CellParser
from cideMOD.helpers.miscellaneous import constant_expression, format_time
from cideMOD.helpers.warehouse import Warehouse
from cideMOD.mesh.base_mesher import DolfinMesher, SubdomainMapper
from cideMOD.mesh.gmsh_adapter import GmshMesher
from cideMOD.models.cell_components import CurrentColector, Electrode, Separator
from cideMOD.models.degradation.equations import *
from cideMOD.models.electrochemical.equations import *
from cideMOD.models.mechanical import mechanical_model
from cideMOD.models.model_options import ModelOptions
from cideMOD.models.particle_models import *
from cideMOD.models.thermal.equations import *
from cideMOD.numerics import solver_conf
from cideMOD.numerics.time_scheme import TimeScheme
from cideMOD.helpers.extract_fom_info import get_mesh_info, initialize_results, extend_results

# Activate this only for production
# set_log_active(False)

comm = MPI.comm_world
if MPI.size(comm) > 1:
    parameters["ghost_mode"] = "shared_facet"

def _print(*args, **kwargs):
    if MPI.rank(comm) == 0:
        print(*args, **kwargs)
        sys.stdout.flush()

class Problem:

    def __init__(self, cell:CellParser, model_options:ModelOptions, save_path=None):

        self.model_options = model_options
        self.c_s_implicit_coupling = model_options.particle_coupling == 'implicit'

        self.cell = cell
        self.save_path = save_path
        
        self.number_of_anode_materials = len(self.cell.negative_electrode.active_materials)
        self.number_of_cathode_materials = len(self.cell.positive_electrode.active_materials)

        if self.model_options.solve_SEI and not cell.negative_electrode.SEI:
            print('SEI model has bad properties in the anode, turning off ...')
            self.model_options.solve_SEI = False

        self.build_cell_properties()
        self.SOC_ini = 1
        self.WH = Warehouse(self.save_path, self)
        self.ready = False

        self._init_rom_dict()

    def set_cell_state(self, SOC, T_ext:float = 298.15, T_ini=None):
        if T_ini is None:
            T_ini = T_ext
        self.T_ini.assign(T_ini)
        self.T_ext.assign(T_ext)
        self.SOC_ini = SOC

    def mesh(self, mesh_engine=DolfinMesher, copy=False):
        if mesh_engine is None:
            mesh_engine = DolfinMesher
        self.mesher = mesh_engine(options=self.model_options, cell=self.cell)
        self.mesher.build_mesh()

    def build_cell_properties(self):
        self.R = self.cell.R
        self.F = self.cell.F
        self.alpha = self.cell.alpha
        self.C_dl_cc = self.cell.doubleLayerCapacitance_cc
        self.c_e_ini = self.cell.electrolyte.initialConcentration
        self.thermal_expansion_rate = self.cell.thermalExpansionRate
        self.Q = self.cell.capacity
        self.area = min([i for i in [self.cell.negative_electrode.area, self.cell.positive_electrode.area] if i])

        self.T_ini = Constant(298.15, name='T_ini')
        self.T_ext = Constant(298.15, name='T_ext')

    def _build_nonlinear_properties(self):
        self.D_e = constant_expression(self.cell.electrolyte.diffusionConstant, **{**self.f_1._asdict(), 'T_0':self.T_ini, 't_p':self.t_p})
        self.kappa = constant_expression(self.cell.electrolyte.ionicConductivity, **{**self.f_1._asdict(), 'T_0':self.T_ini, 't_p':self.t_p})
        self.activity = constant_expression(self.cell.electrolyte.activityDependence, **{**self.f_1._asdict(), 'T_0':self.T_ini, 't_p':self.t_p})

    def build_transport_properties(self):
        self.h_t = self.cell.heatConvection
        self.t_p = self.cell.electrolyte.transferenceNumber

        self._build_nonlinear_properties()
        self.D_e_Ea = self.cell.electrolyte.diffusionConstant_Ea
        self.D_e_Tref = self.cell.electrolyte.diffusionConstant_Tref
        self.kappa_Ea = self.cell.electrolyte.ionicConductivity_Ea
        self.kappa_Tref = self.cell.electrolyte.ionicConductivity_Tref
            
        self.anode = Electrode('a', self.cell.negative_electrode); self.anode.setup(self, self.SOC_ini)
        self.cathode = Electrode('c', self.cell.positive_electrode); self.cathode.setup(self, self.SOC_ini)
        self.separator = Separator(self.cell.separator); self.separator.setup(self)
        self.negativeCC = CurrentColector('negative', self.cell.negative_curent_colector); self.negativeCC.setup(self)
        self.positiveCC = CurrentColector('positive', self.cell.positive_curent_colector); self.positiveCC.setup(self)

    def reset(self):

        self.time = 0.
        # Warehouse setup
        self.WH = Warehouse(self.save_path, self)
        self.set_storage_order()
        self.WH.post_processing_functions(self.post_processing_functions)
        self.WH.internal_variables(self.internal_storage_order)
        self.WH.global_variables(self.global_storage_order)

        # TODO: Review
        _print('\r - Initializing state ... ', end='\r')
        self.initial_guess()
        block_assign(self.u_2, self.u_1)
        block_assign(self.u_0, self.u_1)

        _print(' - Initializing state - Done ')

    def _init_rom_dict(self):
        # Initilize dictionaries that will stores the simulation results and information for ROM model
        self.fom2rom = dict()
        self.fom2rom['results'] = dict()
        self.current_timestep = 0

    def set_new_state(self, time, new_state):
        self.time = time
        assert all(k in new_state.keys() for k in ['ce', 'phie', 'phis', 'jLi', 'cs'])
        _print('\r - Initializing state ... ', end='\r')
        if not hasattr(self, 'nd_model'): # problem with dimensions

            ######################## ce ########################
            assign(self.f_0.c_e, self.P1_map.generate_function({'anode':new_state['ce'],'separator':new_state['ce'],'cathode':new_state['ce']}))

            ######################## phie ########################
            assign(self.f_0.phi_e, self.P1_map.generate_function({'anode':new_state['phie'],'separator':new_state['phie'],'cathode':new_state['phie']}))

            ######################## phis ########################
            assign(self.f_0.phi_s, self.P1_map.generate_function({'anode':new_state['phis'],'cathode':new_state['phis']}))
            if self.fom2rom['areCC']:
                assign(self.f_0.phi_s_cc, self.P1_map.generate_function({'negativeCC':new_state['phis'],'positiveCC':new_state['phis']}))

            ######################## jLi ########################
            assign(self.f_0.j_Li_a0, self.P1_map.generate_function({'anode':self.F*new_state['jLi']}))
            assign(self.f_0.j_Li_c0, self.P1_map.generate_function({'cathode':self.F*new_state['jLi']}))
            # TODO: write in a generalized way for more than one material

            ######################## cs ########################
            fields = self.f_0._fields

            # Get the number of mesh dofs
            ndofs = self.f_0[fields.index("c_s_0_a0")].vector()[:].shape[0]

            # Add values of the first coefficients to the cs_surf calculation
            cs_0 = new_state['cs'][:ndofs]

            # Loop through SGM order
            for j in range(1, self.SGM.order):

                idx_a = fields.index("c_s_"+str(j)+"_a0")
                idx_c = fields.index("c_s_"+str(j)+"_c0")
                cs_jth = new_state['cs'][j*ndofs:(j+1)*ndofs]

                assign(self.f_0[idx_a], self.P1_map.generate_function({'anode':cs_jth}))
                assign(self.f_0[idx_c], self.P1_map.generate_function({'cathode':cs_jth}))
                
                # Add to the cs_surf variable
                cs_0 += cs_jth

            # cs_surf initialization
            assign(self.f_0[fields.index("c_s_0_a0")], self.P1_map.generate_function({'anode':cs_0}))
            assign(self.f_0[fields.index("c_s_0_c0")], self.P1_map.generate_function({'cathode':cs_0}))
            
        else: # adimensional problem
            varnames = {'ce':'c_e', 'phie':'phi_e', 'phis':'phi_s', 'jLi':'j_Li', 'cs': 'cs'}
            new_state = {varnames[key]:item for key, item in new_state.items()}
            adim_state = self.nd_model.scale_variables(new_state)

            ######################## ce ########################
            assign(self.f_0.c_e, self.P1_map.generate_function({'anode':adim_state['c_e'],'separator':adim_state['c_e'],'cathode':adim_state['c_e']}))

            ######################## phie ########################
            assign(self.f_0.phi_e, self.P1_map.generate_function({'anode':adim_state['phi_e'],'separator':adim_state['phi_e'],'cathode':adim_state['phi_e']}))

            ######################## phis ########################
            assign(self.f_0.phi_s, self.P1_map.generate_function({'anode':adim_state['phi_s'],'cathode':adim_state['phi_s']}))
            if self.fom2rom['areCC']:
                assign(self.f_0.phi_s_cc, self.P1_map.generate_function({'negativeCC':adim_state['phi_s'],'positiveCC':adim_state['phi_s']}))

            ######################## jLi ########################
            a_s_a = self.anode.active_material[0].a_s*self.F
            a_s_c = self.cathode.active_material[0].a_s*self.F
            assign(self.f_0.j_Li_a0, self.P1_map.generate_function({'anode':a_s_a*adim_state['j_Li']}))
            assign(self.f_0.j_Li_c0, self.P1_map.generate_function({'cathode':a_s_c*adim_state['j_Li']}))
            # TODO: write in a generalized way for more than one material

            ######################## cs ########################
            fields = self.f_0._fields

            # Get the number of mesh dofs
            ndofs = self.f_0[fields.index("c_s_0_a0")].vector()[:].shape[0]

            # Add values of the first coefficients to the cs_surf calculation
            cs_0 = new_state['cs'][:ndofs]
            adim_cs_a = self.nd_model.scale_variables({'c_s_0_a0':cs_0})['c_s_0_a0']
            adim_cs_c = self.nd_model.scale_variables({'c_s_0_c0':cs_0})['c_s_0_c0']
            # Loop through SGM order
            for j in range(1, self.SGM.order):

                idx_a = fields.index(f"c_s_{j}_a0")
                idx_c = fields.index(f"c_s_{j}_c0")

                cs_jth = new_state['cs'][j*ndofs:(j+1)*ndofs]
                adim_cs = self.nd_model.scale_variables({f"c_s_{j}_a0":cs_jth, f"c_s_{j}_c0":cs_jth})
                # Save current coefficients in their corresponding variables
                assign(self.f_0[idx_a], self.P1_map.generate_function({'anode':adim_cs[f"c_s_{j}_a0"]}))
                assign(self.f_0[idx_c], self.P1_map.generate_function({'cathode':adim_cs[f"c_s_{j}_c0"]}))

                # Add to the cs_surf variable
                adim_cs_a += adim_cs[f"c_s_{j}_a0"]
                adim_cs_c += adim_cs[f"c_s_{j}_c0"]

            # cs_surf initialization
            assign(self.f_0[fields.index("c_s_0_a0")], self.P1_map.generate_function({'anode':adim_cs_a}))
            assign(self.f_0[fields.index("c_s_0_c0")], self.P1_map.generate_function({'cathode':adim_cs_c}))

        block_assign(self.u_2, self.u_1)
        block_assign(self.u_0, self.u_1)
        
        # Save mesh information to avoid obtain it again
        mesh_  = self.fom2rom['mesh'].copy()
        areCC_ = self.fom2rom['areCC']

        self._init_rom_dict()

        # Save older mesh information
        self.fom2rom['mesh'] = mesh_
        self.fom2rom['areCC'] = areCC_

        _print(' - Initializing state - Done ')

    def _build_extra_models(self):
        #Extra models
        if not 'nd_model' in self.__dict__:
            self.SEI_model = SEI()
            self.mechanics = mechanical_model(self.cell)

    def setup(self, mesh_engine=None):
        if not 'mesher' in self.__dict__:
            self.mesh(mesh_engine)
        timer = Timer('Problem Setup')

        self._build_extra_models()        
        self.use_options = False
        _print('Building problem setup')
        _print('\r - Build cell parameters ... ', end='\r')

        # Internal switches
        self.DT = TimeScheme(self.model_options.time_scheme)
        self.beta = Constant(0, name='beta')
        self.v_app = Constant(0, name='v_app')
        self.i_app = Constant(0, name='i_app')
        self.time = 0.
        self.Q_sei = 0

        self.build_implicit_sgm()
        self.build_fs()
        self.build_transport_properties()
        if not self.c_s_implicit_coupling:
            self.build_explicit_sgm()

        # Extract mesh info needed for ROM model
        results_label = 'scaled' if not hasattr(self, 'nd_model') else 'unscaled'
        if not 'mesh' in self.fom2rom.keys():
            # Empty dictionary
            get_mesh_info(self, results_label)

        # Store 'Cs' SGM matrices
        self.fom2rom['SGM'] = dict()
        self.fom2rom['SGM']['M'] = self.SGM.M
        self.fom2rom['SGM']['K'] = self.SGM.K
        self.fom2rom['SGM']['P'] = self.SGM.P

        _print(' - Build cell parameters - Done ')

        # Warehouse setup
        self.set_storage_order()
        self.WH.post_processing_functions(self.post_processing_functions)
        self.WH.internal_variables(self.internal_storage_order)
        self.WH.global_variables(self.global_storage_order)

        # Solver parameters
        self.snes_solver_parameters = {
            "nonlinear_solver": "snes",
            "snes_solver": {
                "linear_solver": "mumps",
                "absolute_tolerance": 1e-4 if self.model_options.mode=='P4D' else 1e-6,
                # "relative_tolerance": 1e-7,
                "maximum_iterations": 10,
                "report": False,
                "line_search": "bt",
                "error_on_nonconvergence": True,
                # "preconditioner": "ilu"
            }
        }

        _print('\r - Initializing state ... ', end='\r')
        self.initial_guess()
        block_assign(self.u_2, self.u_1)
        block_assign(self.u_0, self.u_1)
        self.build_wf_0()
        self.problem_0 = BlockNonlinearProblem(
            self.F_var_0,  self.u_2, self.bc, self.J_var_0)
        self.solver_0 = BlockPETScSNESSolver(self.problem_0)
        self.set_use_options(self.solver_0,use_options=self.use_options)
        _print(' - Initializing state - Done ')

        _print('\r - Build variational formulation ... ', end='\r')
        self.build_wf_implicit_coupling_problem()
        if not self.c_s_implicit_coupling:
            self.build_wf_explicit_coupling_problem()
        _print(' - Build variational formulation - Done ')

        timer.stop()
        _print('Problem Setup finished.')
        _print("Problem has {} dofs.\n".format(MPI.sum(comm,len(self.u_2.block_vector()))))
        self.ready=True

    def set_use_options(self, solver, pc='hypre', use_options=False):
        if use_options:
            if pc == 'hypre':
                num_functions = len(self.f_1._fields)
                petsc_options = solver_conf.hypre()
                petsc_options['pc_hypre_boomeramg_numfunctions']=num_functions
            if pc == 'gamg':
                petsc_options = solver_conf.gamg()
        else:
            petsc_options = solver_conf.base_options.copy()
        if self.save_path:
            petsc_options['log_view'] = ':{}'.format(os.path.join(self.save_path,'snes_profile.log'))
        else:
            petsc_options.pop('log_view', None)
        for key, value in petsc_options.items():
            if value is not None:
                PETScOptions.set(key, value)
            else:
                PETScOptions.set(key)

        if use_options:
            solver.set_from_options()
        else:
            solver.parameters.update(
                self.snes_solver_parameters["snes_solver"])

    def set_storage_order(self):
        self.internal_storage_order = ['c_e', 'phi_e', 'phi_s', 'j_Li']
        if self.model_options.solve_SEI:
            self.internal_storage_order.extend(['c_EC_0_a0', 'delta_a0', 'j_sei_a0'])
        if 'a' in self.cell.structure:
            self.internal_storage_order.append('j_Li_e_a')
            self.internal_storage_order.append(
                    ['x_a', 'list_of_scalar', len(self.anode.active_material)])
        if 'c' in self.cell.structure:
            self.internal_storage_order.append('j_Li_e_c')
            self.internal_storage_order.append(
                    ['x_c', 'list_of_scalar', len(self.cathode.active_material)])
        self.post_processing_functions = []
        self.post_processing_functions.append(self.calc_SOC)

        self.global_storage_order = {
            'voltage': {
                'fnc': self.get_voltage,
                'header': 'Voltage [V]',
            },
            'current': {
                'fnc': self.get_current,
                'header': 'Current [A]'
            },
            'capacity': {
                'fnc': self.get_capacity,
                'header': 'Energy output [Ah]'
            },
            'cathode_SOC': {
                'fnc': self.get_soc_c,
                'header': 'Cathode SOC [-]'
            },
            # Check if lithium is conserved (activate for debug in particle model)
            # 'total_lithium': {
            #     'fnc': self.calculate_total_lithium,
            #     'header': 'Total_lithium [mol]'
            # }
        }
        if self.model_options.solve_thermal:
            self.internal_storage_order.append('temp')
            self.global_storage_order['temperature'] = {
                'fnc': self.get_temperature,
                'header': 'Temperature [K]'
            }
        if self.model_options.solve_mechanic:
            self.internal_storage_order.extend(self.mechanics.storage_order())
            # self.global_storage_order['thickness'] = {
            #     'fnc': self.calculate_thickness_change,
            #     'header': "Thickness change [m]"
            # }
        if self.model_options.solve_SEI:
            self.global_storage_order['Q_sei'] = {
                'fnc': self.get_Q_sei,
                'header': 'Capacity loss to SEI [Ah]'
            }
            self.global_storage_order['delta_sei'] = {
                'fnc': self.get_L_sei,
                'header': 'Average SEI thickness [m]'
            }

        # Additional internal variables. Electric current density.
        # self.internal_storage_order.append(['electric_current', 'vector'])
        # self.internal_storage_order.append(['ionic_current', 'vector'])
        # self.internal_storage_order.extend(['q_ohmic_e', 'q_ohmic_s'])
        # self.internal_storage_order.append(['q_rev_a', 'list_of_scalar', len(self.anode.active_material)])
        # self.internal_storage_order.append(['q_rev_c', 'list_of_scalar', len(self.cathode.active_material)])
        # self.internal_storage_order.append(['q_irrev_a', 'list_of_scalar', len(self.anode.active_material)])
        # self.internal_storage_order.append(['q_irrev_c', 'list_of_scalar', len(self.cathode.active_material)])

    def build_fs(self):
        timer = Timer('Building function space')
        P1 = FunctionSpace(self.mesher.mesh, 'CG', 1)
        LM = FunctionSpace(self.mesher.mesh, 'R', 0)

        elements = []
        # Add electrochemical model elements
        if 'ncc' in self.cell.structure or 'pcc' in self.cell.structure:
            # Add phi_s for Current Colectors if present
            elements += ['c_e', 'phi_e', 'phi_s', 'phi_s_cc', 'lm_phi_s']
        else:
            elements += ['c_e','phi_e','phi_s']
        elements.append('lm_app')
        elements += ['j_Li_a{}'.format(i) for i in range(self.number_of_anode_materials)]
        elements += ['j_Li_c{}'.format(i) for i in range(self.number_of_cathode_materials)]
        if self.model_options.solve_SEI:
            elements += ['j_sei_a{}'.format(i) for i in range(self.number_of_anode_materials)]
            elements += ['delta_a{}'.format(i) for i in range(self.number_of_anode_materials)]
            # Add c_EC elements
            elements += self.SEI_model.fields(self.number_of_anode_materials)
        if self.c_s_implicit_coupling:
            # Add SGM elements
            elements += self.SGM.fields(self.number_of_anode_materials, 'anode')
            elements += self.SGM.fields(self.number_of_cathode_materials, 'cathode')
        # Add Temperature elements
        elements += ['temp']
        if self.model_options.solve_mechanic:
            elements += self.mechanics.fields()
        
        self.FE = namedtuple('Finite_Element_Function',' '.join(elements))

        # Define Electrochemical Function Spaces
        E_c_e = (P1, self.mesher.electrolyte)
        E_phi_e = (P1, self.mesher.electrolyte)
        E_phi_s = (P1, self.mesher.electrodes)
        E_phi_s_CC = (P1, self.mesher.current_colectors)
        LM_phi_s_CC = (P1, self.mesher.electrode_cc_interfaces)
        LM_app = (P1, self.mesher.positive_tab)
        E_j_Li_a = [(P1, self.mesher.anode) for i in range(self.number_of_anode_materials)]
        E_j_Li_c = [(P1, self.mesher.cathode) for i in range(self.number_of_cathode_materials)]
        if self.model_options.solve_SEI:
            E_j_sei_a = [(P1, self.mesher.anode) for i in range(self.number_of_anode_materials)]
            E_delta_a = [(P1, self.mesher.anode) for i in range(self.number_of_anode_materials)]
            # Define c_EC function spaces
            E_c_EC_a = [(P1, self.mesher.anode) for fs in range(len(self.SEI_model.fields(
                self.number_of_anode_materials)))]

        E_electrochemical = [E_c_e, E_phi_e, E_phi_s]
        if 'ncc' in self.cell.structure or 'pcc' in self.cell.structure:
            E_electrochemical.extend([ E_phi_s_CC, LM_phi_s_CC])
        E_electrochemical.append(LM_app)
        E_electrochemical += E_j_Li_a + E_j_Li_c
        if self.model_options.solve_SEI:
            E_electrochemical += E_j_sei_a + E_delta_a + E_c_EC_a
            
        # Define SGM function spaces
        E_c_s = []
        if self.c_s_implicit_coupling:
            E_c_s_a = [(P1, self.mesher.anode) for fs in self.SGM.fields(
                self.number_of_anode_materials, 'anode')]
            E_c_s_c = [(P1, self.mesher.cathode) for fs in self.SGM.fields(
                self.number_of_cathode_materials, 'cathode')]
            E_c_s = E_c_s_a + E_c_s_c

        # Define Thermal function spaces
        if self.model_options.solve_thermal:
            E_T = (P1, None)
        else:
            E_T = (LM, None)
        E_thermal = [E_T]

        # Define Mechanics function spaces
        if self.model_options.solve_mechanic:
            E_mechanics = self.mechanics.shape_functions(self.mesher.mesh)
        else:
            E_mechanics = []

        FS_total = E_electrochemical + E_c_s + E_thermal + E_mechanics

        self.W = BlockFunctionSpace([FS[0] for FS in FS_total], restrict=[FS[1] for FS in FS_total])
        self.V = FunctionSpace(self.mesher.mesh, 'CG', 1)
        self.V_vec = VectorFunctionSpace(self.mesher.mesh, 'CG', 1)
        self.V_0 = FunctionSpace(self.mesher.mesh, 'DG', 0)
        self.V_vec_0 = VectorFunctionSpace(self.mesher.mesh, 'DG', 0)
        # self.P1_map = SubdomainMapper(self.mesher.subdomains, self.mesher.field_data, self.V)
        self.P1_map = SubdomainMapper(self.mesher.field_restrictions, self.V)
        self.du = BlockTrialFunction(self.W)

        self.u_2 = BlockFunction(self.W)
        self.u_1 = BlockFunction(self.W)
        self.u_0 = BlockFunction(self.W)
        if not self.c_s_implicit_coupling:
            self.u_aux_1 = BlockFunction(self.W)
            self.u_aux_2 = BlockFunction(self.W)
        self.u_post_filter = BlockFunction(self.W)

        self.u = BlockTestFunction(self.W)

        self.f_1 = self.FE._make(block_split(self.u_2))
        for i, name in enumerate(self.f_1._fields):
            self.f_1[i].rename(name, 'a Function')
        self.f_0 = self.FE._make(block_split(self.u_1))
        for i, name in enumerate(self.f_1._fields):
            self.f_0[i].rename('{}_0'.format(name), 'a Function')

        self.test = self.FE._make(block_split(self.u))

        # Explicit coupled SGM functions
        self.c_s_surf_1_anode = [Function(self.V) for n in range(
            self.number_of_anode_materials)]
        self.c_s_surf_0_anode = [Function(self.V) for n in range(
            self.number_of_anode_materials)]
        self.c_s_surf_1_cathode = [Function(self.V) for n in range(
            self.number_of_cathode_materials)]
        self.c_s_surf_0_cathode = [Function(self.V) for n in range(
            self.number_of_cathode_materials)]
        
        self.eigenstrain = Function(self.V)

        timer.stop()

    def build_implicit_sgm(self):
        # Build SGM with Implicit Coupling
        if self.c_s_implicit_coupling:
            if self.model_options.solve_mechanic:
                self.SGM = StressEnhancedSpectralModel(self.model_options.particle_order)
            else:
                self.SGM = SpectralLegendreModel(self.model_options.particle_order)
        else:
            self.SGM = StrongCoupledPM()

    def build_explicit_sgm(self):
        # Build SGM with Explicit Coupling
        timer = Timer('Build SGM')
        self.build_electrode_dof_mask()
        if self.model_options.solve_mechanic:
            self.anode_particle_model = StressEnhancedIntercalation(
                active_material=self.anode.active_material, alpha=self.alpha, F=self.F, R=self.R, N_s=self.model_options.N_p, DT=self.DT, nodes=len(self.anode_dofs))
            self.cathode_particle_model = StressEnhancedIntercalation(
                active_material=self.cathode.active_material, alpha=self.alpha, F=self.F, R=self.R, N_s=self.model_options.N_p, DT=self.DT, nodes=len(self.cathode_dofs))
        else:
            self.anode_particle_model = SpectralLegendreModel(
                active_material=self.anode.active_material, alpha=self.alpha, F=self.F, R=self.R, N_s=self.model_options.N_p, DT=self.DT, nodes=len(self.anode_dofs))
            self.cathode_particle_model = SpectralLegendreModel(
                active_material=self.cathode.active_material, alpha=self.alpha, F=self.F, R=self.R, N_s=self.model_options.N_p, DT=self.DT, nodes=len(self.cathode_dofs))
        timer.stop()

    def build_electrode_dof_mask(self):
        self.anode_dofs = self.P1_map.domain_dof_map.get('anode', [])
        self.cathode_dofs = self.P1_map.domain_dof_map.get('cathode', [])

    def update_sgm(self):
        '''
        Once the macroscopic problem is solved the particle models needs
        to be updated.
        '''
        timer = Timer('Update SGM')
        c_e = self.f_1.c_e.vector()
        phi = self.f_1.phi_s.vector() - self.f_1.phi_e.vector()
        T = self.f_1.temp.vector()
        if len(T) == 1:
            temp = T[0]
            T=c_e[:].copy()
            T[:] = temp

        if self.model_options.solve_mechanic:
            if 'a' in self.cell.structure:
                P_s_a = self.mechanics.inclusion_surface_pressure(self.f_1, self.anode.grad, self.c_e_ini, self.anode, self.eigenstrain)
                P_surf_a = project(P_s_a, self.V).vector()
                self.anode_particle_model.microscale_update(
                    c_e[self.anode_dofs], phi[self.anode_dofs], T[self.anode_dofs], P_surf_a[self.anode_dofs])
            if 'c' in self.cell.structure:
                P_s_c = self.mechanics.inclusion_surface_pressure(self.f_1, self.cathode.grad, self.c_e_ini, self.cathode, self.eigenstrain)
                P_surf_c = project(P_s_c, self.V).vector()
                self.cathode_particle_model.microscale_update(
                    c_e[self.cathode_dofs], phi[self.cathode_dofs], T[self.cathode_dofs], P_surf_c[self.cathode_dofs])
        else:
            if 'a' in self.cell.structure:
                self.anode_particle_model.microscale_update(
                    c_e[self.anode_dofs], phi[self.anode_dofs], T[self.anode_dofs])
            if 'c' in self.cell.structure:
                self.cathode_particle_model.microscale_update(
                    c_e[self.cathode_dofs], phi[self.cathode_dofs], T[self.cathode_dofs])

        timer.stop()

    def update_c_s_surf(self, relaxation=1):
        '''
        Once the particle models problem are solved the macroscopic model needs
        to be updated. This function update the concentration on the solid
        surface for the anode and the cathode.
        '''
        timer = Timer('Update C_s_surf')
        for i in range(self.number_of_anode_materials):
            self.c_s_surf_1_anode[i].vector()[self.anode_dofs] = self.anode_particle_model.c_s_surf()[:, i].flatten()
            if self.model_options.solve_mechanic:
                self.eigenstrain.vector()[self.anode_dofs] = self.anode_particle_model.eigenstrain()[:,i].flatten()
        for j in range(self.number_of_cathode_materials):
            self.c_s_surf_1_cathode[j].vector()[self.cathode_dofs] = self.cathode_particle_model.c_s_surf()[:, j].flatten()
            if self.model_options.solve_mechanic:
                self.eigenstrain.vector()[self.cathode_dofs] = self.cathode_particle_model.eigenstrain()[:,j].flatten()
        timer.stop()

    def initial_guess(self):
        # c_e initial
        assign(self.f_0.c_e, self.P1_map.generate_function({'anode':self.c_e_ini, 'separator':self.c_e_ini, 'cathode':self.c_e_ini}))

        # c_s initial
        self.c_s_a_ini = [material.c_s_ini for material in self.anode.active_material]
        self.c_s_c_ini = [material.c_s_ini for material in self.cathode.active_material]
        
        # Init implicit SGM
        self.SGM.initial_guess(self.f_0, 'anode', self.c_s_a_ini)
        self.SGM.initial_guess(self.f_0, 'cathode', self.c_s_c_ini)
        
        # Init explicit SGM
        if not self.c_s_implicit_coupling:
            self.cathode_particle_model.initial_guess(self.c_s_c_ini)
            self.anode_particle_model.initial_guess(self.c_s_a_ini)
            self.update_c_s_surf()

        # phi_s initial
        # First OCV of each material is calculated according with their initial concentrations
        U_a_ini = [material._U_check([material.c_s_ini/material.c_s_max])
                for material in self.anode.active_material]
        U_c_ini = [material._U_check([material.c_s_ini/material.c_s_max])
                for material in self.cathode.active_material]
        # Then the largest or lowest is selected to avoid overcharge/underdischarge
        if round(self.SOC_ini) == 1:
            phi_s_a = max(U_a_ini)[0] if U_a_ini else 0
            phi_s_c = min(U_c_ini)[0] if U_c_ini else 0
        else:
            phi_s_a = min(U_a_ini)[0] if U_a_ini else 0
            phi_s_c = max(U_c_ini)[0] if U_c_ini else 0

        if self.model_options.solve_SEI:
            self.SEI_model.initial_guess(self.f_0, self.anode.SEI.c_EC_sln * self.anode.SEI.eps)
            assign(self.f_0.delta_a0, interpolate(Constant(self.anode.SEI.delta0), self.f_0.delta_a0.function_space()))
            
        # Finally the values are incorporated in the Function
        assign(self.f_0.phi_s, self.P1_map.generate_function({'anode':0, 'cathode':phi_s_c-phi_s_a }))

        if 'pcc' in self.cell.structure or 'ncc' in self.cell.structure:
            assign(self.f_0.phi_s_cc, self.P1_map.generate_function({'negativeCC':0, 'positiveCC':phi_s_c-phi_s_a }))

        # Initial temp
        assign(self.f_0.temp, interpolate(Constant(self.T_ini), self.f_0.temp.function_space()))
        self.get_state()

    def prepare_solve(self, store_delay=1, time = 0):
        self.i_app.assign(0)
        self.v_app.assign(0)
        self.beta.assign(0)

        self.WH.set_delay(store_delay)

    def get_state(self):
        v = self.get_voltage(self.f_1)
        cur = self.get_current(self.f_1)
        self.state = {'t':self.time, 'v':v, 'i': cur}
        if self.beta.values()[0]==0:
            self.i_app.assign(cur/self.Q)
        else:
            self.v_app.assign(v)


    def solve_ie(self, i_app=30.0, v_app=None, t_f=3600, store_delay=1, max_step=3600, min_step=0.01, triggers=[], adaptive=True):

        if not self.ready:
            self.setup()

        store_fom = not adaptive
        if store_fom:
            if self.current_timestep == 0:
                initialize_results(self, int(np.ceil((t_f-self.time)/min_step)))
                self.WH.store(self.time, store_fom=store_fom)
            else:
                extend_results(self, int(np.ceil((t_f-self.time)/min_step))-1)
        else:
            self.WH.store(self.time, store_fom=store_fom)

        self.prepare_solve(store_delay, self.time)
        if i_app is not None:
            v_app_t=None
            assert isinstance(i_app, (str, int, float))
            if isinstance(i_app, str):
                i_app_t = 0
            else:
                i_app_t = i_app
        if v_app is not None:
            i_app_t=None
            assert isinstance(v_app, (str, int, float))
            if isinstance(v_app, str):
                v_app_t = 0
                v_0 = self.get_voltage()
            else:
                v_app_t = v_app

        self.tau = 1
        self.nu = self.tau*(1+self.tau)/(1+2*self.tau)
        
        _print('Solving ...')

        timer = Timer('Simulation time')
        it = 0
        PETScOptions.set('snes_lag_jacobian', 1)
        PETScOptions.set('snes_max_it', 20)
        self.get_state()
        while self.time < t_f:
            # if it > 0 and not self.use_options:
            #     PETScOptions.set('snes_lag_jacobian', 5)
            #     PETScOptions.set('snes_max_it', 30)

            if isinstance(i_app,str):
                i_app_t = eval( i_app, globals(), {'time': self.time} )
            if isinstance(v_app,str):
                v_app_t = eval( v_app, globals(), {'time': self.time, 'v0': v_0} )

            it += 1

            if adaptive:
                errorcode = self.adaptive_timestep(i_app=i_app_t, v_app=v_app_t, max_step=max_step, min_step=min_step, t_max=t_f, triggers= triggers, initialize=(it==1))
            else:
                errorcode = self.constant_timestep(i_app=i_app_t, v_app=v_app_t, timestep=min_step, triggers=triggers, initialize=(it==1))
            _print('Voltage: {v:.4f}\tCurrent: {i:.2e}\tTime: {time}\033[K'.format(
                time=format_time(self.state['t']),
                **self.state),end='\r')
            
            if errorcode != 0:
                timer.stop()
                if store_fom:
                    self.WH.crop_results() # Crop results
                return self.exit(errorcode)
        _print(f"Reached max time {self.time:.2f} \033[K\n")
        timer.stop()
        return self.exit(errorcode)

    def set_timestep(self, timestep):
        self.DT.set_timestep(timestep)

    def get_timestep(self):
        return self.DT.get_timestep()

    def stoichiometry_checks(self):
        if 'c' in self.cell.structure or 'a' in self.cell.structure:
            x_a, x_c = self.get_stoichiometry()
            for i, material in enumerate(self.cathode.active_material):
                if x_c[i] < min(material.stoichiometry0, material.stoichiometry1):
                    block_assign(self.u_2, self.u_1)
                    return 20
                elif x_c[i] > max(material.stoichiometry1, material.stoichiometry0):
                    block_assign(self.u_2, self.u_1)
                    return 21
            for i, material in enumerate(self.anode.active_material):
                if x_a[i] < min(material.stoichiometry0, material.stoichiometry1):
                    block_assign(self.u_2, self.u_1)
                    return 20
                elif x_a[i] > max(material.stoichiometry1, material.stoichiometry0):
                    block_assign(self.u_2, self.u_1)
                    return 21
        return 0
      
    def constant_timestep(self, i_app, v_app, timestep, triggers= [], store_fom=True, initialize=False):
        timer = Timer('Constant TS')
        errorcode = self.timestep( timestep, i_app, v_app, initialize)
        errorcode = self.accept_timestep(i_app, v_app, timestep, triggers, timer, errorcode, store_fom)
        return errorcode

    def adaptive_timestep(self, i_app, v_app, max_step=1000, min_step=1, t_max=None, triggers=[], initialize=False):
        # Decide wich timestep to use
        timer = Timer('Adaptive TS')
        h = max( min( self.get_timestep()*self.tau, max_step), min_step)
        if t_max is not None:
            if self.time + h > t_max or self.time + h + min_step > t_max:
                h = t_max - self.time
                min_step=h
                max_step=h
        errorcode = self.timestep( h, i_app, v_app, initialize)
        if errorcode != 0:
            if h == min_step:
                return errorcode
            else:
                timer.stop()
                self.tau = 0.5
                self.nu = self.tau*(1+self.tau)/(1+2*self.tau)
                errorcode = self.adaptive_timestep(i_app, v_app, max_step, min_step, t_max, triggers=triggers)
                return errorcode
        error = self.get_time_filter_error()
        self.tau = self.DT.update_time_step(max(error), h, tol = 1e-2, max_step = max_step, min_step = min_step)
        self.nu = self.tau*(1+self.tau)/(1+2*self.tau)
        if self.tau < 1:
            # This means the result is not accurate, need to recompute
            timer.stop()
            errorcode = self.adaptive_timestep(i_app, v_app, max_step, min_step, t_max, triggers=triggers)
            return errorcode
        elif self.tau >= 1:
            # This means the result is acepted, advance
            errorcode = self.accept_timestep(i_app, v_app, h, triggers, timer, errorcode, False)
            return errorcode

    def accept_timestep(self, i_app, v_app, ts, triggers, timer, errorcode, store_fom):
        self.get_state()
        try:
            for t in triggers:
                t.check(self.state)
        except TriggerSurpassed as e:
            timer.stop()
            new_tstep = e.new_tstep(self.get_timestep())
            block_assign(self.u_2, self.u_1) # Reset solution to avoid possible Nan values
            errorcode = self.constant_timestep(i_app, v_app, new_tstep, triggers=triggers, store_fom=store_fom)
            return errorcode
        except TriggerDetected as e:
            errorcode = e
            print(f"{str(e)} at {self.state['t']:.2f} s \033[K\n")
        self.time += ts
        self.advance_problem(store_fom)
        timer.stop()
        return errorcode

    def advance_problem(self, store_fom=False):
        self.WH.store(self.time, force=self.time == 0, store_fom=store_fom)
        block_assign(self.u_0, self.u_1)
        block_assign(self.u_1, self.u_2)
        if not self.c_s_implicit_coupling:
            self.anode_particle_model.advance_problem()
            self.cathode_particle_model.advance_problem()

    def timestep(self, timestep, i_app=None, v_app=None, initialize=False):
        timer = Timer('Basic TS')
        try:
            if self.c_s_implicit_coupling:
                self.tstep_implicit(timestep, i_app=i_app, v_app=v_app, initialize=initialize)
            else:
                it, err = self.tstep_explicit(timestep, i_app=i_app, v_app=v_app)
            timer.stop()
            return 0
        except Exception as e:
            timer.stop()
            return SolverCrashed(e)

    def running_mode(self, i_app, v_app):
        assert None in (i_app, v_app)
        assert i_app is not None or v_app is not None
        if i_app is None:
            self.set_voltage(v_app)
        else:
            self.set_current(i_app)

    def tstep_implicit(self, h=10, i_app=None, v_app=None, initialize=False):
        self.set_timestep(h)
        self.running_mode(i_app, v_app)
        try:
            if initialize:
                print("initializing solution")
                self.solver_0.solve()
            self.solver_implicit.solve()
        except Exception as e:
            raise e
        
    def tstep_explicit(self, h=10, i_app=None, v_app=None, tol=1e-6, max_iter=200):
        '''
        Needed operations for each time step such as: saving data,
        solving calculations and uptating variables.

        Parameters
        ----------
        - h : Real, default = 10.
            Time step value.
        - i_app : Real, default = 30.
            Apparent current.
        - tol : Real, default = 1e-6.
            Error tolerance for convergence.
        - max_iter : Int, default = 50.
            Number of maximun iterations.

        Return
        ------
        - it : Int.
            Number of iterations.
        - err : Real.
            Last iteration error.

        '''

        self.set_timestep(h)
        self.running_mode(i_app, v_app)
        err = 1e3
        err_ref = err
        it = 0
        try:
            while err > tol or it < 2:
                if it > max_iter:
                    _print('{} iter reached error: '.format(max_iter), err, "\033[K")
                    break
                it += 1
                timer = Timer('Inner Loop')
                # save previous result and initialize loop result
                block_assign(self.u_aux_2, self.u_aux_1)
                block_assign(self.u_aux_1, self.u_2)
                for c_s_surf_0, c_s_surf_1 in zip(self.c_s_surf_0_anode, self.c_s_surf_1_anode):
                    assign(c_s_surf_0, c_s_surf_1)
                for c_s_surf_0, c_s_surf_1 in zip(self.c_s_surf_0_cathode, self.c_s_surf_1_cathode):
                    assign(c_s_surf_0, c_s_surf_1)

                # self.solver_explicit.solve()
                # Calculation step
                self.sgm_timestep()  # first particles
                self.solver_explicit.solve()

                # Calculate errors
                err_anode = [errornorm(c_s_surf_1, c_s_surf_0, 'l2', 0, self.mesher.mesh)
                             for c_s_surf_1, c_s_surf_0 in zip(self.c_s_surf_1_anode, self.c_s_surf_0_anode)]
                err_cathode = [errornorm(c_s_surf_1, c_s_surf_0, 'l2', 0, self.mesher.mesh)
                               for c_s_surf_1, c_s_surf_0 in zip(self.c_s_surf_1_cathode, self.c_s_surf_0_cathode)]
                err_mesoscale = [0 for i in range(self.W.num_sub_spaces())]
                for i in range(self.W.num_sub_spaces()):
                    err_mesoscale[i] = errornorm(self.u_aux_1.sub(i), self.u_2.sub(
                        i), degree_rise=0, mesh=self.mesher.mesh)
                err = max(max(max(err_anode or [0]), max(
                    err_cathode or [0])), max(err_mesoscale))

                _print('Voltage: {v:.4f}\tCurrent: {i:.2e}\tTime: {time}\tIter: {it:03d}\tError:{err:.2e}'.format(
                    it=it,err=err,time=format_time(self.state['t']),**self.state),end='\r')
                timer.stop()
                if it%10 == 0: 
                    if err/err_ref>0.25:
                        max_iter = it
                    err_ref = err
        except Exception as e:
            raise e

        return it, err

    def sgm_timestep(self, relaxation=1):
        '''
        Needed operations for each time step for the SGM, solving
        calculations and updating variables.

        Parameters
        ----------
        - relaxation : Real between 0-1, default = 1.
            Parameter which modifies the concentration improving the convergence.

        '''

        timer = Timer('SGM Timestep')
        self.update_sgm()

        self.anode_particle_model.solve()
        self.cathode_particle_model.solve()

        self.update_c_s_surf(relaxation=relaxation)
        timer.stop()

    def build_wf_0(self):
        # Notice that some of them could be list.
        d = self.mesher.get_measures()
        self.j_Li_e_a = 0
        self.j_Li_e_c = 0
        self.j_Li_e_a += sum((material.a_s*self.f_1._asdict()['j_Li_a{}'.format(i)]
            for i, material in enumerate(self.anode.active_material)))
        self.j_Li_e_c += sum((material.a_s*self.f_1._asdict()['j_Li_c{}'.format(i)]
            for i, material in enumerate(self.cathode.active_material)))
        if self.model_options.solve_SEI:
            self.j_Li_e_a += sum(material.a_s*self.f_1._asdict()['j_sei_a{}'.format(i)]
            for i, material in enumerate(self.anode.active_material))

        # c_e_0
        F_c_e_0 = [(self.f_1.c_e - self.f_0.c_e) * self.test.c_e * d.x_a +
                   (self.f_1.c_e - self.f_0.c_e) * self.test.c_e * d.x_s +
                   (self.f_1.c_e - self.f_0.c_e) * self.test.c_e * d.x_c]

        # c_s_0
        F_c_s_0 = []
        if self.c_s_implicit_coupling:
            F_c_s_0_a = self.SGM.wf_0(
                self.f_0, self.f_1, self.test, 'anode', d.x_a)
            F_c_s_0_c = self.SGM.wf_0(
                self.f_0, self.f_1, self.test, 'cathode', d.x_c)
            F_c_s_0 = F_c_s_0_a + F_c_s_0_c

        # phi_e
        F_phi_e_a = phi_e_equation(phi_e=self.f_1.phi_e, test=self.test.phi_e, dx=d.x_a, c_e=self.f_1.c_e, j_Li=self.j_Li_e_a, kappa=self.anode.kappa, kappa_D=self.anode.kappa_D, domain_grad=self.anode.grad, L=self.anode.L)
        F_phi_e_s = phi_e_equation(phi_e=self.f_1.phi_e, test=self.test.phi_e, dx=d.x_s, c_e=self.f_1.c_e, j_Li=None, kappa=self.separator.kappa, kappa_D=self.separator.kappa_D, domain_grad=self.separator.grad, L=self.separator.L)
        F_phi_e_c = phi_e_equation(phi_e=self.f_1.phi_e, test=self.test.phi_e, dx=d.x_c, c_e=self.f_1.c_e, j_Li=self.j_Li_e_c, kappa=self.cathode.kappa, kappa_D=self.cathode.kappa_D, domain_grad=self.cathode.grad, L=self.cathode.L)

        self.F_phi_e = F_phi_e_a \
                     + F_phi_e_s \
                     + F_phi_e_c

        self.F_phi_e = [self.F_phi_e]

        # phi_s
        if 'ncc' in self.cell.structure:
            sigma_ratio_a = self.anode.sigma/self.negativeCC.sigma
            F_phi_s_a = phi_s_equation(phi_s=self.f_1.phi_s, test=self.test.phi_s, dx=d.x_a, j_Li=self.j_Li_e_a, sigma=self.anode.sigma, domain_grad=self.anode.grad, L=self.anode.L, lagrange_multiplier=self.f_1.lm_phi_s, dS=d.S_a_ncc, phi_s_test=self.test.phi_s)
            F_phi_s_ncc = phi_s_equation(phi_s=self.f_1.phi_s_cc, test=self.test.phi_s_cc, dx=d.x_ncc, j_Li=None, sigma=self.negativeCC.sigma, domain_grad=self.negativeCC.grad, L=self.negativeCC.L, lagrange_multiplier=self.f_1.lm_phi_s, dS=d.S_ncc_a, phi_s_cc_test=self.test.phi_s_cc, scale_factor=sigma_ratio_a) 
        else:
            F_phi_s_a = phi_s_equation(phi_s=self.f_1.phi_s, test=self.test.phi_s, dx=d.x_a, j_Li=self.j_Li_e_a, sigma=self.anode.sigma, domain_grad=self.anode.grad, L=self.anode.L)
            F_phi_s_ncc = 0
        
        if 'pcc' in self.cell.structure:
            sigma_ratio_c = self.cathode.sigma/self.positiveCC.sigma 
            F_phi_s_c = phi_s_equation(phi_s=self.f_1.phi_s, test=self.test.phi_s, dx=d.x_c, j_Li=self.j_Li_e_c, sigma=self.cathode.sigma, domain_grad=self.cathode.grad, L=self.cathode.L, lagrange_multiplier=self.f_1.lm_phi_s, dS=d.S_c_pcc, phi_s_test=self.test.phi_s)
            F_phi_s_pcc = phi_s_equation(phi_s=self.f_1.phi_s_cc, test=self.test.phi_s_cc, dx=d.x_pcc, j_Li=None, sigma=self.positiveCC.sigma, domain_grad=self.positiveCC.grad, L=self.positiveCC.L, lagrange_multiplier=self.f_1.lm_phi_s, dS=d.S_pcc_c, phi_s_cc_test=self.test.phi_s_cc, scale_factor=sigma_ratio_c) 
            F_phi_bc_p = phi_s_bc(I_app = self.f_1.lm_app, test=self.test.phi_s_cc, ds = d.s_c, scale_factor=sigma_ratio_c)
            bc_index = 1
        elif 'c' in self.cell.structure:
            F_phi_s_c = phi_s_equation(phi_s=self.f_1.phi_s, test=self.test.phi_s, dx=d.x_c, j_Li=self.j_Li_e_c, sigma=self.cathode.sigma, domain_grad=self.cathode.grad, L=self.cathode.L)
            F_phi_bc_p = phi_s_bc(I_app = self.f_1.lm_app, test=self.test.phi_s, ds = d.s_c)
            bc_index = 0
            F_phi_s_pcc = 0
        else:
            F_phi_s_c = 0
            F_phi_s_pcc = 0
            F_phi_bc_p = phi_s_bc(I_app = self.f_1.lm_app, test=self.test.phi_s_cc, ds = d.s_c, scale_factor=1)
            bc_index = 1

        if 'ncc' in self.cell.structure or 'pcc' in self.cell.structure:
            F_phi_s_lm = phi_s_continuity(self.f_1.phi_s, self.f_1.phi_s_cc, self.test.lm_phi_s, dS_el=d.S_a_ncc, dS_cc=d.S_ncc_a) + phi_s_continuity(self.f_1.phi_s, self.f_1.phi_s_cc, self.test.lm_phi_s, dS_el=d.S_c_pcc, dS_cc=d.S_pcc_c)
            self.F_phi_s = [F_phi_s_a + F_phi_s_c, F_phi_s_ncc + F_phi_s_pcc, F_phi_s_lm]
        else:
            self.F_phi_s = [F_phi_s_a + F_phi_s_c] 
        self.F_phi_s[bc_index] += -F_phi_bc_p

        # j_Li
        F_j_Li = []
        # Build c_s surface for anode and cathode
        c_s_surf_a = self.c_s_a_ini
        c_s_surf_c = self.c_s_c_ini

        # ANODE
        for i, material in enumerate(self.anode.active_material):
            j_li_index = self.f_1._fields.index(f'j_Li_a{i}')
            if self.model_options.solve_SEI:
                delta_index = self.f_1._fields.index(f'delta_a{i}')
                j_sei_index = self.f_1._fields.index(f'j_sei_a{i}')
                J_total= self.f_1[j_sei_index]+self.f_1[j_li_index]
                j_li = j_Li_equation(material, self.f_1.c_e, c_s_surf_a[i],
                                    self.alpha, self.f_1.phi_s, self.f_1.phi_e, self.F, self.R, self.f_1.temp, self.i_app,
                                    J_total, self.anode.SEI, self.f_1[delta_index])
            else:
                j_li = j_Li_equation(material, self.f_1.c_e, c_s_surf_a[i], 
                                    self.alpha, self.f_1.phi_s, self.f_1.phi_e, self.F, self.R, self.f_1.temp, self.i_app)

            F_j_Li.append(
                self.f_1[j_li_index] * self.test[j_li_index] * d.x_a -
                j_li * self.test[j_li_index] * d.x_a
            )

        if self.model_options.solve_SEI:
            F_j_Li.extend(self.SEI_model.equations(self.f_0, self.f_1, self.test, d.x_a, self.anode.active_material, self.F, self.R))

        # CATHODE
        for i, material in enumerate(self.cathode.active_material):
            j_li_index = self.f_1._fields.index(f'j_Li_c{i}')
            j_li = j_Li_equation(material, self.f_1.c_e, c_s_surf_c[i], self.alpha, self.f_1.phi_s, 
            self.f_1.phi_e, self.F, self.R, self.f_1.temp, -self.i_app)

            F_j_Li.append(
                self.f_1[j_li_index] * self.test[j_li_index] * d.x_c -
                j_li * self.test[j_li_index] * d.x_c
            )


        # T_0

        F_T_0 = [(self.f_1.temp - self.f_0.temp) * self.test.temp * d.x]

        # Boundary Conditions
        # - anode: Dirichlet
        
        if 'ncc' in self.cell.structure:
            bcs = [DirichletBC(self.W.sub(3), Constant(0), self.mesher.boundaries, self.mesher.field_data['negativePlug'])]
        else:
            bcs = [DirichletBC(self.W.sub(2), Constant(0), self.mesher.boundaries, self.mesher.field_data['negativePlug'])]
        
        if 'pcc' in self.cell.structure:
            phi_s = self.f_1.phi_s_cc
        else:
            phi_s = self.f_1.phi_s
        # - cathode: Newman or Dirichlet via Lagrange multiplier
        self.F_lm_app = [
            (1 - self.beta) * (self.f_1.lm_app - self.i_app*self.Q/self.area) * self.test.lm_app * d.s_c +
            self.beta * (phi_s - self.v_app) * self.test.lm_app * d.s_c
            ]

        # Mechanics
        if self.model_options.solve_mechanic:
            F_disp = 0
            if 'ncc' in self.cell.structure:
                F_disp += self.mechanics.displacement_wf(self.f_1, self.test, d.x_ncc, self.negativeCC, grad=grad)
            if 'a' in self.cell.structure:
                F_disp += self.mechanics.displacement_wf(self.f_1, self.test, d.x_a, self.anode, self.c_e_ini, self.eigenstrain, grad=grad)
            F_disp += self.mechanics.displacement_wf(self.f_1, self.test, d.x_s, self.separator, self.c_e_ini, grad=grad)
            if 'c' in self.cell.structure:
                F_disp += self.mechanics.displacement_wf(self.f_1, self.test, d.x_c, self.cathode, self.c_e_ini, self.eigenstrain, grad=grad)
            if 'pcc' in self.cell.structure:
                F_disp += self.mechanics.displacement_wf(self.f_1, self.test, d.x_pcc, self.positiveCC, grad=grad)
            
            F_hydr_stress = 0
            if 'ncc' in self.cell.structure:
                F_hydr_stress += self.mechanics.hydrostatic_stress_wf(self.f_1, self.test, d.x_ncc, self.negativeCC, grad=grad)
            if 'a' in self.cell.structure:
                F_hydr_stress += self.mechanics.hydrostatic_stress_wf(self.f_1, self.test, d.x_a, self.anode, self.c_e_ini, self.eigenstrain, grad=grad)
            F_hydr_stress += self.mechanics.hydrostatic_stress_wf(self.f_1, self.test, d.x_s, self.separator, self.c_e_ini, grad=grad)
            if 'c' in self.cell.structure:
                F_hydr_stress += self.mechanics.hydrostatic_stress_wf(self.f_1, self.test, d.x_c, self.cathode, self.c_e_ini, self.eigenstrain, grad = grad)
            if 'pcc' in self.cell.structure:
                F_hydr_stress += self.mechanics.hydrostatic_stress_wf(self.f_1, self.test, d.x_pcc, self.positiveCC, grad=grad)
            
            fixed_bc = self.mechanics.fixed_bc(self.W,self.f_1, [(0,)])
            slip_bc = self.mechanics.slip_bc(self.W, self.f_1, [(None,0), (None,1), (None,None,0), (None,None,1)])
            stress_bc = self.mechanics.pressure_bc(10, self.test, d.s_a, self.mesher.mesh) + self.mechanics.pressure_bc(10, self.test, d.s_c, self.mesher.mesh)

            bcs += fixed_bc
            self.F_mechanics = [F_disp, F_hydr_stress]
        else:
            self.F_mechanics = []

        # Build Residual Form and Jacobian
        F_var_0 = F_c_e_0 \
            + self.F_phi_e \
            + self.F_phi_s + self.F_lm_app \
            + F_j_Li  \
            + F_c_s_0\
            + F_T_0 \
            + self.F_mechanics

        J_var_0 = block_derivative(F_var_0, self.u_2, self.du)
        self.F_var_0 = BlockForm(F_var_0)
        self.J_var_0 = BlockForm(J_var_0)
        self.bc = BlockDirichletBC(bcs, self.W)

    def implicit_sgm_wf(self):
        d = self.mesher.get_measures()
        F_c_s_a = self.SGM.wf_implicit_coupling(
            self.f_0, self.f_1, self.test, 'anode', d.x_a, self.DT, self.anode.active_material, self.F, self.R)
        F_c_s_c = self.SGM.wf_implicit_coupling(
            self.f_0, self.f_1, self.test, 'cathode', d.x_c, self.DT, self.cathode.active_material, self.F, self.R)
        F_c_s = F_c_s_a + F_c_s_c
        return F_c_s

    def build_wf_implicit_coupling_problem(self):

        d = self.mesher.get_measures()

        # Double layer capacitance:
        if self.anode.C_dl:
            self.j_Li_e_a += self.anode.C_dl * self.DT.dt(self.f_0.phi_s-self.f_0.phi_e,self.f_1.phi_s-self.f_1.phi_e) * sum([material.a_s for material in self.anode.active_material])
        if self.cathode.C_dl:
            self.j_Li_e_c += self.cathode.C_dl * self.DT.dt(self.f_0.phi_s-self.f_0.phi_e,self.f_1.phi_s-self.f_1.phi_e) * sum([material.a_s for material in self.cathode.active_material])

        # c_e
        F_c_e_a = c_e_equation(c_e_0=self.f_0.c_e,c_e=self.f_1.c_e, test=self.test.c_e, dx=d.x_a, DT=self.DT, j_Li=self.j_Li_e_a, D_e=self.anode.D_e,eps_e=self.anode.eps_e,t_p=self.t_p, F=self.F, domain_grad=self.anode.grad, L=self.anode.L)
        F_c_e_s = c_e_equation(c_e_0=self.f_0.c_e,c_e=self.f_1.c_e, test=self.test.c_e, dx=d.x_s, DT=self.DT, j_Li=None, D_e=self.separator.D_e,eps_e=self.separator.eps_e,t_p=self.t_p, F=self.F, domain_grad=self.separator.grad, L=self.separator.L)
        F_c_e_c = c_e_equation(c_e_0=self.f_0.c_e,c_e=self.f_1.c_e, test=self.test.c_e, dx=d.x_c, DT=self.DT, j_Li=self.j_Li_e_c, D_e=self.cathode.D_e,eps_e=self.cathode.eps_e,t_p=self.t_p, F=self.F, domain_grad=self.cathode.grad, L=self.cathode.L)

        self.F_c_e = [F_c_e_a + F_c_e_s + F_c_e_c]

        # c_s
        F_c_s = []
        F_c_s = self.implicit_sgm_wf()

        if self.SGM.c_s_surf(self.f_1, 'anode'):
            self.c_s_surf_a = self.SGM.c_s_surf(self.f_1, 'anode')
        else:
            self.c_s_surf_a = self.c_s_a_ini

        if self.SGM.c_s_surf(self.f_1, 'cathode'):
            self.c_s_surf_c = self.SGM.c_s_surf(self.f_1, 'cathode')
        else:
            self.c_s_surf_c = self.c_s_c_ini

        # j_Li
        F_j_Li = []
        for i, material in enumerate(self.anode.active_material):
            j_li_index = self.f_1._fields.index(f'j_Li_a{i}')
            if self.model_options.solve_SEI:
                delta_index = self.f_1._fields.index(f'delta_a{i}')
                j_li = j_Li_equation(material=material, c_e=self.f_1.c_e, c_s_surf=self.c_s_surf_a[i],
                                    alpha=self.alpha, phi_s=self.f_1.phi_s, phi_e=self.f_1.phi_e, F=self.F, R=self.R, T=self.f_1.temp, current=self.i_app,
                                    J=self.j_Li_e_a/material.a_s, SEI=self.anode.SEI, delta=self.f_1[delta_index])
            else:
                j_li = j_Li_equation(material=material, c_e=self.f_1.c_e, c_s_surf=self.c_s_surf_a[i],
                                    alpha=self.alpha, phi_s=self.f_1.phi_s, phi_e=self.f_1.phi_e, F=self.F, R=self.R, T=self.f_1.temp, current=self.i_app)
            F_j_Li.append(
                self.f_1[j_li_index] * self.test[j_li_index] * d.x_a - j_li * self.test[j_li_index] * d.x_a
            )
        
        if self.model_options.solve_SEI:
                F_j_Li.append(self.SEI_model.equations(self.f_0, self.f_1, self.test, d.x_a, self.anode.active_material, self.F, self.R, self.DT))

        for i, material in enumerate(self.cathode.active_material):
            j_li_index = self.f_1._fields.index('j_Li_c0')
            j_li = j_Li_equation(material, self.f_1.c_e, self.c_s_surf_c[i], self.alpha, self.f_1.phi_s, 
                self.f_1.phi_e, self.F, self.R, self.f_1.temp, -self.i_app)
            F_j_Li.append(
                self.f_1[j_li_index+i] * self.test[j_li_index+i] * d.x_c - j_li * self.test[j_li_index+i] * d.x_c
            )
        
        # phi_e
        F_phi_e_a = phi_e_equation(phi_e=self.f_1.phi_e, test=self.test.phi_e, dx=d.x_a, c_e=self.f_1.c_e, j_Li=self.j_Li_e_a, kappa=self.anode.kappa, kappa_D=self.anode.kappa_D, domain_grad=self.anode.grad, L=self.anode.L)
        F_phi_e_s = phi_e_equation(phi_e=self.f_1.phi_e, test=self.test.phi_e, dx=d.x_s, c_e=self.f_1.c_e, j_Li=None, kappa=self.separator.kappa, kappa_D=self.separator.kappa_D, domain_grad=self.separator.grad, L=self.separator.L)
        F_phi_e_c = phi_e_equation(phi_e=self.f_1.phi_e, test=self.test.phi_e, dx=d.x_c, c_e=self.f_1.c_e, j_Li=self.j_Li_e_c, kappa=self.cathode.kappa, kappa_D=self.cathode.kappa_D, domain_grad=self.cathode.grad, L=self.cathode.L)
        
        self.F_phi_e = F_phi_e_a \
                     + F_phi_e_s \
                     + F_phi_e_c

        self.F_phi_e = [self.F_phi_e]
        
        # phi_s
        if 'ncc' in self.cell.structure:
            sigma_ratio_a = self.anode.sigma/self.negativeCC.sigma
            F_phi_s_a = phi_s_equation(phi_s=self.f_1.phi_s, test=self.test.phi_s, dx=d.x_a, j_Li=self.j_Li_e_a, sigma=self.anode.sigma, domain_grad=self.anode.grad, L=self.anode.L, lagrange_multiplier=self.f_1.lm_phi_s, dS=d.S_a_ncc, phi_s_test=self.test.phi_s)
            F_phi_s_ncc = phi_s_equation(phi_s=self.f_1.phi_s_cc, test=self.test.phi_s_cc, dx=d.x_ncc, j_Li=None, sigma=self.negativeCC.sigma, domain_grad=self.negativeCC.grad, L=self.negativeCC.L, lagrange_multiplier=self.f_1.lm_phi_s, dS=d.S_ncc_a, phi_s_cc_test=self.test.phi_s_cc, scale_factor=sigma_ratio_a) 
        else:
            F_phi_s_a = phi_s_equation(phi_s=self.f_1.phi_s, test=self.test.phi_s, dx=d.x_a, j_Li=self.j_Li_e_a, sigma=self.anode.sigma, domain_grad=self.anode.grad, L=self.anode.L)
            F_phi_s_ncc = 0
        
        if 'pcc' in self.cell.structure:
            sigma_ratio_c = self.cathode.sigma/self.positiveCC.sigma 
            F_phi_s_c = phi_s_equation(phi_s=self.f_1.phi_s, test=self.test.phi_s, dx=d.x_c, j_Li=self.j_Li_e_c, sigma=self.cathode.sigma, domain_grad=self.cathode.grad, L=self.cathode.L, lagrange_multiplier=self.f_1.lm_phi_s, dS=d.S_c_pcc, phi_s_test=self.test.phi_s)
            F_phi_s_pcc = phi_s_equation(phi_s=self.f_1.phi_s_cc, test=self.test.phi_s_cc, dx=d.x_pcc, j_Li=None, sigma=self.positiveCC.sigma, domain_grad=self.positiveCC.grad, L=self.positiveCC.L, lagrange_multiplier=self.f_1.lm_phi_s, dS=d.S_pcc_c, phi_s_cc_test=self.test.phi_s_cc, scale_factor=sigma_ratio_c) 
            F_phi_bc_p = phi_s_bc(I_app = self.f_1.lm_app, test=self.test.phi_s_cc, ds = d.s_c, scale_factor=sigma_ratio_c)
            bc_index = 1
        elif 'c' in self.cell.structure:
            F_phi_s_c = phi_s_equation(phi_s=self.f_1.phi_s, test=self.test.phi_s, dx=d.x_c, j_Li=self.j_Li_e_c, sigma=self.cathode.sigma, domain_grad=self.cathode.grad, L=self.cathode.L)
            F_phi_bc_p = phi_s_bc(I_app = self.f_1.lm_app, test=self.test.phi_s, ds = d.s_c)
            bc_index = 0
            F_phi_s_pcc = 0
        else:
            F_phi_s_c = 0
            F_phi_s_pcc = 0
            F_phi_bc_p = phi_s_bc(I_app = self.f_1.lm_app, test=self.test.phi_s_cc, ds = d.s_c, scale_factor=1)
            bc_index = 1

        if 'ncc' in self.cell.structure or 'pcc' in self.cell.structure:
            F_phi_s_lm = phi_s_continuity(self.f_1.phi_s, self.f_1.phi_s_cc, self.test.lm_phi_s, dS_el=d.S_a_ncc, dS_cc=d.S_ncc_a) + phi_s_continuity(self.f_1.phi_s, self.f_1.phi_s_cc, self.test.lm_phi_s, dS_el=d.S_c_pcc, dS_cc=d.S_pcc_c)
            self.F_phi_s = [F_phi_s_a + F_phi_s_c, F_phi_s_ncc + F_phi_s_pcc, F_phi_s_lm]
        else:
            self.F_phi_s = [F_phi_s_a + F_phi_s_c] 
        self.F_phi_s[bc_index] += -F_phi_bc_p

        # T

        if self.model_options.solve_thermal:

            j_li_a = []
            j_li_c = []

            for i, material in enumerate(self.anode.active_material):
                j_li_index = self.f_1._fields.index('j_Li_a0')
                j_li_a.append(self.f_1[j_li_index+i])
            for i, material in enumerate(self.cathode.active_material):
                j_li_index = self.f_1._fields.index('j_Li_c0')
                j_li_c.append(self.f_1[j_li_index+i])

            q_ncc = q_equation(self.negativeCC, self.f_1, None, self.test.temp, d.x_ncc, None)
            q_a = q_equation(self.anode, self.f_1, self.c_s_surf_a, self.test.temp, d.x_a, self.i_app)
            q_s = q_equation(self.separator, self.f_1, None, self.test.temp, d.x_s, None)
            q_c = q_equation(self.cathode, self.f_1, self.c_s_surf_c, self.test.temp, d.x_c, self.i_app)
            q_pcc = q_equation(self.positiveCC, self.f_1, None, self.test.temp, d.x_pcc, None)

            F_T_ncc = T_equation(T_0=self.f_0.temp, T=self.f_1.temp, test=self.test.temp, dx=d.x_ncc, DT=self.DT, rho=self.negativeCC.rho,
                                 c_p=self.negativeCC.c_p, k_t=self.negativeCC.k_t, q=q_ncc, grad=self.negativeCC.grad, L=self.negativeCC.L, alpha=1e-3)
            F_T_a = T_equation(T_0=self.f_0.temp, T=self.f_1.temp, test=self.test.temp, dx=d.x_a, DT=self.DT,
                               rho=self.anode.rho, c_p=self.anode.c_p, k_t=self.anode.k_t, q=q_a, grad=self.anode.grad, L=self.anode.L)
            F_T_s = T_equation(T_0=self.f_0.temp, T=self.f_1.temp, test=self.test.temp, dx=d.x_s, DT=self.DT, rho=self.separator.rho,
                               c_p=self.separator.c_p, k_t=self.separator.k_t, q=q_s, grad=self.separator.grad, L=self.separator.L)
            F_T_c = T_equation(T_0=self.f_0.temp, T=self.f_1.temp, test=self.test.temp, dx=d.x_c, DT=self.DT, rho=self.cathode.rho,
                               c_p=self.cathode.c_p, k_t=self.cathode.k_t, q=q_c, grad=self.cathode.grad, L=self.cathode.L)
            F_T_pcc = T_equation(T_0=self.f_0.temp, T=self.f_1.temp, test=self.test.temp, dx=d.x_pcc, DT=self.DT, rho=self.positiveCC.rho,
                                 c_p=self.positiveCC.c_p, k_t=self.positiveCC.k_t, q=q_pcc, grad=self.positiveCC.grad, L=self.positiveCC.L, alpha=1e-3)

            F_T_boundary = self.h_t * (self.f_1.temp - self.T_ext) * self.test.temp * d.s

            self.F_T = [F_T_ncc+F_T_a +
                        F_T_s +
                        F_T_c + F_T_pcc +
                        F_T_boundary]
        else:
            self.F_T = [(self.f_1.temp - self.T_ini) * self.test.temp * d.x]

        F_var_implicit = self.F_c_e \
            + self.F_phi_e \
            + self.F_phi_s + self.F_lm_app\
            + F_j_Li \
            + F_c_s \
            + self.F_T \
            + self.F_mechanics

        J_var_implicit = block_derivative(F_var_implicit, self.u_2, self.du)
        self.F_var_implicit = BlockForm(F_var_implicit)
        self.J_var_implicit = BlockForm(J_var_implicit)

        self.problem_implicit = BlockNonlinearProblem(
            self.F_var_implicit,  self.u_2, self.bc, self.J_var_implicit)
        self.solver_implicit = BlockPETScSNESSolver(self.problem_implicit)
        self.set_use_options(self.solver_implicit, use_options=self.use_options)

    def build_wf_explicit_coupling_problem(self):

        d = self.mesher.get_measures()
        # j_Li
        F_j_Li = []
        for i, material in enumerate(self.anode.active_material):
            j_li_index = self.f_1._fields.index('j_Li_a0')
            j_li = j_Li_equation(material, self.f_1.c_e, self.c_s_surf_1_anode[i], self.alpha, self.f_1.phi_s,
                                 self.f_1.phi_e, self.F, self.R, self.f_1.temp, self.i_app)

            F_j_Li.append(
                self.f_1[j_li_index+i] * self.test[j_li_index+i] * d.x_a - j_li * self.test[j_li_index+i] * d.x_a
            )

        for i, material in enumerate(self.cathode.active_material):
            j_li_index = self.f_1._fields.index('j_Li_c0')
            j_li = j_Li_equation(material, self.f_1.c_e, self.c_s_surf_1_cathode[i], self.alpha, self.f_1.phi_s,
                                 self.f_1.phi_e, self.F, self.R, self.f_1.temp, -self.i_app)

            F_j_Li.append(
                self.f_1[j_li_index+i] * self.test[j_li_index+i] * d.x_c - j_li * self.test[j_li_index+i] * d.x_c
            )


        F_var_explicit = self.F_c_e \
            + self.F_phi_e \
            + self.F_phi_s + self.F_lm_app \
            + F_j_Li \
            + self.F_T \
            + self.F_mechanics

        J_var_explicit = block_derivative(F_var_explicit, self.u_2, self.du)
        self.F_var_explicit = BlockForm(F_var_explicit)
        self.J_var_explicit = BlockForm(J_var_explicit)

        self.problem_explicit = BlockNonlinearProblem(
            self.F_var_explicit,  self.u_2, self.bc, self.J_var_explicit)
        self.solver_explicit = BlockPETScSNESSolver(self.problem_explicit)
        if self.use_options:
            self.solver_explicit.set_from_options()
        else:
            self.solver_explicit.parameters.update(
                self.snes_solver_parameters["snes_solver"])

        self.anode_particle_model.setup()
        self.cathode_particle_model.setup()

    def calculate_total_lithium(self):
        internal_li = 0
        d = self.mesher.get_measures()
        if self.c_s_implicit_coupling:
            if self.anode.L:
                internal_li += self.anode.L * \
                    self.SGM.Li_amount(self.f_1, 'anode',
                                       self.anode.active_material, d.x_a)
            if self.cathode.L:
                internal_li += self.cathode.L * \
                    self.SGM.Li_amount(self.f_1, 'cathode',
                                       self.cathode.active_material, d.x_c)
        else:
            if self.anode.L:
                internal_li += self.anode.L * self.anode_particle_model.Li_amount()
            if self.cathode.L:
                internal_li += self.cathode.L * self.cathode_particle_model.Li_amount()
        disolved_li = 0
        if self.anode.L:
            disolved_li += self.anode.eps_e * self.anode.L * assemble(self.f_1.c_e*d.x_a)
        if self.separator.L:
            disolved_li += self.separator.eps_e * self.separator.L * assemble(self.f_1.c_e*d.x_s)
        if self.cathode.L:
            disolved_li += self.cathode.eps_e * self.cathode.L * assemble(self.f_1.c_e*d.x_c)
        total_li = (internal_li+disolved_li)*self.area
        return total_li

    def calc_SOC(self):
        if not self.c_s_implicit_coupling:
            c_s_surf_a = self.c_s_surf_1_anode
            c_s_surf_c = self.c_s_surf_1_cathode
        else:
            c_s_surf_a = self.SGM.c_s_surf(self.f_1, 'anode')
            c_s_surf_c = self.SGM.c_s_surf(self.f_1, 'cathode')
        self.x_a = [0 for x in self.anode.active_material]
        self.x_c = [0 for x in self.cathode.active_material]
        for i, material in enumerate(self.anode.active_material):
            self.x_a[i] = c_s_surf_a[i]/material.c_s_max
        for i, material in enumerate(self.cathode.active_material):
            self.x_c[i] = c_s_surf_c[i]/material.c_s_max

    def get_voltage(self, x=None):
        if x is None:
            x = self.f_1
        if 'pcc' in self.cell.structure or not 'c' in self.cell.structure:
            phi_s = x.phi_s_cc
        else:
            phi_s = x.phi_s
        return assemble(phi_s * self.mesher.ds_c)

    def get_temperature(self, x=None):
        if x is None:
            x=self.f_1
        return x.temp.vector().max() 

    def get_time_filter_error(self):
        t=Timer('TF Error')
        error = []
        for index in range(len(self.u_2.block_split())):
            try:
                self.u_post_filter[index].assign(
                    self.nu/2*(2/(1+self.tau)*self.u_2[index]-2*self.u_1[index]+2*self.tau/(1+self.tau)*self.u_0[index]))
                # error.append(assemble(self.u_post_filter[index]**2*dx)**0.5)
                if not self.c_s_implicit_coupling:
                    error.append(self.anode_particle_model.get_time_filter_error(self.nu, self.tau))
                    error.append(self.cathode_particle_model.get_time_filter_error(self.nu, self.tau))
            except Exception as e:
                raise e
        error.append(self.u_post_filter.block_vector().norm('linf'))
        t.stop()
        return error

    def get_current(self, x=None):
        if x is None:
            x = self.f_1
        return assemble(x.lm_app * self.mesher.ds_c) * self.area

    def get_capacity(self):
        if 'Q_out' not in self.__dict__:
            self.Q_out = 0
        self.Q_out -= self.get_current() * self.get_timestep() / 3600
        return self.Q_out

    def get_Q_sei(self):
        j_instant_sei = assemble(-(self.f_1.j_sei_a0+self.f_0.j_sei_a0)*self.anode.active_material[0].a_s*self.mesher.dx_a)*self.anode.L
        value =  self.get_timestep() * 0.5 * j_instant_sei * self.cell.area /3600
        self.Q_sei += value
        return self.Q_sei

    def get_L_sei(self):
        L_sei = [assemble(self.f_1[self.f_1._fields.index(f'delta_a{i}')]*self.mesher.dx_a) for i in range(self.number_of_anode_materials)]
        return L_sei

    def get_stoichiometry(self):
        self.calc_SOC()
        X_c = [max(project(x, self.V).vector()[self.P1_map.domain_dof_map['cathode']]) for x in self.x_c]
        X_a = [max(project(x, self.V).vector()[self.P1_map.domain_dof_map['anode']]) for x in self.x_a]
        return [X_a, X_c]

    def get_eps_s_approx_a(self):
        return 100 - assemble(self.anode.active_material[0].eps_s*self.mesher.dx_a)/self.cell.negative_electrode.active_materials[0].volumeFraction * 100

    def get_eps_s_approx_c(self):
        return 100 - assemble(self.cathode.active_material[0].eps_s*self.mesher.dx_c)/self.cell.positive_electrode.active_materials[0].volumeFraction * 100

    def get_soc_c(self):
        if self.c_s_implicit_coupling:
            x_c = []
            leg_int = self.SGM._leg_volume_integral()
            for k, material in enumerate(self.cathode.active_material):
                x_c.append(0)
                c_s_index = self.f_1._fields.index('c_s_0_{}{}'.format('c', k))
                for i in range(self.SGM.order):
                    if i==0:
                        c = assemble((self.f_1[c_s_index]-sum([self.f_1[ind] for ind in range(1,self.SGM.order)]))*self.mesher.dx_c)*leg_int[i]
                    else:
                        c = assemble(self.f_1[c_s_index+i]*self.mesher.dx_c)*leg_int[i]
                    x_c[k]+=c
                x_c[k]*=3/material.c_s_max
            return x_c

    def set_voltage(self, v=None):

        self.beta.assign(1)

        if v == None:
            self.v_app.assign(self.get_voltage(self.f_0))
        else:
            self.v_app.assign(v)

    def set_current(self, i=None):
        """
        Set current in Amperes (A)

        Parameters
        ----------
        i : float, optional
            Current in Amperes (A), by default None
        """

        self.beta.assign(0)

        if i == None:
            self.i_app.assign(self.get_current(self.f_0)/self.Q)
        else:
           self.i_app.assign(i/self.Q)

    def exit(self, errorcode):
        self.fom2rom['results']['time']    = self.WH.global_var_arrays[0].copy()
        v_index = [i+1 for i,v in enumerate(self.WH.global_vars.keys()) if v=='voltage'][0]
        self.fom2rom['results']['voltage'] = self.WH.global_var_arrays[v_index].copy()
        self.WH.write_globals(self.model_options.clean_on_exit)
        if self.c_s_implicit_coupling:
            for i, _ in enumerate(self.c_s_surf_1_anode):
                assign(self.c_s_surf_1_anode[i], project(
                    self.SGM.c_s_surf(self.f_1, 'anode')[i]))
            for i, _ in enumerate(self.c_s_surf_1_cathode):
                assign(self.c_s_surf_1_cathode[i], project(
                    self.SGM.c_s_surf(self.f_1, 'cathode')[i]))
        return errorcode


class StressProblem(Problem):

    def calculate_thickness_change(self):
        timer = Timer('Thickness Change')

        d = self.mesher.get_measures()

        # Thickness change due to thermal expansion
        if self.model_options.solve_thermal:
            F_dL_thermal = sum(
                [(self.f_1.temp * self.T_ini) * self.thermal_expansion_rate * L * dx for L, dx in [
                    (self.negativeCC.L, d.x_ncc),
                    (self.anode.L, d.x_a),
                    (self.separator.L, d.x_s),
                    (self.cathode.L, d.x_c),
                    (self.positiveCC.L, d.x_pcc)
                ] if L]
            )
            dL_th = assemble(F_dL_thermal)
        else:
            dL_th = 0

        # Thickness change due to lithium (de)intercalation
        self.strain = self.calc_strain()

        dL_int = assemble(self.strain*(self.anode.L or 0) *
                          d.x_a + self.strain*(self.cathode.L or 0)*d.x_c)

        timer.stop()

        return dL_th, dL_int

    def calc_strain(self):
        d = self.mesher.get_measures()

        if self.c_s_implicit_coupling:
            c_avg_anode = self.SGM.get_average_c_s(
                self.f_1, self.anode)
            c_avg_cathode = self.SGM.get_average_c_s(
                self.f_1, self.cathode)

            dV = 0
            for i, material in enumerate(self.anode.active_material):
                dV += material.eps_s * material.omega * \
                    (c_avg_anode[i] - material.c_s_ini)

            for i, material in enumerate(self.cathode.active_material):
                dV += material.eps_s * material.omega * \
                    (c_avg_cathode[i] - material.c_s_ini)

        else:
            c_avg_anode = [Function(self.V)
                           for material in self.anode.active_material]
            c_avg_cathode = [Function(self.V)
                             for material in self.cathode.active_material]

            c_s_avg_anode = self.anode_particle_model.get_average_c_s()
            c_s_avg_cathode = self.cathode_particle_model.get_average_c_s()

            dV = 0
            for i, material in enumerate(self.anode.active_material):
                c_avg_anode[i].vector()[self.anode_dofs] = c_s_avg_anode[:, i]
                dV += material.eps_s * material.omega * \
                    (c_avg_anode[i] - material.c_s_ini)

            for i, material in enumerate(self.cathode.active_material):
                c_avg_cathode[i].vector(
                )[self.cathode_dofs] = c_s_avg_cathode[:, i]
                dV += material.eps_s * material.omega * \
                    (c_avg_cathode[i] - material.c_s_ini)
        if assemble(dV*d.x_a + dV*d.x_c) == 0:
            dV += DOLFIN_EPS
        return dV 

    def calc_displacement(self):
        d = self.mesher.get_measures()
        dim = self.mesher.mesh.geometric_dimension()
        V = VectorFunctionSpace(self.mesher.mesh, 'CG', 2)
        W = BlockFunctionSpace([V], restrict=[self.mesher.electrodes])
        u = block_split(BlockTrialFunction(W))
        v = block_split(BlockTestFunction(W))
        a = [
            [inner(sym(grad(u[0])), sym(grad(v[0]))) * d.x_a +
             inner(sym(grad(u[0])), sym(grad(v[0]))) * d.x_c]
        ]
        A = block_assemble(a)
        L = [
            inner(self.strain * Identity(dim), sym(grad(v[0]))) * d.x_a + inner(
                self.strain * Identity(dim), sym(grad(v[0]))) * d.x_c,
        ]
        b = block_assemble(L)

        bc1 = DirichletBC(W.sub(0), Constant(
            [0 for i in range(dim)]), self.mesher.interfaces, 1)
        bc2 = DirichletBC(W.sub(0), Constant(
            [0 for i in range(dim)]), self.mesher.interfaces, 2)
        bcs = BlockDirichletBC([bc1, bc2])
        bcs.apply(A)
        bcs.apply(b)

        displacement = BlockFunction(W)
        block_solve(A, displacement.block_vector(), b)

        self.displacement = block_split(displacement)[0]

    def set_storage_order(self):
        super().set_storage_order()
        self.internal_storage_order.append(['displacement', 'vector'])
        self.post_processing_functions.append(self.calc_displacement)
        self.global_storage_order['thickness'] = {
            'fnc': self.calculate_thickness_change,
            'header': "Thickness change [m]"
        }

from cideMOD.models.nondimensional_model import NondimensionalModel


class NDProblem(Problem):
    def __init__(self, cell:CellParser, model_options:ModelOptions, save_path=None):
        super().__init__(cell, model_options, save_path)
        self.nd_model = NondimensionalModel(self.cell, model_options)
        self.thermal_boundary_conditions = dict()

    def add_thermal_boundary_condition(self, surface_label, h_t, T_ref = None):
        assert surface_label in ['Y_m']
        self.thermal_boundary_conditions[surface_label] = { 'h_t': h_t, 'T_ref': T_ref}
    
    def _build_nonlinear_properties(self):
        dim_f_1 = self.nd_model.dimensional_variables(self.f_1)
        self.dim_variables = self.FE._make(dim_f_1)
        self.D_e = constant_expression(self.cell.electrolyte.diffusionConstant, **{**self.dim_variables._asdict(), 'T_0':self.T_ini, 't_p':self.t_p})
        self.kappa = constant_expression(self.cell.electrolyte.ionicConductivity, **{**self.dim_variables._asdict(), 'T_0':self.T_ini, 't_p':self.t_p})
        self.activity = constant_expression(self.cell.electrolyte.activityDependence, **{**self.dim_variables._asdict(), 'T_0':self.T_ini, 't_p':self.t_p})

    def _build_extra_models(self):
        #Extra models
        self.SEI_model = self.nd_model.SpectralLagrangeModel_EC()
        self.mechanics = mechanical_model(self.cell)
        # if self.cell.electrolyte.type != 'liquid' and self.model_options.solve_mechanic:
        #     self.c_s_implicit_coupling = False

    def mesh(self, mesh_engine=GmshMesher, copy=False):
        if mesh_engine is None:
            mesh_engine = GmshMesher
        assert mesh_engine != DolfinMesher, "Dolfin mesher can't create a good mesh for a nondimensional problem"
        self.mesher = mesh_engine(options=self.model_options, cell=self.cell)
        self.mesher.build_mesh(scale = self.nd_model.L_0)
        if copy and self.save_path:
            mesh_save_path = os.path.join(self.save_path,'mesh')
            os.makedirs(mesh_save_path)
            with open(f'{mesh_save_path}/field_data.json','w') as fout:
                json.dump(self.mesher.field_data,fout,indent=4)
            XDMFFile(f"{mesh_save_path}/mesh.xdmf").write(self.mesher.mesh)
            XDMFFile(f"{mesh_save_path}/mesh_physical_region.xdmf").write(self.mesher.subdomains)
            XDMFFile(f"{mesh_save_path}/mesh_facet_region.xdmf").write(self.mesher.boundaries)

    def build_implicit_sgm(self):
        # Build SGM with Implicit Coupling
        if self.c_s_implicit_coupling:
            self.SGM = NondimensionalSpectralModel(self.model_options.particle_order)
        else:
            self.SGM = StrongCoupledPM()

    def initial_guess(self):
        
        # c_e initial
        c_e_ini_value = self.nd_model.scale_variables({'c_e':self.c_e_ini})['c_e']
        assign(self.f_0.c_e, self.P1_map.generate_function({'anode':c_e_ini_value, 'separator':c_e_ini_value, 'cathode':c_e_ini_value}))

        # c_s initial
        self.c_s_a_ini = [ self.nd_model.scale_variables({'c_s_a{}'.format(material.index): material.c_s_ini })['c_s_a{}'.format(material.index)] for material in self.anode.active_material]
        self.c_s_c_ini = [ self.nd_model.scale_variables({'c_s_c{}'.format(material.index): material.c_s_ini })['c_s_c{}'.format(material.index)] for material in self.cathode.active_material]

        # Init implicit SGM
        self.SGM.initial_guess(self.f_0, 'anode', self.c_s_a_ini)
        self.SGM.initial_guess(self.f_0, 'cathode', self.c_s_c_ini)

        # Init explicit SGM
        if not self.c_s_implicit_coupling:
            self.cathode_particle_model.initial_guess(self.c_s_c_ini)
            self.anode_particle_model.initial_guess(self.c_s_a_ini)
            self.update_c_s_surf()

        # phi_s initial
        # First OCV of each material is calculated according with their initial concentrations 
        U_a_ini = [material._U_check([material.c_s_ini/material.c_s_max]) for material in self.anode.active_material]
        U_c_ini = [material._U_check([material.c_s_ini/material.c_s_max]) for material in self.cathode.active_material]
        # Then the largest or lowest is selected to avoid overcharge/underdischarge
        if round(self.SOC_ini)==1:
            phi_s_a = max(U_a_ini)[0]
            phi_s_c = min(U_c_ini)[0]
        else:
            phi_s_a = min(U_a_ini)[0]
            phi_s_c = max(U_c_ini)[0]
        phi_s_a_el = self.nd_model.scale_variables({'phi_s': 0})['phi_s']
        phi_s_c_el = self.nd_model.scale_variables({'phi_s': phi_s_c-phi_s_a})['phi_s']
        # Finally the values are incorporated in the Function
        assign(self.f_0.phi_s, self.P1_map.generate_function({'anode':phi_s_a_el, 'cathode':phi_s_c_el}))

        if self.model_options.solve_SEI:
            # TODO: scale variables
            self.SEI_model.initial_guess(self.f_0, self.nd_model.scale_variables({'c_EC_a0':self.anode.SEI.c_EC_sln * self.anode.SEI.eps})['c_EC_a0'])
            assign(self.f_0.delta_a0, interpolate(Constant(self.nd_model.scale_variables({'delta_sei_a0':self.anode.SEI.delta0})['delta_sei_a0']), self.f_0.delta_a0.function_space()))

        if 'pcc' in self.cell.structure or 'ncc' in self.cell.structure:
            phi_s_a_cc = self.nd_model.scale_variables({'phi_s_cc': 0})['phi_s_cc']
            phi_s_c_cc = self.nd_model.scale_variables({'phi_s_cc': phi_s_c-phi_s_a})['phi_s_cc']
            # # Finally the values are incorporated as a dolfin Expression
            assign(self.f_0.phi_s_cc, self.P1_map.generate_function({'positiveCC':phi_s_c_cc, 'negativeCC': phi_s_a_cc}))
            
        # Initial temp
        temp_ini = self.nd_model.scale_variables( {'T': self.T_ini} )['T']
        assign(self.f_0.temp, interpolate(Constant(temp_ini), self.f_0.temp.function_space()))

    def build_wf_0(self):
        # Notice that some of them could be list.
        d = self.mesher.get_measures()

        self.j_Li_e_a = sum((self.f_1._asdict()['j_Li_a{}'.format(i)] for i, material in enumerate(self.anode.active_material)))
        self.j_Li_e_c = sum((self.f_1._asdict()['j_Li_c{}'.format(i)] for i, material in enumerate(self.cathode.active_material)))
        if self.model_options.solve_SEI:
            self.j_Li_e_a += sum((self.f_1[self.f_1._fields.index(f'j_sei_a{i}')] for i, mat in enumerate(self.anode.active_material)))
            
        # c_e_0
        F_c_e_0 = [ (self.f_1.c_e - self.f_0.c_e) * self.test.c_e * d.x_a +
                    (self.f_1.c_e - self.f_0.c_e) * self.test.c_e * d.x_s +
                    (self.f_1.c_e - self.f_0.c_e) * self.test.c_e * d.x_c ]

        # c_s_0
        F_c_s_0_a = self.SGM.wf_0(self.f_0, self.f_1, self.test, 'anode', d.x_a)
        F_c_s_0_c = self.SGM.wf_0(self.f_0, self.f_1, self.test, 'cathode', d.x_c)

        F_c_s_0 = F_c_s_0_a + F_c_s_0_c

        # phi_e
        F_phi_e_a = self.nd_model.phi_e_equation(phi_e=self.f_1.phi_e, test=self.test.phi_e, dx=d.x_a, c_e=self.f_1.c_e, j_li=self.j_Li_e_a, T=self.f_1.temp, domain=self.anode)
        F_phi_e_s = self.nd_model.phi_e_equation(phi_e=self.f_1.phi_e, test=self.test.phi_e, dx=d.x_s, c_e=self.f_1.c_e, j_li=None, T=self.f_1.temp, domain=self.separator)
        F_phi_e_c = self.nd_model.phi_e_equation(phi_e=self.f_1.phi_e, test=self.test.phi_e, dx=d.x_c, c_e=self.f_1.c_e, j_li=self.j_Li_e_c, T=self.f_1.temp, domain=self.cathode)
        
        self.F_phi_e = F_phi_e_a \
                     + F_phi_e_s \
                     + F_phi_e_c

        # phi_s
        F_phi_s_el = 0
        F_phi_s_cc = 0
        if 'pcc' in self.cell.structure:
            F_phi_s_pcc = self.nd_model.phi_s_conductor_equation(domain=self.positiveCC, phi_s_cc=self.f_1.phi_s_cc, test=self.test.phi_s_cc, dx=d.x_pcc, lagrange_multiplier=self.f_1.lm_phi_s, dS=d.S_pcc_c)
            F_phi_s_c = self.nd_model.phi_s_electrode_equation(domain=self.cathode, phi_s=self.f_1.phi_s, test=self.test.phi_s, dx=d.x_c, j_li=self.j_Li_e_c, lagrange_multiplier=self.f_1.lm_phi_s, dS=d.S_c_pcc)
            F_phi_bc_p = self.nd_model.phi_s_bc(i_app = self.f_1.lm_app, test=self.test.phi_s_cc, ds = d.s_c, area_ratio=self.mesher.area_ratio_c, eq_scaling=1)
            F_phi_s_el += F_phi_s_c
            F_phi_s_cc += F_phi_s_pcc - F_phi_bc_p
        else:
            F_phi_s_c = self.nd_model.phi_s_electrode_equation(domain=self.cathode, phi_s=self.f_1.phi_s, test=self.test.phi_s, dx=d.x_c, j_li=self.j_Li_e_c)
            F_phi_bc_p = self.nd_model.phi_s_bc(i_app = self.f_1.lm_app, test=self.test.phi_s, ds = d.s_c, area_ratio=self.mesher.area_ratio_c, eq_scaling=self.nd_model.sigma_ref/self.cathode.sigma)
            F_phi_s_el += F_phi_s_c - F_phi_bc_p

        if 'ncc' in self.cell.structure:
            F_phi_s_a = self.nd_model.phi_s_electrode_equation(domain=self.anode, phi_s=self.f_1.phi_s, test=self.test.phi_s, dx=d.x_a, j_li=self.j_Li_e_a, lagrange_multiplier=self.f_1.lm_phi_s, dS=d.S_a_ncc)
            F_phi_s_ncc = self.nd_model.phi_s_conductor_equation(domain=self.negativeCC, phi_s_cc=self.f_1.phi_s_cc, test=self.test.phi_s_cc, dx=d.x_ncc, lagrange_multiplier=self.f_1.lm_phi_s, dS=d.S_ncc_a)
            F_phi_s_el += F_phi_s_a
            F_phi_s_cc += F_phi_s_ncc
        else:
            F_phi_s_a = self.nd_model.phi_s_electrode_equation(domain=self.anode, phi_s=self.f_1.phi_s, test=self.test.phi_s, dx=d.x_a, j_li=self.j_Li_e_a)
            F_phi_s_el += F_phi_s_a 

        if 'pcc' in self.cell.structure or 'ncc' in self.cell.structure:
            F_lm_phi_s = self.nd_model.phi_s_continuity(phi_s_electrode=self.f_1.phi_s, phi_s_cc=self.f_1.phi_s_cc, lm_test=self.test.lm_phi_s, dS_el=d.S_a_ncc, dS_cc=d.S_ncc_a) + \
                self.nd_model.phi_s_continuity(phi_s_electrode=self.f_1.phi_s, phi_s_cc=self.f_1.phi_s_cc, lm_test=self.test.lm_phi_s, dS_el=d.S_c_pcc, dS_cc=d.S_pcc_c) 
            self.F_phi_s = [ F_phi_s_el, F_phi_s_cc, F_lm_phi_s ]
        else:
            self.F_phi_s = [ F_phi_s_el ]
        self.F_phi_e = [self.F_phi_e]
        # j_Li
        # Build c_s surface for anode and cathode
        c_s_surf_a = self.c_s_a_ini
        c_s_surf_c = self.c_s_c_ini

        F_j_Li = []
        for i, material in enumerate(self.anode.active_material):
            j_li_index = self.f_1._fields.index(f'j_Li_a{i}')
            if self.model_options.solve_SEI:
                delta_index = self.f_1._fields.index(f'delta_a{i}')
                j_li = self.nd_model.j_Li_equation(material=material, c_e=self.f_1.c_e, c_s_surf=c_s_surf_a[i],
                                    alpha=self.alpha, phi_s=self.f_1.phi_s, phi_e=self.f_1.phi_e, F=self.F, R=self.R, T=self.f_1.temp, current=self.i_app,
                                    J=self.j_Li_e_a/material.a_s, SEI=self.anode.SEI, delta=self.f_1[delta_index])
            else:
                j_li = self.nd_model.j_Li_equation(material=material, c_e=self.f_1.c_e, c_s_surf=c_s_surf_a[i], phi_s=self.f_1.phi_s, phi_e=self.f_1.phi_e, T=self.f_1.temp, current=self.i_app)

            F_j_Li.append(
                self.f_1[j_li_index] * self.test[j_li_index] * d.x_a - \
                j_li * self.test[j_li_index] * d.x_a
            )
        
        if self.model_options.solve_SEI:
            F_j_Li.extend(self.nd_model.SEI_equations(self.f_0, self.f_1, self.test, d.x_a, self.anode.active_material))

        for i, material in enumerate(self.cathode.active_material):
            j_li_index = self.f_1._fields.index('j_Li_c0')
            j_li = self.nd_model.j_Li_equation(material=material, c_e=self.f_1.c_e, c_s_surf=c_s_surf_c[i], phi_s=self.f_1.phi_s, phi_e=self.f_1.phi_e, T=self.f_1.temp, current=-self.i_app)

            F_j_Li.append(
                self.f_1[j_li_index+i] * self.test[j_li_index+i] * d.x_c - \
                j_li * self.test[j_li_index+i] * d.x_c
            )
        
        # T_0

        F_T_0 = [(self.f_1.temp- self.f_0.temp) * self.test.temp * d.x ]

        # Boundary Conditions
        if 'ncc' in self.cell.structure or 'pcc' in self.cell.structure:
            phi_s_bound_index = 3
            phi_s_bound_field = self.nd_model.phi_s_ref + self.nd_model.solid_potential*self.f_1.phi_s_cc
            bc_value = self.nd_model.scale_variables({'phi_s_cc': 0})['phi_s_cc']
        else:
            phi_s_bound_index = 2
            phi_s_bound_field = self.nd_model.phi_s_ref+ self.nd_model.solid_potential*self.f_1.phi_s
            bc_value = self.nd_model.scale_variables({'phi_s': 0})['phi_s']
        # - anode: Dirichlet
        self.bc = BlockDirichletBC([DirichletBC(self.W.sub(phi_s_bound_index), Constant(bc_value), self.mesher.boundaries, self.mesher.field_data['negativePlug'])])
        # - cathode: Newman or Dirichlet via Lagrange multiplier
        self.F_lm_app = [
            (1 - self.beta) * (self.f_1.lm_app - self.i_app) * self.test.lm_app * d.s_c +
            self.beta * (phi_s_bound_field - self.v_app) * self.test.lm_app * d.s_c
        ]
        
        F_var_0 = F_c_e_0 \
                     + self.F_phi_e \
                     + self.F_phi_s + self.F_lm_app \
                     + F_j_Li  \
                     + F_T_0 \
                     + F_c_s_0

        J_var_0 = block_derivative(F_var_0, self.u_2, self.du)
        self.F_var_0 = BlockForm(F_var_0)
        self.J_var_0 = BlockForm(J_var_0)

    def implicit_sgm_wf(self):
        d = self.mesher.get_measures()
        F_c_s_a = self.SGM.wf_implicit_coupling(f_0=self.f_0, f_1=self.f_1, test=self.test, electrode='anode', dx=d.x_a, DT=self.DT, materials=self.anode.active_material, nd_model=self.nd_model)
        F_c_s_c = self.SGM.wf_implicit_coupling(f_0=self.f_0, f_1=self.f_1, test=self.test, electrode='cathode', dx=d.x_c, DT=self.DT, materials=self.cathode.active_material, nd_model=self.nd_model)
        F_c_s = F_c_s_a + F_c_s_c
        return F_c_s

    def build_wf_implicit_coupling_problem(self):

        d = self.mesher.get_measures()

        # c_e
        F_c_e_a = self.nd_model.c_e_equation(c_e_0=self.f_0.c_e,c_e=self.f_1.c_e, test=self.test.c_e, dx=d.x_a, DT=self.DT, j_li=self.j_Li_e_a, domain=self.anode)
        F_c_e_s = self.nd_model.c_e_equation(c_e_0=self.f_0.c_e,c_e=self.f_1.c_e, test=self.test.c_e, dx=d.x_s, DT=self.DT, j_li=None, domain=self.separator)
        F_c_e_c = self.nd_model.c_e_equation(c_e_0=self.f_0.c_e,c_e=self.f_1.c_e, test=self.test.c_e, dx=d.x_c, DT=self.DT, j_li=self.j_Li_e_c, domain=self.cathode)
        
        self.F_c_e = [ F_c_e_a \
              + F_c_e_s \
              + F_c_e_c ]

        # c_s

        F_c_s = self.implicit_sgm_wf()

        if self.SGM.c_s_surf(self.f_1, 'anode') and self.SGM.c_s_surf(self.f_1, 'cathode'):
            self.c_s_surf_a = self.SGM.c_s_surf(self.f_1, 'anode')
            self.c_s_surf_c = self.SGM.c_s_surf(self.f_1, 'cathode')
        else:
            self.c_s_surf_a = self.c_s_a_ini
            self.c_s_surf_c = self.c_s_c_ini

        F_j_Li = []
        for i, material in enumerate(self.anode.active_material):
            j_li_index = self.f_1._fields.index(f'j_Li_a{i}')
            if self.model_options.solve_SEI:
                delta_index = self.f_1._fields.index(f'delta_a{i}')
                j_li = self.nd_model.j_Li_equation(material=material, c_e=self.f_1.c_e, c_s_surf=self.c_s_surf_a[i],
                                    alpha=self.alpha, phi_s=self.f_1.phi_s, phi_e=self.f_1.phi_e, F=self.F, R=self.R, T=self.f_1.temp, current=self.i_app,
                                    J=self.j_Li_e_a/material.a_s, SEI=self.anode.SEI, delta=self.f_1[delta_index])
            else:
                j_li = self.nd_model.j_Li_equation(material=material, c_e=self.f_1.c_e, c_s_surf=self.c_s_surf_a[i], phi_s=self.f_1.phi_s, phi_e=self.f_1.phi_e, T=self.f_1.temp, current=self.i_app)

            F_j_Li.append(
                (self.f_1[j_li_index] - j_li) * self.test[j_li_index] * d.x_a
            )

        if self.model_options.solve_SEI:
            F_j_Li.extend(self.nd_model.SEI_equations(self.f_0, self.f_1, self.test, d.x_a, self.anode.active_material, self.DT))

        for i, material in enumerate(self.cathode.active_material):
            j_li_index = self.f_1._fields.index(f'j_Li_c{i}')
            j_li = self.nd_model.j_Li_equation(material = material, c_e=self.f_1.c_e, c_s_surf=self.c_s_surf_c[i], phi_s=self.f_1.phi_s, phi_e=self.f_1.phi_e, T=self.f_1.temp, current=-self.i_app)

            F_j_Li.append(
                ( self.f_1[j_li_index] - j_li ) * self.test[j_li_index] * d.x_c
            )

        if self.model_options.solve_thermal:
            F_T_ncc = self.nd_model.T_equation(self.negativeCC, self.DT, self.f_1.temp, self.f_0.temp, self.test.temp, self.f_1, None, self.i_app, d.x_ncc)
            F_T_a = self.nd_model.T_equation(self.anode, self.DT, self.f_1.temp, self.f_0.temp, self.test.temp, self.f_1, self.c_s_surf_a, self.i_app, d.x_a)
            F_T_s = self.nd_model.T_equation(self.separator, self.DT, self.f_1.temp, self.f_0.temp, self.test.temp, self.f_1, None, None, d.x_s)
            F_T_c = self.nd_model.T_equation(self.cathode, self.DT, self.f_1.temp, self.f_0.temp, self.test.temp, self.f_1, self.c_s_surf_c, -self.i_app, d.x_c)
            F_T_pcc = self.nd_model.T_equation(self.positiveCC, self.DT, self.f_1.temp, self.f_0.temp, self.test.temp, self.f_1, None, -self.i_app, d.x_pcc)
            F_T_bc_c = self.nd_model.T_bc_equation(self.positiveCC if 'pcc' in self.cell.structure else self.cathode, self.f_1.temp, self.T_ext, self.h_t, self.test.temp, d.s_c)
            F_T_bc_a = self.nd_model.T_bc_equation(self.negativeCC if 'ncc' in self.cell.structure else self.anode, self.f_1.temp, self.T_ext, self.h_t, self.test.temp, d.s_a)
            h_t = self.thermal_boundary_conditions['Y_m']['h_t'] if 'Y_m' in self.thermal_boundary_conditions else self.h_t
            T_ref = (self.thermal_boundary_conditions['Y_m']['T_ref'] or self.T_ext) if 'Y_m' in self.thermal_boundary_conditions else self.T_ext
            F_T_bc_Ym = self.nd_model.T_bc_equation(None, self.f_1.temp, T_ref, h_t, self.test.temp, d.s_Ym)
            self.F_T = [F_T_ncc + F_T_a + F_T_s + F_T_c + F_T_pcc + F_T_bc_c + F_T_bc_a + F_T_bc_Ym ]
        else:
            self.F_T = [(self.f_1.temp - self.f_0.temp) * self.test.temp * d.x ]

        F_var_implicit = self.F_c_e \
                        + self.F_phi_e \
                        + self.F_phi_s + self.F_lm_app\
                        + F_j_Li \
                        + F_c_s + self.F_T

        J_var_implicit = block_derivative(F_var_implicit, self.u_2, self.du)
        self.F_var_implicit = BlockForm(F_var_implicit)
        self.J_var_implicit = BlockForm(J_var_implicit)

        self.problem_implicit = BlockNonlinearProblem(self.F_var_implicit,  self.u_2, self.bc, self.J_var_implicit)
        self.solver_implicit = BlockPETScSNESSolver(self.problem_implicit)
        self.set_use_options(self.solver_implicit, use_options=self.use_options)

    def set_timestep(self, timestep):
        ts = timestep/self.nd_model.t_c
        self.DT.set_timestep(ts)

    def get_timestep(self):
        ts = self.DT.get_timestep()
        return ts*self.nd_model.t_c

    def get_voltage(self, x=None):
        if x is None:
            x = self.f_1
        if 'ncc' in self.cell.structure or 'pcc' in self.cell.structure:
            return self.nd_model.phi_s_ref + self.nd_model.solid_potential * (assemble( x.phi_s_cc * self.mesher.ds_c) / assemble(1*self.mesher.ds_c) )
        else:
            return self.nd_model.phi_s_ref + self.nd_model.solid_potential * (assemble( x.phi_s * self.mesher.ds_c) / assemble(1*self.mesher.ds_c))

    def get_current(self, x=None):
        if x is None:
            x = self.f_1
        return assemble( x.lm_app * self.mesher.ds_c ) / assemble(1*self.mesher.ds_c) * self.Q

    def get_temperature(self, x=None):
        if x is None:
            x=self.f_1
        return self.nd_model.T_ref+ self.nd_model.thermal_gradient*x.temp.vector().max()

    def get_Q_sei(self):
        if 'Q_sei' not in self.__dict__:
            self.Q_sei = 0
        j_instant_sei = assemble(-(self.f_1.j_sei_a0+self.f_0.j_sei_a0)*self.mesher.dx_a)*self.nd_model.I_0*self.nd_model.L_0**2
        value =  self.get_timestep() * 0.5 * j_instant_sei /3600
        self.Q_sei += value
        return self.Q_sei

    def get_L_sei(self):
        L_sei = [self.f_1[self.f_1._fields.index(f'delta_a{i}')].vector()[self.P1_map.domain_dof_map['anode']].mean()* self.nd_model.delta_sei_a[i] for i in range(self.number_of_anode_materials)]
        return L_sei

