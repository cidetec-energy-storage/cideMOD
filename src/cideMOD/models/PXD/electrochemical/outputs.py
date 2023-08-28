#
# Copyright (c) 2023 CIDETEC Energy Storage.
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
import dolfinx as dfx
from collections import OrderedDict

from cideMOD.cell.warehouse import Warehouse
from cideMOD.cell.variables import ProblemVariables
from cideMOD.cell.components import BatteryCell
from cideMOD.cell.dimensional_analysis import DimensionalAnalysis
from cideMOD.mesh.base_mesher import BaseMesher
from cideMOD.models.PXD.base_model import BasePXDModelOutputs


class ElectrochemicalModelOutputs(BasePXDModelOutputs):
    """
    A class that contains the mandatory methods to be overrided
    related to the outputs of :class:`cideMOD.models.ElectrochemicalModel`.
    """

    def get_outputs_info(self, warehouse: Warehouse) -> None:
        """
        This method modifies a dictionary containing the information of
        both the global and internal variables that can be outputed by
        the electrochemical model.

        Parameters
        ----------
        warehouse: Warehouse
            Object that postprocess, store and write the outputs.
        """
        # Global variables
        warehouse.add_global_variable_info('voltage', fnc=self.get_voltage, default=True,
                                           header="Voltage [V]")
        warehouse.add_global_variable_info('current', fnc=self.get_current, default=True,
                                           header="Current [A]")
        warehouse.add_global_variable_info('capacity', fnc=self.get_capacity, default=True,
                                           header="Discharged capacity [Ah]")

        header = [f"c_e_{domain}_avg [mol/m^3]" for domain in ['a', 's', 'c']]
        warehouse.add_global_variable_info('c_e_avg', fnc=self.get_c_e_avg, default=False,
                                           dtype='list_of_scalar', header=header)

        header = [f"phi_e_{domain}_avg [V]" for domain in ['a', 's', 'c']]
        warehouse.add_global_variable_info('phi_e_avg', fnc=self.get_phi_e_avg, default=False,
                                           dtype='list_of_scalar', header=header)

        header = [f"phi_s_{domain}_avg [V]" for domain in ['ncc', 'a', 'c', 'pcc']]
        warehouse.add_global_variable_info('phi_s_avg', fnc=self.get_phi_s_avg, default=False,
                                           dtype='list_of_scalar', header=header)

        header = [f"i_Li_{domain}_int_avg [A/m^2]" for domain in ['a', 'c']]
        warehouse.add_global_variable_info('i_Li_int_avg', fnc=self.get_i_Li_int_avg,
                                           default=False, dtype='list_of_scalar', header=header)

        header = [f"i_Li_{domain}_total_avg [A/m^2]" for domain in ['a', 'c']]
        warehouse.add_global_variable_info('i_Li_total_avg', fnc=self.get_i_Li_total_avg,
                                           default=False, dtype='list_of_scalar', header=header)

        # Internal variables
        warehouse.add_internal_variable_info('c_e', subdomains='electrolyte',
                                             dtype='scalar', default=True)
        warehouse.add_internal_variable_info('phi_e', subdomains='electrolyte',
                                             dtype='scalar', default=True)
        warehouse.add_internal_variable_info('phi_s', subdomains='solid_conductor',
                                             dtype='scalar', default=True)
        warehouse.add_internal_variable_info('i_Li_int', subdomains='electrodes',
                                             dtype='scalar', default=False)
        warehouse.add_internal_variable_info('i_Li_total', subdomains='electrodes',
                                             dtype='scalar', default=False)
        warehouse.add_internal_variable_info('j_Li_int_a', subdomains='anode',
                                             dtype='list_of_scalar', default=True)
        warehouse.add_internal_variable_info('j_Li_int_c', subdomains='cathode',
                                             dtype='list_of_scalar', default=True)
        warehouse.add_internal_variable_info('j_Li_total_a', subdomains='anode',
                                             dtype='list_of_scalar')
        warehouse.add_internal_variable_info('j_Li_total_c', subdomains='cathode',
                                             dtype='list_of_scalar')
        warehouse.add_internal_variable_info('overpotential_a', subdomains='anode',
                                             dtype='list_of_scalar', default=True)
        warehouse.add_internal_variable_info('overpotential_c', subdomains='cathode',
                                             dtype='list_of_scalar', default=True)
        warehouse.add_internal_variable_info('ionic_current', subdomains='electrolyte',
                                             function_space='P1', dtype='vector')
        warehouse.add_internal_variable_info('electric_current', subdomains='solid_conductor',
                                             function_space='P1', dtype='vector')
        warehouse.add_internal_variable_info('li_ion_flux', subdomains='electrolyte',
                                             dtype='vector')
        warehouse.add_internal_variable_info('li_ion_flux_migration', subdomains='electrolyte',
                                             dtype='vector')
        warehouse.add_internal_variable_info('li_ion_flux_diffusion', subdomains='electrolyte',
                                             dtype='vector')

    def prepare_outputs(self, warehouse: Warehouse, var: ProblemVariables, cell: BatteryCell,
                        mesher: BaseMesher, DA: DimensionalAnalysis, problem) -> None:
        """
        This method computes the expression of the requested internal
        variables to be ready for being evaluated and stored.

        Parameters
        ----------
        warehouse: Warehouse
            Object that postprocess, store and write the outputs.
        var: ProblemVariables
            Object containing the problem variables.
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
        mesher: BaseMesher
            Object that store the mesh information.
        DA: DimensionalAnalysis
            Object where the dimensional analysis is performed.
        problem: Problem
            Object that handles the battery cell simulation.
        """

        # Prepare global variables
        self.problem = problem  # Needed within global variables methods
        self.mesher = mesher
        d = mesher.get_measures()

        if cell.has_collectors:
            self._voltage_form = dfx.fem.form(var.f_1.phi_s_cc * d.s_c)
        else:
            self._voltage_form = dfx.fem.form(var.f_1.phi_s * d.s_c)
        self._current_form = dfx.fem.form(cell.area * var.f_1.lm_app * d.s_c)

        dx_list = [d.x_a, d.x_s, d.x_c]
        self._c_e_forms = ([dfx.fem.form(var.f_1.c_e * dx) for dx in dx_list], dx_list)
        self._phi_e_forms = ([dfx.fem.form(var.f_1.phi_e * dx) for dx in dx_list], dx_list)

        if cell.has_collectors:
            header = [f"phi_s_{domain}_avg [V]" for domain in ['ncc', 'a', 'c', 'pcc']]
            dx_list = [d.x_ncc, d.x_a, d.x_c, d.x_pcc]
            self._phi_s_forms = [var.phi_s_cc * d.x_ncc, var.phi_s * d.x_a,
                                 var.phi_s * d.x_c, var.phi_s_cc * d.x_pcc]
            self._phi_s_forms = ([dfx.fem.form(form) for form in self._phi_s_forms], dx_list)

        else:
            header = [f"phi_s_{domain}_avg [V]" for domain in ['a', 'c']]
            dx_list = [d.x_a, d.x_c]
            self._phi_s_forms = ([dfx.fem.form(var.f_1.phi_s * dx) for dx in dx_list], dx_list)
        warehouse._outputs_info['globals']['phi_s_avg']['header'] = header

        dx_list = [d.x_a, d.x_c]
        self._i_Li_int_forms = ([dfx.fem.form(var.j_Li_a_term.int * d.x_a),
                                 dfx.fem.form(var.j_Li_c_term.int * d.x_c)], dx_list)
        self._i_Li_total_forms = ([dfx.fem.form(var.j_Li_a_term.total * d.x_a),
                                   dfx.fem.form(var.j_Li_c_term.total * d.x_c)], dx_list)

        # Prepare internal variables
        porous_components = [component for component in cell._components_.values()
                             if component.type == 'porous']
        warehouse.setup_internal_variable('c_e', {component.tag: var.c_e
                                                  for component in porous_components})
        warehouse.setup_internal_variable('phi_e', {component.tag: var.phi_e
                                                    for component in porous_components})
        if cell.has_collectors:
            warehouse.setup_internal_variable('phi_s', {
                'anode': var.phi_s,
                'cathode': var.phi_s,
                'negativeCC': var.phi_s_cc,
                'positiveCC': var.phi_s_cc
            })
        else:
            warehouse.setup_internal_variable('phi_s', {
                'anode': var.phi_s,
                'cathode': var.phi_s
            })
        warehouse.setup_internal_variable('i_Li_int', {
            'anode': var.j_Li_a_term.int,
            'cathode': var.j_Li_c_term.int
        })
        warehouse.setup_internal_variable('i_Li_total', {
            'anode': var.j_Li_a_term.total,
            'cathode': var.j_Li_c_term.total
        })

        # Reaction variables
        for component in cell._components_.values():
            if not component.name == 'electrode':  # or not component.is_active
                continue
            label = component.label
            n_mat = component.n_mat
            warehouse.setup_internal_variable(f'j_Li_int_{label}', var(f'j_Li_{label}').int,
                                              length=n_mat)
            warehouse.setup_internal_variable(f'j_Li_total_{label}', var(f'j_Li_{label}').total,
                                              length=n_mat)
            warehouse.setup_internal_variable(f'overpotential_{label}',
                                              var(f'overpotential_{label}'), length=n_mat)

        if cell.has_collectors:
            warehouse.setup_internal_variable('electric_current', {
                'anode': var.electric_current_a,
                'cathode': var.electric_current_c,
                'negativeCC': var.electric_current_ncc,
                'positiveCC': var.electric_current_pcc
            })
        else:
            warehouse.setup_internal_variable('electric_current', {
                'anode': var.electric_current_a,
                'cathode': var.electric_current_c
            })

        ionic_list = ['ionic_current', 'li_ion_flux',
                      'li_ion_flux_migration', 'li_ion_flux_diffusion']
        for output in ionic_list:
            out_dict = dict()
            for component in porous_components:
                out_dict[component.tag] = var(f'{output}_{component.label}')
            warehouse.setup_internal_variable(output, out_dict)

    def get_voltage(self):
        return self.problem.get_avg(self._voltage_form, self.mesher.ds_c)

    def get_current(self):
        return self.problem.get_avg(self._current_form, self.mesher.ds_c)

    def get_capacity(self):
        return self.Q_out

    def get_c_e_avg(self):
        return [self.problem.get_avg(form, dx) for form, dx in zip(*self._c_e_forms)]

    def get_phi_e_avg(self):
        return [self.problem.get_avg(form, dx) for form, dx in zip(*self._phi_e_forms)]

    def get_phi_s_avg(self):
        return [self.problem.get_avg(form, dx) for form, dx in zip(*self._phi_s_forms)]

    def get_i_Li_int_avg(self):
        return [self.problem.get_avg(form, dx) for form, dx in zip(*self._i_Li_int_forms)]

    def get_i_Li_total_avg(self):
        return [self.problem.get_avg(form, dx) for form, dx in zip(*self._i_Li_total_forms)]

    def get_cell_state(self, cell_state: OrderedDict, problem) -> None:
        """
        This method updates the cell state dictionary with the current
        cell state variables of this specific model.

        Parameters
        ----------
        cell_state: OrderedDict
            Dictionary containing the current cell state variables
        problem: Problem
            Object that handles the battery cell simulation.
        """
        cell_state['voltage'] = self.get_voltage()
        cell_state['current'] = self.get_current()
