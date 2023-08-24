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
from petsc4py.PETSc import ScalarType
from collections import namedtuple

from cideMOD.helpers.logging import VerbosityLevel, _print
from cideMOD.numerics.fem_handler import BlockFunction, _evaluate_parameter, _max, _min
from cideMOD.numerics.time_scheme import TimeScheme
from cideMOD.cell.parser import CellParser
from cideMOD.cell.components import BatteryCell, BaseCellComponent
from cideMOD.cell.variables import ProblemVariables
from cideMOD.cell.dimensional_analysis import DimensionalAnalysis
from cideMOD.models.PXD.base_model import BasePXDModelPreprocessing


class ElectrochemicalModelPreprocessing(BasePXDModelPreprocessing):
    """
    Base mixin class that contains the mandatory methods to be overrided
    related to the preprocessing of the model inputs.
    """
    # ******************************************************************************************* #
    # ***                                       Problem                                       *** #
    # ******************************************************************************************* #

    def set_state_variables(self, state_vars: list, mesher, V, V_vec, problem) -> None:
        """
        This method sets the state variables of the electrochemical
        model.

        Parameters
        ----------
        state_vars : List(Tuple(str, numpy.ndarray, dolfinx.fem.FunctionSpace))
            List of tuples, each one containing the name, the
            subdomain and the function space of the state variable.

        mesher : BaseMesher
            Object that contains the mesh information.

        V : dolfinx.fem.FunctionSpace
            Common FunctionSpace to be used for each model.

        V_vec : dolfinx.fem.VectorFunctionSpace
            Common VectorFunctionSpace to be used for each model.

        Examples
        --------
        >>> res = mesher.get_restrictions()
        >>> state_vars.append(('new_var', res.electrolyte, V.clone()))
        """
        res = mesher.get_restrictions()

        state_vars.append(('c_e', V.clone(), res.electrolyte))
        state_vars.append(('phi_e', V.clone(), res.electrolyte))
        state_vars.append(('phi_s', V.clone(), res.electrodes))

        if problem.cell_parser.has_collectors:
            state_vars.append(('phi_s_cc', V.clone(), res.current_collectors))
            state_vars.append(('lm_phi_s', V.clone(), res.electrode_cc_facets))
        state_vars.append(('lm_app', V.clone(), res.positive_tab))

        state_vars.extend([(f'j_Li_a{i}', V.clone(), res.anode)
                           for i in range(problem.cell_parser.anode.n_mat)])
        state_vars.extend([(f'j_Li_c{i}', V.clone(), res.cathode)
                           for i in range(problem.cell_parser.cathode.n_mat)])

        state_var_names = [var_name for var_name, _, _ in state_vars]
        self._state_vars = state_var_names[state_var_names.index('c_e'):]  # To be used later

    def set_problem_variables(self, var: ProblemVariables, DT: TimeScheme, problem) -> None:
        """
        This method sets the problem variables of the electrochemical
        model.

        Parameters
        ----------
        var: ProblemVariables
            Object that store the preprocessed problem variables.
        DT: TimeScheme
            Object that provide the temporal derivatives with the
            specified scheme.
        problem: Problem
            Object that handles the battery cell simulation.

        Notes
        -----
        This method is called within :class:`ProblemVariables` right
        after setting up the state variables and before the
        :class:`BatteryCell` is created. In this class is meant to
        create the control variables and those ones that will help
        cell parameters preprocessing.
        """

        # Temperature
        if 'temp' not in problem._f_1.var_names:
            var.temp = problem.T_ini

        # Number of active materials
        var.n_mat_a = problem.cell_parser.anode.n_mat
        var.n_mat_c = problem.cell_parser.cathode.n_mat

        # Control variables
        var.i_app = dfx.fem.Constant(problem.mesher.mesh, ScalarType(0))
        var.switch = dfx.fem.Constant(problem.mesher.mesh, ScalarType(0))
        var.v_app = dfx.fem.Constant(problem.mesher.mesh, ScalarType(0))

    def set_dependent_variables(self, var: ProblemVariables,
                                cell: BatteryCell, DT: TimeScheme, problem):
        """
        This method sets the dependent variables of the electrochemical
        model.

        Parameters
        ----------
        var: ProblemVariables
            Object that store the preprocessed problem variables.
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
        DT: TimeScheme
            Object that provide the temporal derivatives with the
            specified scheme.
        problem: Problem
            Object that handles the battery cell simulation.
        """

        a, s, c, e = cell.anode, cell.separator, cell.cathode, cell.electrolyte
        if cell.has_collectors:
            ncc, pcc = cell.negativeCC, cell.positiveCC

        # Overpotential
        var.overpotential_a = []
        for i, am in enumerate(a.active_materials):
            x_am = var.x_a_surf[i]
            ocv_a = am.U(x_am, var.i_app)
            if am.delta_S_was_provided:
                ocv_a += am.delta_S(x_am, var.i_app) * (var.temp - am.U_T_ref)
            var.overpotential_a.append(var.phi_s - var.phi_e - ocv_a)

        var.overpotential_c = []
        for i, am in enumerate(c.active_materials):
            x_am = var.x_c_surf[i]
            ocv_c = am.U(x_am, -var.i_app)
            if am.delta_S_was_provided:
                ocv_c += am.delta_S(x_am, -var.i_app) * (var.temp - am.U_T_ref)
            var.overpotential_c.append(var.phi_s - var.phi_e - ocv_c)

        # build j_Li
        var._J_Li = namedtuple('J_Li', ['total', 'int', 'LLI', 'C_dl', 'total_0'])
        for electrode in [a, c]:
            j_Li = var._J_Li._make([list() for _ in range(len(var._J_Li._fields))])
            setattr(var, f'j_Li_{electrode.label}', j_Li)
            for idx, am in enumerate(electrode.active_materials):
                # Intercalation/Deintercalation
                j_Li.int.append(var.f_1(f'j_Li_{electrode.label}{idx}'))
                # Double layer capacitance
                j_Li.C_dl.append(
                    electrode.C_dl * DT.dt(var.f_0.phi_s - var.f_0.phi_e, var.phi_s - var.phi_e)
                    if electrode.C_dl else 0)
                # Lost of Lithium Inventory
                j_Li.LLI.append(0)
                # Total Li flux
                j_Li.total_0.append(j_Li.int[idx] + j_Li.LLI[idx])  # To be used inside wf 0
                j_Li.total.append(j_Li.int[idx] + j_Li.LLI[idx] + j_Li.C_dl[idx])

        # build j_Li term = sum(j_Li * a_s)
        for electrode, j_Li in zip([a, c], [var.j_Li_a, var.j_Li_c]):
            j_Li_term = []
            for ff, field in enumerate(j_Li._fields):
                j_Li_term.append(0)
                for i, am in enumerate(electrode.active_materials):
                    j_Li_term[ff] += j_Li._asdict()[field][i] * am.a_s
            setattr(var, f'j_Li_{electrode.label}_term', var._J_Li._make(j_Li_term))

        # Ionic_current
        var.ionic_current_a = (
            - a.kappa * a.grad(var.phi_e)
            + 2 * a.kappa * cell.R * var.temp / cell.F * (1 - e.t_p)
            * e.activity * a.grad(var.c_e) / var.c_e
        )
        var.ionic_current_s = (
            - s.kappa * s.grad(var.phi_e)
            + 2 * s.kappa * cell.R * var.temp / cell.F * (1 - e.t_p)
            * e.activity * s.grad(var.c_e) / var.c_e
        )
        var.ionic_current_c = (
            - c.kappa * c.grad(var.phi_e)
            + 2 * c.kappa * cell.R * var.temp / cell.F * (1 - e.t_p)
            * e.activity * c.grad(var.c_e) / var.c_e
        )

        # Electronic_current
        var.electric_current_a = - a.sigma * a.grad(var.phi_s)
        var.electric_current_c = - c.sigma * c.grad(var.phi_s)
        if cell.has_collectors:
            var.electric_current_ncc = - ncc.sigma * ncc.grad(var.phi_s_cc)
            var.electric_current_pcc = - pcc.sigma * pcc.grad(var.phi_s_cc)

        # Li_ion_flux
        var.li_ion_flux_diffusion_a = - a.D_e * a.grad(var.c_e)
        var.li_ion_flux_diffusion_c = - c.D_e * c.grad(var.c_e)
        var.li_ion_flux_diffusion_s = - s.D_e * s.grad(var.c_e)

        var.li_ion_flux_migration_a = e.t_p / cell.F * var.ionic_current_a
        var.li_ion_flux_migration_c = e.t_p / cell.F * var.ionic_current_c
        var.li_ion_flux_migration_s = e.t_p / cell.F * var.ionic_current_s

        var.li_ion_flux_a = var.li_ion_flux_diffusion_a + var.li_ion_flux_migration_a
        var.li_ion_flux_c = var.li_ion_flux_diffusion_c + var.li_ion_flux_migration_c
        var.li_ion_flux_s = var.li_ion_flux_diffusion_s + var.li_ion_flux_migration_s

    def initial_guess(self, f: BlockFunction, var: ProblemVariables,
                      cell: BatteryCell, problem) -> None:
        """
        This method initializes the state variables based on the initial
        conditions and assuming that the simulation begins after a
        stationary state.

        Parameters
        ----------
        f: BlockFunction
            Block function that contain the state variables to be
            initialized.
        var: ProblemVariables
            Object that store the preprocessed problem variables.
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        P1_map = problem.P1_map

        # c_e initial
        c_e_ini = cell.electrolyte.c_e_ini
        P1_map.interpolate(
            {'anode': c_e_ini, 'separator': c_e_ini, 'cathode': c_e_ini}, f.c_e)

        # phi_s initial
        # - First OCV of each material is calculated according with their initial concentrations
        U_a_ini = [am.ref_U([_evaluate_parameter(am.c_s_ini / am.c_s_max)])
                   for am in cell.anode.active_materials]
        U_c_ini = [am.ref_U([_evaluate_parameter(am.c_s_ini / am.c_s_max)])
                   for am in cell.cathode.active_materials]

        # - Then the largest or lowest is selected to avoid overcharge/underdischarge
        if round(problem.SoC_ini) == 1:
            phi_s_a = max(U_a_ini)[0] if U_a_ini else 0
            phi_s_c = min(U_c_ini)[0] if U_c_ini else 0
        else:
            phi_s_a = min(U_a_ini)[0] if U_a_ini else 0
            phi_s_c = max(U_c_ini)[0] if U_c_ini else 0

        # - Finally the values are incorporated in the Function
        # NOTE: phi_s_a = phi_s_ncc = 0 (already initialized to 0)
        P1_map.interpolate({'cathode': phi_s_c - phi_s_a}, f.phi_s)
        if cell.has_collectors:
            P1_map.interpolate({'positiveCC': phi_s_c - phi_s_a}, f.phi_s_cc)

        # NOTE: phi_s, phi_e and j_Li will be adjusted when solving the stationary problem

        # Update control variables to stationary problem
        var.switch.value = 0
        var.i_app.value = 0
        var.v_app.value = 0  # self.get_voltage()

    def setup(self, problem):
        """
        This method setup the electrochemical model.

        Parameters
        ----------
        problem: Problem
            Object that handles the battery cell simulation.
        """
        self.Q_out = 0

    def update_control_variables(self, var: ProblemVariables, problem, i_app=30.0, v_app=None):
        """
        This method updates the control variables of the electrochemical
        model. Either CC and CV are supported. Varying current and
        voltage is supported via an expression dependant of the time.

        Parameters
        ----------
        var: ProblemVariables
            Object that store the preprocessed problem variables.
        problem: Problem
            Object that handles the battery cell simulation.
        i_app : Union[float,str], optional
            The applied current in Amperes. If CV use None.
            Default to 30.
        v_app : Union[float,str], optional
            The applied voltage in Volts. If CC use None.
            Default to None.
        """
        # TODO: Improve the efficiency of this, just peform the checks once for each
        #       Problem.solve call
        if i_app is not None:
            if not isinstance(i_app, (str, int, float)):
                raise TypeError("'i_app' must be one of (str, int, float, None)")
            elif isinstance(i_app, str):
                raise NotImplementedError("'i_app' cannot be an expression yet")
                # i_app = 0  # Check how to proceed if the I varies...
            else:
                i_app = i_app

        if v_app is not None:
            if not isinstance(v_app, (str, int, float)):
                raise TypeError("'v_app' must be one of (str, int, float, None)")
            elif isinstance(v_app, str):
                raise NotImplementedError("'v_app' cannot be an expression yet")
                # v_app = 0
                # self.v_0 = problem.get_voltage()
            else:
                v_app = v_app

        self._running_mode(var, i_app, v_app)

    def _running_mode(self, var, i_app, v_app):
        if None not in (i_app, v_app):
            raise ValueError("Can only input either an applied current 'i_app' or an applied "
                             + "voltage 'v_app', but not both")
        elif i_app is None and v_app is None:
            raise ValueError("Need to input either an applied current 'i_app' or an applied "
                             + "voltage 'v_app'.")
        elif i_app is None:
            self._set_voltage(var, v_app)
        else:
            self._set_current(var, i_app)

    def _set_voltage(self, var, v_app):
        """
        Set voltage in voltios (V)

        Parameters
        ----------
        v_app : float, optional
            Voltage in voltios (V)
        """
        var.switch.value = 1
        var.v_app.value = v_app

    def _set_current(self, var, i_app):
        """
        Set current in Amperes (A)

        Parameters
        ----------
        i : float
            Current in Amperes (A)
        capacity: float
            Cell capacity
        """
        var.switch.value = 0
        var.i_app.value = i_app

    # ******************************************************************************************* #
    # ***                                 DimensionalAnalysis                                 *** #
    # ******************************************************************************************* #

    # ******************************************************************************************* #
    # ***                                     BatteryCell                                     *** #
    # ******************************************************************************************* #

    def _set_component_parameters(self, component: BaseCellComponent, problem) -> None:
        """
        This method preprocesses the geometric parameters of the cell
        component.

        Parameters
        ----------
        component: BaseCellComponent
            Object where component parameters are preprocessed and
            stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        mesher = problem.mesher

        component.L = component.parser.thickness.get_value(problem)
        component.H = component.parser.height.get_value(problem)
        component.W = component.parser.width.get_value(problem)
        component.area = component.parser.area.get_value(problem)

        component.grad = mesher.get_component_gradient(component.tag, component.L,
                                                       component.H, component.W,
                                                       problem.model_options.dimensionless)

        # d = problem.mesher.get_measures()
        # for name, measure in d._asdict().items():
        #     if len(name) > 2 and component._label_ in name[2:]:
        #         setattr(component, f'd{name}', measure)

    def _set_porous_component_parameters(self, component: BaseCellComponent, problem) -> None:
        """
        This method preprocesses the common parameters of the porous
        components.

        Parameters
        ----------
        component: BaseCellComponent
            Object where porous component parameters are preprocessed
            and stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        parser = component.parser
        cell = component.cell
        electrolyte = cell.electrolyte
        T = problem._vars.temp
        vars_dic = {'t_p': electrolyte.t_p, 'T_0': problem.T_ini, 'temp': T}
        # NOTE: 'temp' is added just in case the temperature is not a state variable.

        # Preprocess common porous component parameters
        self._set_component_parameters(component, problem)

        component.eps_e = parser.porosity.get_value(problem)
        component.bruggeman = parser.bruggeman.get_value(problem)
        component.tortuosity_e = parser.tortuosity_e.get_value(problem)
        component.tortuosity_s = parser.tortuosity_s.get_value(problem)

        component.D_e = parser.D_e.get_value(
            problem, eps=component.eps_e, brug=component.bruggeman,
            tau=component.tortuosity_e, R=cell.R, **vars_dic)

        component.kappa = parser.kappa.get_value(
            problem, eps=component.eps_e, brug=component.bruggeman,
            tau=component.tortuosity_e, R=cell.R, **vars_dic)

        parser.kappa_D.set_value((- 2 * cell.R / cell.F) * (1 - electrolyte.t_p)
                                 * T * component.kappa * electrolyte.activity)
        component.kappa_D = parser.kappa_D.get_value()

    def set_cell_parameters(self, cell: BatteryCell, problem) -> None:
        """
        This method preprocesses the cell parameters of the
        electrochemical model.

        Parameters
        ----------
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        # NOTE: A flag that was created when setting up CellParser components is added here.
        cell.has_collectors = cell.parser.has_collectors

        # Constants
        cell.R = cell.parser.R.get_value(problem)
        cell.F = cell.parser.F.get_value(problem)
        cell.C_dl_cc = cell.parser.doubleLayerCapacitance_cc.get_value(problem)

    def set_electrode_parameters(self, electrode: BaseCellComponent, problem) -> None:
        """
        This method preprocesses the electrode parameters of the
        electrochemical model.

        Parameters
        ----------
        electrode: BaseCellComponent
            Object where electrode parameters are preprocessed and
            stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        parser = electrode.parser
        cell = electrode.cell
        electrolyte = cell.electrolyte
        T = problem._vars.temp
        vars_dic = {'t_p': electrolyte.t_p, 'T_0': problem.T_ini, 'temp': T}
        # NOTE: 'temp' is added just in case the temperature is not a state variable.

        # Preprocess electrode parameters
        self._set_porous_component_parameters(electrode, problem)

        electrode.type = parser.type.get_value(problem)
        electrode.rho = parser.density.get_value(problem)
        electrode.C_dl = parser.double_layer_capacitance.get_value(problem)

        eps_s = sum([am.volume_fraction.get_value(problem) for am in parser.active_materials])
        electrode.sigma = parser.electronic_conductivity.get_value(
            problem, eps=eps_s, brug=1.5, tau=electrode.tortuosity_s,
            R=cell.R, **vars_dic)

    def set_active_material_parameters(self, am: BaseCellComponent, problem) -> None:
        """
        This method preprocesses the active material parameters of the
        electrochemical model.

        Parameters
        ----------
        am: BaseCellComponent
            Object where active material parameters are preprocessed and
            stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        T = problem._vars.temp
        SoC_ini = problem.SoC_ini
        bruggeman = 1.5
        c_e_ini = am.cell.electrolyte.c_e_ini

        am.index = am.parser.index
        am.R_s = am.parser.particle_radius.get_value(problem)
        am.eps_s = am.parser.volume_fraction.get_value(problem)
        am.porosity = am.parser.porosity.get_value(problem)

        vars_dic = {'y': problem._vars(f'x_{am.electrode.label}_surf')[am.index], 'temp': T}
        am.k_0 = am.parser.kinetic_constant.get_value(problem, R=am.cell.R, **vars_dic)
        am.alpha = am.parser.alpha.get_value(problem)
        am.c_s_max = am.parser.maximum_concentration.get_value(problem)
        am.stoichiometry0 = am.parser.stoichiometry0.get_value(problem)
        am.stoichiometry1 = am.parser.stoichiometry1.get_value(problem)

        am.c_s_ini = am.c_s_max * (SoC_ini * (am.stoichiometry1 - am.stoichiometry0)
                                   + am.stoichiometry0)

        am.parser.a_s.set_value(3. * am.eps_s / am.R_s)
        if not am.parser.tortuosity_s.was_provided:
            am.parser.tortuosity_s.set_value(am.eps_s ** (1 - bruggeman))
        am.a_s = am.parser.a_s.get_value()
        am.tortuosity_s = am.parser.tortuosity_s.get_value()

        am.U = am.parser.ocp.get_value(problem, temp=T)
        am.U_T_ref = am.parser.ocp.T_ref  # NOTE: It could be None
        am.ref_U = am.parser.ocp.get_reference_value(
            temp=problem.T_ini, c_e=c_e_ini)

        am.delta_S = am.parser.entropy_coefficient.get_value(problem, temp=T)
        am.delta_S_was_provided = am.parser.entropy_coefficient.was_provided
        am.ref_delta_S = am.parser.entropy_coefficient.get_reference_value(
            temp=problem.T_ini, c_e=c_e_ini)

    def set_separator_parameters(self, separator, problem) -> None:
        """
        This method preprocesses the separator parameters of the
        electrochemical model.

        Parameters
        ----------
        separator: BaseCellComponent
            Object where separator parameters are preprocessed and
            stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        self._set_porous_component_parameters(separator, problem)
        separator.type = separator.parser.type.get_value()
        separator.rho = separator.parser.density.get_value(problem)

    def set_current_collector_parameters(self, cc, problem) -> None:
        """
        This method preprocesses the current collector parameters of the
        electrochemical model.

        Parameters
        ----------
        cc: BaseCellComponent
            Object where current collector parameters are preprocessed
            and stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        self._set_component_parameters(cc, problem)
        cc.type = cc.parser.type.get_value()
        cc.rho = cc.parser.density.get_value(problem)
        cc.sigma = cc.parser.electronic_conductivity.get_value(problem)

    def set_electrolyte_parameters(self, electrolyte, problem) -> None:
        """
        This method preprocesses the electrolyte parameters of the
        electrochemical model.

        Parameters
        ----------
        electrolyte: BaseCellComponent
            Object where electrolyte parameters are preprocessed and
            stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        electrolyte.type = electrolyte.parser.type.get_value()
        electrolyte.c_e_ini = electrolyte.parser.initial_concentration.get_value(problem)
        electrolyte.t_p = electrolyte.parser.transference_number.get_value(problem)

        # Build nonlinear properties
        # NOTE: 'temp' is added just in case the temperature is not a state variable.
        vars_dic = {'t_p': electrolyte.t_p, 'T_0': problem.T_ini, 'temp': problem._vars.temp}
        electrolyte.D_e = electrolyte.parser.diffusion_constant.get_value(
            problem, R=electrolyte.cell.R, **vars_dic)
        electrolyte.kappa = electrolyte.parser.ionic_conductivity.get_value(
            problem, R=electrolyte.cell.R, **vars_dic)
        electrolyte.activity = electrolyte.parser.activity_dependence.get_value(
            problem, **vars_dic)

    def compute_cell_properties(self, cell: BatteryCell):
        """
        This method computes the general cell properties of the
        electrochemical model.

        Parameters
        ----------
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.

        Notes
        -----
        This method is called once the cell parameters has been
        preprocessed.
        """
        cell.anode.capacity = self._get_electrode_capacity(cell.anode)
        cell.cathode.capacity = self._get_electrode_capacity(cell.cathode)
        cell.capacity = _min(cell.anode.capacity or 9e99, cell.cathode.capacity or 9e99)

        cell.area = _min(cell.anode.area or 9e99, cell.cathode.area or 9e99)

        components = [v for k, v in cell._components_.items() if k != 'electrolyte']
        if any([element.H for element in components]):
            cell.H = _max([element.H for element in components if element.H])
        else:
            cell.H = None

        if any([element.W for element in components]):
            cell.W = _max([element.W for element in components if element.W])
        else:
            cell.W = None

    def update_reference_values(self, updated_values: dict,
                                cell_parser: CellParser, problem=None) -> None:
        """
        This method updates the reference cell cell properties of the
        electrochemical model.

        Parameters
        ----------
        updated_values: Dict[str, float]
            Dictionary containing the cell parameters that have already
            been updated.
        cell_parser: CellParser
            Parser of the cell dictionary.
        problem: Problem, optional
            Object that handles the battery cell simulation.
        Notes
        -----
        This method is called each time a set of dynamic parameters have
        been updated. If problem is not given, then it is assumed that
        it have not been already defined.
        """
        # <*> Update reference cell properties
        # NOTE: Update cell properties, no matter which dynamic parameters have been updated.
        #       In case this update operations are really slow, then perform some checks before.
        # NOTE: Here we know this parameters do not need further preprocessing like arrhenius or
        #       bruggeman and thats why we use the values provided by the user directly.
        if problem is None or not problem._ready:
            self.compute_reference_cell_properties(cell_parser)
            return

        cell = problem.cell
        DA = problem._DA

        cell_parser.anode.ref_capacity = _evaluate_parameter(cell.anode.capacity)
        cell_parser.cathode.ref_capacity = _evaluate_parameter(cell.cathode.capacity)
        cell_parser.ref_capacity = min(cell_parser.anode.ref_capacity or 9e99,
                                       cell_parser.cathode.ref_capacity or 9e99)

        if cell_parser.verbose >= VerbosityLevel.BASIC_PROBLEM_INFO:
            _print(f"Negative electrode capacity: {cell_parser.anode.ref_capacity:.6f}",
                   comm=cell_parser._comm)
            _print(f"Positive electrode capacity: {cell_parser.cathode.ref_capacity :.6f}",
                   comm=cell_parser._comm)
            _print(f"Cell capacity: {cell_parser.ref_capacity:.6f}", comm=cell_parser._comm)

        # NOTE: If problem is not given, then it is assumed that it have not been already defined.
        cell_parser.ref_area = _evaluate_parameter(cell.area)
        cell_parser.ref_height = _evaluate_parameter(cell.H) if cell.H is not None else None
        cell_parser.ref_width = _evaluate_parameter(cell.W) if cell.W is not None else None

        # <*> Update dimensionless numbers reference values
        # NOTE: In order to do this, take into account that there could be parameters that need
        #       further preproccessing. In addition, this reference parameters are included in the
        #       equations, so if they can vary, then they should have been defined as
        #       dolfinx.fem.Constant, despite the fact they are reference values.
        # NOTE: What you have to do here is to recompute the reference dependent parameters that
        #       are not associated to a CellParameter (as these will be updated each time
        #       CellParameter.get_reference_value is called).
        if DA.dimensionless:
            raise NotImplementedError(
                "Update dimensionless numbers reference values not implemented yet")

    def reset(self, problem, new_parameters=None, deep_reset=False) -> None:
        """
        This method resets the problem variables related with the
        electrochemical model in order to be ready for running another
        simulation with the same initial conditions, and maybe using
        different parameters.

        Parameters
        ----------
        problem: Problem
            Object that handles the battery cell simulation.
        new_parameters: Dict[str, float], optional
            Dictionary containing the cell parameters that have already
            been updated.
        deep_reset: bool
            Whether or not a deep reset will be performed. It means
            that the Problem setup stage will be run again as the mesh
            has been changed. Default to False.
        """
        # Reset control variables if they have been already defined
        if problem._ready and not deep_reset:
            var = problem._vars
            var.i_app.value = 0
            var.v_app.value = 0
            var.switch.value = 0

        # Reset internal variables
        if deep_reset:
            self._T_ext = None
            self._T_ini = None
        else:
            self.Q_out = 0.

    def _get_electrode_capacity(self, electrode):
        cap = 0
        for am in electrode.active_materials:
            cap += (am.eps_s * am.porosity * am.c_s_max
                    * abs(am.stoichiometry1 - am.stoichiometry0) / 3600)
            for inc in am.inclusions:
                cap += (am.eps_s * inc.eps_s * inc.c_s_max * inc.porosity
                        * abs(inc.stoichiometry1 - inc.stoichiometry0) / 3600)
        return cap * electrode.area * electrode.L * electrode.cell.F
