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
from ufl import inner, grad

from cideMOD.numerics.fem_handler import BlockFunction, interpolate
from cideMOD.numerics.time_scheme import TimeScheme
from cideMOD.cell.components import BatteryCell, BaseCellComponent
from cideMOD.cell.variables import ProblemVariables
from cideMOD.models.PXD.base_model import BasePXDModelPreprocessing


class ThermalModelPreprocessing(BasePXDModelPreprocessing):
    """
    Base mixin class that contains the mandatory methods to be overrided
    related to the preprocessing of the model inputs.
    """
    # ******************************************************************************************* #
    # ***                                 DimensionalAnalysis                                 *** #
    # ******************************************************************************************* #

    # ******************************************************************************************* #
    # ***                                       Problem                                       *** #
    # ******************************************************************************************* #

    def set_state_variables(self, state_vars: list, mesher, V, V_vec, problem) -> None:
        """
        This method sets the state variables of the thermal model.

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
        state_vars.append(('temp', V.clone(), res.cell))
        self._state_vars = ['temp']

    def set_problem_variables(self, var: ProblemVariables, DT: TimeScheme, problem) -> None:
        """
        This method sets the problem variables of the thermal model.

        Parameters
        ----------
        var: ProblemVariables
            Object that store the preprocessed problem variables.
        DT: TimeScheme
            Object that provide the temporal derivatives with the
            specified scheme.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        # TODO: Allow the user to vary T_ext
        var.T_ext = problem.T_ext

    def set_dependent_variables(self, var: ProblemVariables,
                                cell: BatteryCell, DT: TimeScheme, problem):
        """
        This method sets the dependent variables of the thermal model.

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
        m_factor = 1e-3
        domains = [v for k, v in cell._components_.items() if k != 'electrolyte']

        # Electrolyte ohmic heat
        for domain in domains:
            label = domain.label
            # Heat flux
            setattr(var, f'heat_flux_{label}',
                    -m_factor * domain.L * domain.k_t * domain.grad(var.temp))
            if domain.type == 'porous':
                # Electrolyte ohmic heat
                q_ohm_e_kappa = domain.L * domain.kappa * inner(grad(var.phi_e), grad(var.phi_e))
                q_ohm_e_kappa_D = (domain.L * (domain.kappa_D / var.c_e)
                                   * inner(grad(var.c_e), grad(var.phi_e)))
                setattr(var, f'q_ohm_e_kappa_{label}', q_ohm_e_kappa)
                setattr(var, f'q_ohm_e_kappa_D_{label}', q_ohm_e_kappa_D)
                setattr(var, f'q_ohm_e_{label}', q_ohm_e_kappa + q_ohm_e_kappa_D)

                if not domain.name == 'electrode':
                    continue

                # Solid ohmic heat
                setattr(var, f'q_ohm_s_{label}',
                        domain.L * domain.sigma * inner(grad(var.phi_s), grad(var.phi_s)))
                # Reaction reversible and irreversible heat
                setattr(var, f'q_irrev_{label}', [])
                setattr(var, f'q_rev_{label}', [])
                for i, material in enumerate(domain.active_materials):
                    j_Li = var.f_1(f'j_Li_{label}{i}')
                    eta = var(f'overpotential_{label}')[i]
                    delta_s = material.delta_S(var(f'x_{label}_surf')[i], var.i_app)
                    var(f'q_irrev_{label}').append(domain.L * material.a_s * j_Li * eta)
                    var(f'q_rev_{label}').append(
                        domain.L * material.a_s * j_Li * var.temp * delta_s)
                setattr(var, f'q_irrev_{label}_total', sum(var(f'q_irrev_{label}')))
                setattr(var, f'q_rev_{label}_total', sum(var(f'q_rev_{label}')))

            elif domain.type == 'solid':  # current_collectors
                # Solid ohmic heat
                setattr(var, f'q_ohm_s_{label}',
                        domain.L * domain.sigma * inner(grad(var.phi_s_cc), grad(var.phi_s_cc)))

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
        # Temperature initial
        interpolate(problem.T_ini, f.temp)

    # ******************************************************************************************* #
    # ***                                     BatteryCell                                     *** #
    # ******************************************************************************************* #

    def set_cell_parameters(self, cell: BatteryCell, problem) -> None:
        """
        This method preprocesses the cell parameters of the thermal
        model.

        Parameters
        ----------
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        # Cell parameters
        cell.h_t = cell.parser.heat_convection.get_value(problem)
        cell.thermal_expansion_rate = cell.parser.thermal_expansion_rate.get_value(problem)

    def set_electrode_parameters(self, electrode: BaseCellComponent, problem) -> None:
        """
        This method preprocesses the electrode parameters of the thermal
        model.

        Parameters
        ----------
        electrode: BaseCellComponent
            Object where electrode parameters are preprocessed and
            stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        parser = electrode.parser
        eps_s = sum([am.volume_fraction.get_value(problem) for am in parser.active_materials])
        electrode.k_t = parser.thermal_conductivity.get_value(
            problem, eps=eps_s, brug=electrode.bruggeman, tau=electrode.tortuosity_s)
        electrode.c_p = parser.specific_heat.get_value(problem, eps=eps_s, tau=1)

    def set_separator_parameters(self, separator, problem) -> None:
        """
        This method preprocesses the separator parameters of the thermal
        model.

        Parameters
        ----------
        separator: BaseCellComponent
            Object where separator parameters are preprocessed and
            stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        parser = separator.parser
        # FIXME: Fix the thermal model by defining macroeffective parameters that take into
        #        account liquid and solid phases.
        # eps_s = 1 - separator.eps_e
        # separator.k_t = parser.thermalConductivity.get_value(
        #     problem, eps=eps_s, brug=separator.bruggeman, tau=separator.tortuosity_s)
        # separator.c_p = parser.specificHeat.get_value(
        #     problem, eps=eps_s, brug=separator.bruggeman, tau=separator.tortuosity_s)
        separator.k_t = parser.thermal_conductivity.get_value(
            problem, eps=separator.eps_e, brug=separator.bruggeman, tau=separator.tortuosity_e)
        separator.c_p = parser.specific_heat.get_value(problem, eps=separator.eps_e, tau=1)

    def set_current_collector_parameters(self, cc: BaseCellComponent, problem) -> None:
        """
        This method preprocesses the current collector parameters of the
        thermal model.

        Parameters
        ----------
        cc: BaseCellComponent
            Object where current collector parameters are preprocessed
            and stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        cc.k_t = cc.parser.thermal_conductivity.get_value(problem)
        cc.c_p = cc.parser.specific_heat.get_value(problem)

    def set_electrolyte_parameters(self, electrolyte, problem) -> None:
        """
        This method preprocesses the electrolyte parameters of the
        thermal model.

        Parameters
        ----------
        electrolyte: BaseCellComponent
            Object where electrolyte parameters are preprocessed and
            stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        electrolyte.k_t = electrolyte.parser.thermal_conductivity.get_value(problem)
        electrolyte.c_p = electrolyte.parser.specific_heat.get_value(problem)
