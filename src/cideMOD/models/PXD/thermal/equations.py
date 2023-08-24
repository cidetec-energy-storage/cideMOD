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

from ufl import inner

from cideMOD.numerics.fem_handler import BlockFunctionSpace
from cideMOD.numerics.time_scheme import TimeScheme
from cideMOD.cell.components import BatteryCell
from cideMOD.cell.equations import ProblemEquations
from cideMOD.cell.variables import ProblemVariables
from cideMOD.mesh.base_mesher import BaseMesher
from cideMOD.models.PXD.base_model import BasePXDModelEquations


class ThermalModelEquations(BasePXDModelEquations):

    def get_solvers_info(self, solvers_info, problem) -> None:
        """
        This method get the solvers information that concerns the
        thermal model.

        Parameters
        ----------
        solvers_info: dict
            Dictionary containing solvers information.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        solvers_info['solver']['state_variables'].extend(self._state_vars)
        # solvers_info['solver_transitory']['state_variables'].extend([])
        # solvers_info['solver_stationary']['state_variables'].extend([])

    def build_weak_formulation(self, eq: ProblemEquations, var: ProblemVariables,
                               cell: BatteryCell, mesher: BaseMesher, DT: TimeScheme,
                               W: BlockFunctionSpace, problem) -> None:
        """
        This method builds the weak formulation of the thermal model.

        Parameters
        ----------
        equations: ProblemEquations
            Object that contains the system of equations of the problem.
        var: ProblemVariables
            Object that store the preprocessed problem variables.
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
        mesher: BaseMesher
            Object that store the mesh information.
        DT: TimeScheme
            Object that provide the temporal derivatives with the
            specified scheme.
        W: BlockFunctionSpace
            Object that store the function space of each state variable.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        d = mesher.get_measures()._asdict()

        # Heat source
        F_temp = 0
        for component in cell._components_.values():
            label = component.label
            if component.label in ('ncc', 'a', 's', 'c', 'pcc'):  # in cell.structure:
                # Heat source
                q = self.q_equation(component, var, d[f"x_{label}"])
                # Temperature
                F_temp += self.T_equation(var, component, q, d[f"x_{label}"], DT)

        F_temp_boundary = cell.h_t * (var.temp - var.T_ext) * var.test.temp * d['s']

        F_temp_continuity = 0
        # for dS in [d.S_ncc_a, d.S_a_s, d.S_s_c, d.S_c_pcc]:
        #     F_T_continuity += T_continuity(T=var.f_1.temp, test=var.test.temp, dS=dS,
        #                                    k_t=cell._k_t, L=cell._L, n=mesher.normal)

        # TODO: Allow Dirichlet bcs. (Implement BaseBoundaryCondition and Heater first)
        self.F_temp = F_temp + F_temp_boundary + F_temp_continuity

        eq.add('temp', self.F_temp)

    def build_weak_formulation_transitory(
            self, eq: ProblemEquations, var: ProblemVariables, cell: BatteryCell,
            mesher: BaseMesher, W: BlockFunctionSpace, problem
    ):
        """
        This method builds and adds the weak formulation of the thermal
        model that will be used to solve the transitory problem.

        Parameters
        ----------
        equations: ProblemEquations
            Object that contains the system of equations of the
            transitory problem.
        var: ProblemVariables
            Object that store the preprocessed problem variables.
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
        mesher: BaseMesher
            Object that store the mesh information.
        W: BlockFunctionSpace
            Object that store the function space of each state variable.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        # NOTE: The temperature must remain the same in the transitory problem, as it is a
        #       continuous variable in time.

    def build_weak_formulation_stationary(
            self, eq: ProblemEquations, var: ProblemVariables, cell: BatteryCell,
            mesher: BaseMesher, W: BlockFunctionSpace, problem
    ):
        """
        This method builds and adds the weak formulation of the thermal
        model that will be used to solve the stationary problem.

        Parameters
        ----------
        equations: ProblemEquations
            Object that contains the system of equations of the
            stationary problem.
        var: ProblemVariables
            Object that store the preprocessed problem variables.
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
        mesher: BaseMesher
            Object that store the mesh information.
        W: BlockFunctionSpace
            Object that store the function space of each state variable.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        # FIXME: Assumes var.temp has been initialized to a constant value
        #        througout the whole domain.
        # d = mesher.get_measures()

        # # T_0
        # F_T_0 = (var.temp - var.f_0.temp) * var.test.temp * d.x

        # eq.add('temp', F_T_0)

    def electrolyte_ohmic_heat_equation(self, kappa, kappa_D, phi_e, c_e, test, dx, grad, L, eps):
        F_q = (L * kappa * inner(grad(phi_e), grad(phi_e))
               * test * dx(metadata={"quadrature_degree": 1})
               + L * (kappa_D / c_e) * inner(grad(c_e), grad(phi_e))
               * test * dx(metadata={"quadrature_degree": 2}))
        return F_q

    def solid_ohmic_heat_equation(self, sigma, phi_s, test, dx, grad, L, eps):
        F_q = (L * sigma * inner(grad(phi_s), grad(phi_s))
               * test * dx(metadata={"quadrature_degree": 1}))
        return F_q

    def reaction_irreversible_heat(self, material, j_Li, test, eta, dx, L):
        F_q = L * material.a_s * j_Li * eta * test * dx(metadata={"quadrature_degree": 3})
        return F_q

    def reaction_reversible_heat(self, material, j_Li, T, c_s, current, test, dx, L):
        delta_s = material.delta_S(c_s / material.c_s_max, current)
        F_q = L * material.a_s * j_Li * T * delta_s * test * dx(metadata={"quadrature_degree": 3})
        return F_q

    def q_equation(self, component, var, dx):
        name = component.name
        if name not in ('electrode', 'separator', 'current_collector'):
            raise ValueError(f"Invalid component '{name}' for heat source computation")
        if dx.subdomain_id() not in dx.subdomain_data().values:
            return 0
        test = var.test.temp
        q = 0
        if name in ('electrode', 'separator'):
            q += self.electrolyte_ohmic_heat_equation(
                component.kappa, component.kappa_D, var.phi_e, var.c_e,
                test, dx, component.grad, component.L, component.eps_e)
        if name == 'current_collector':
            q += self.solid_ohmic_heat_equation(component.sigma, var.phi_s_cc,
                                                test, dx, component.grad, component.L, None)
        if name == 'electrode':
            overpotential = getattr(var, f'overpotential_{component.label}')
            c_s_surf = getattr(var, f'c_s_{component.label}_surf')
            j_Li = getattr(var, f'j_Li_{component.label}').total
            current = var.i_app if component.label.startswith('a') else -var.i_app
            q += self.solid_ohmic_heat_equation(component.sigma, var.phi_s, test,
                                                dx, component.grad, component.L, component.eps_e)
            for i, material in enumerate(component.active_materials):
                q += self.reaction_irreversible_heat(material, j_Li[i], test, overpotential[i], dx,
                                                     component.L)
                q += self.reaction_reversible_heat(material, j_Li[i], var.temp, c_s_surf[i],
                                                   current, test, dx, component.L)
        return q

    def T_equation(self, var, component, q, dx, DT):
        """
        Implements variational form of Temperature Equation

        Parameters
        ----------
        var: ProblemVariables
            Object that store the preprocessed problem variables.
            Requires:

            * f_0.temp : Function - Temperature field previous timestep
            * temp : Function - Temperature
            * test.temp : TestFunction - Temperature equation
              TestFunction

        component: BaseCellComponent
            Object where component parameters are preprocessed and
            stored. Contains:

            * rho : Constant or similar - Density of the material
            * c_p : Constant or similar - Specific heat of material
            * k_t : Constant or similar - Heat Conductivity of material
            * grad : function - python function that returns the UFL
              gradient of the argument
            * L : Constant - Thickness used to normalize the domain

        q : Sum or Expression or similar
            Heat generated
        dx : Measure
            Measure of the domain over the integral must integrate
        DT : TimeScheme
            Instance of the TimeScheme class

        Returns
        -------
        Form
            Temperature Equation
        """
        # TODO: review thermal model and units to remove this x1000 factor
        if dx.subdomain_id() not in dx.subdomain_data().values:
            return 0
        grad = component.grad
        m_factor = 1e-3 if component.label.endswith('cc') else 1
        T, T0, test = var.temp, var.f_0.temp, var.test.temp
        F_t = (m_factor * component.L * component.rho * component.c_p * DT.dt(T0, T)
               * test * dx(metadata={"quadrature_degree": 2})
               + m_factor * component.L * component.k_t * inner(grad(T), grad(test))
               * dx(metadata={"quadrature_degree": 0}))
        if q is not None:
            F_t -= m_factor * q
        return F_t

    # def T_continuity(self, T, test, dS, k_t, L, n, scale=1):
    #     return (scale / L('+') * k_t('+') * inner(n('+'), grad(T('+'))) * test('+') * dS
    #             + scale / L('-') * k_t('-') * inner(n('-'), grad(T('-'))) * test('-') * dS)
