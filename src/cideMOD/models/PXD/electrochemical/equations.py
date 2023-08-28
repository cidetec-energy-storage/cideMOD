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
from ufl import conditional, inner, lt, exp, sinh
from petsc4py.PETSc import ScalarType

from cideMOD.numerics.fem_handler import BlockFunctionSpace
from cideMOD.numerics.time_scheme import TimeScheme
from cideMOD.cell.components import BatteryCell
from cideMOD.cell.equations import ProblemEquations
from cideMOD.cell.variables import ProblemVariables
from cideMOD.mesh.base_mesher import BaseMesher
from cideMOD.models.PXD.base_model import BasePXDModelEquations


class ElectrochemicalModelEquations(BasePXDModelEquations):

    def get_solvers_info(self, solvers_info, problem) -> None:
        """
        This method get the solvers information that concerns the
        electrochemical model.

        Parameters
        ----------
        solvers_info: dict
            Dictionary containing solvers information.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        # TODO: Implement the stationary equations
        solvers_info['solver']['state_variables'].extend(self._state_vars)
        solvers_info['solver_transitory']['state_variables'].extend(self._state_vars[1:])
        # solvers_info['solver_stationary']['state_variables'].extend(self._state_vars[1:])

    def build_weak_formulation(self, eq: ProblemEquations, var: ProblemVariables,
                               cell: BatteryCell, mesher: BaseMesher, DT: TimeScheme,
                               W: BlockFunctionSpace, problem) -> None:
        """
        This method builds the weak formulation of the electrochemical
        model.

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

        # TODO: probably move this to preprocessing?
        if cell.has_collectors:
            cell.negativeCC.sigma_ratio = (problem.get_avg(cell.anode.sigma, d['x_a'])
                                           / cell.negativeCC.sigma)
            cell.positiveCC.sigma_ratio = (problem.get_avg(cell.cathode.sigma, d['x_c'])
                                           / cell.positiveCC.sigma)

        self.F_c_e = 0
        self.F_phi_e, self.F_phi_s = 0, 0
        self.F_j_Li_a, self.F_j_Li_c = [], []

        # Boundary condition of phi_s
        F_phi_s_bc = self.phi_s_bc(var, cell, d['s_c'])

        # phi_s. Lagrange multiplier
        if cell.has_collectors:
            self.F_lm_phi_s = self.phi_s_continuity(var, dS_el=d['S_a_ncc'], dS_cc=d['S_ncc_a'])
            self.F_lm_phi_s += self.phi_s_continuity(var, dS_el=d['S_c_pcc'], dS_cc=d['S_pcc_c'])
            self.F_phi_s_cc = -F_phi_s_bc
        else:
            self.F_phi_s -= F_phi_s_bc

        for component in cell._components_.values():
            label = component.label
            if component.type == 'porous':
                self.F_c_e += self.c_e_equation(var, d[f'x_{label}'], DT, cell, component)
                self.F_phi_e += self.phi_e_equation(var, d[f'x_{label}'], component)
                if label in ('a', 'c'):  # component.is_active
                    self.F_phi_s += self.phi_s_equation(var, d, cell, component)
                    for i, material in enumerate(component.active_materials):
                        j_Li_am_form = self.j_Li_equation(var, d[f'x_{label}'], cell, material)
                        if label == 'a':
                            self.F_j_Li_a.append(j_Li_am_form)
                        else:
                            self.F_j_Li_c.append(j_Li_am_form)
            elif component.name == 'current_collector':
                self.F_phi_s_cc += self.phi_s_equation(var, d, cell, component)

        F_c_e_continuity = 0
        # for dS in [d.S_a_s, d.S_s_c]:
        #     F_c_e_continuity += c_e_continuity(c_e=var.c_e, test=var.test.c_e, dS=dS,
        #                                        n=mesher.normal, D_e=cell._D_e, L=cell._L)
        self.F_c_e += F_c_e_continuity

        F_phi_e_continuity = 0
        # for dS in [d.S_a_s, d.S_s_c]:
        #     F_phi_e_continuity += self.phi_e_continuity(phi_e=var.phi_e, c_e=var.c_e,
        #                                                 test=var.test.phi_e, dS=dS,
        #                                                 n=mesher.normal, kappa=var._kappa,
        #                                                 kappa_D=var._kappa_D, L=var._L)
        self.F_phi_e += F_phi_e_continuity

        # Boundary Conditions
        # - anode: Dirichlet
        # TODO: Move this preprocessing steps to the mesh module
        ntab_tag = mesher.field_data['negative_plug']
        negative_tab = mesher.boundaries.indices[mesher.boundaries.values == ntab_tag]
        if 'ncc' in cell.structure:
            negative_tab_dofs = dfx.fem.locate_dofs_topological(
                W('phi_s_cc'), mesher.boundaries.dim, negative_tab)
            self.bcs_ntab = dfx.fem.dirichletbc(dfx.fem.Constant(mesher.mesh, ScalarType(0)),
                                                negative_tab_dofs, W('phi_s_cc'))
        else:
            negative_tab_dofs = dfx.fem.locate_dofs_topological(
                W('phi_s'), mesher.boundaries.dim, negative_tab)
            self.bcs_ntab = dfx.fem.dirichletbc(dfx.fem.Constant(mesher.mesh, ScalarType(0)),
                                                negative_tab_dofs, W('phi_s'))

        # - cathode: Neumann or Dirichlet via Lagrange multiplier
        phi_s = var.phi_s_cc if cell.has_collectors else var.phi_s
        self.F_lm_app = (
            (1 - var.switch) * (var.lm_app - var.i_app / cell.area)
            * var.test.lm_app * d['s_c']
            + var.switch * (phi_s - var.v_app) * var.test.lm_app * d['s_c']
        )

        # Add the problem equations
        # for state_var in ('c_e', 'phi_e', 'phi_s'):
        #     F_state_var = getattr(self, f'F_{state_var}')
        #     if F_state_var:
        #         eq.add(state_var, F_state_var)
        eq.add('c_e', self.F_c_e)
        eq.add('phi_e', self.F_phi_e)
        eq.add('phi_s', self.F_phi_s)
        if cell.has_collectors:
            eq.add('phi_s_cc', self.F_phi_s_cc)
            eq.add('lm_phi_s', self.F_lm_phi_s)
        eq.add('lm_app', self.F_lm_app)

        for i in range(var.n_mat_a):
            eq.add(f'j_Li_a{i}', self.F_j_Li_a[i])

        for i in range(var.n_mat_c):
            eq.add(f'j_Li_c{i}', self.F_j_Li_c[i])

        # Add Dirichlet boundary conditions
        if cell.has_collectors:
            eq.add_boundary_conditions('phi_s_cc', self.bcs_ntab)
        else:
            eq.add_boundary_conditions('phi_s', self.bcs_ntab)

    def build_weak_formulation_transitory(
        self, eq: ProblemEquations, var: ProblemVariables, cell: BatteryCell,
        mesher: BaseMesher, W: BlockFunctionSpace, problem
    ):
        """
        This method builds and adds the weak formulation of the
        electrochemical model that will be used to solve the transitory
        problem.

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
        # Add the equations of the transitory problem
        eq.add('phi_e', self.F_phi_e)
        eq.add('phi_s', self.F_phi_s)
        if cell.has_collectors:
            eq.add('phi_s_cc', self.F_phi_s_cc)
            eq.add('lm_phi_s', self.F_lm_phi_s)
        eq.add('lm_app', self.F_lm_app)

        for i in range(var.n_mat_a):
            eq.add(f'j_Li_a{i}', self.F_j_Li_a[i])

        for i in range(var.n_mat_c):
            eq.add(f'j_Li_c{i}', self.F_j_Li_c[i])

        # Add Dirchlet boundary conditions
        if cell.has_collectors:
            eq.add_boundary_conditions('phi_s_cc', self.bcs_ntab)
        else:
            eq.add_boundary_conditions('phi_s', self.bcs_ntab)

    def build_weak_formulation_stationary(
        self, eq: ProblemEquations, var: ProblemVariables, cell: BatteryCell,
        mesher: BaseMesher, W: BlockFunctionSpace, problem
    ):
        """
        This method builds and adds the weak formulation of the
        electrochemical model that will be used to solve the stationary
        problem.

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

        d = mesher.get_measures()

        # c_e_0
        # F_c_e_0 = ((var.c_e - var.f_0.c_e) * var.test.c_e * d.x_a
        #            + (var.c_e - var.f_0.c_e) * var.test.c_e * d.x_s
        #            + (var.c_e - var.f_0.c_e) * var.test.c_e * d.x_c)

        # FIXME: Assumes var.f_1.c_e has been initialized to a constant value througout the
        #        whole domain.

        # Add the problem equations
        # eq.add('c_e', F_c_e_0)
        eq.add('phi_e', self.F_phi_e)
        eq.add('phi_s', self.F_phi_s)
        if cell.has_collectors:
            eq.add('phi_s_cc', self.F_phi_s_cc)
            eq.add('lm_phi_s', self.F_lm_phi_s)
        eq.add('lm_app', self.F_lm_app)

        for i in range(var.n_mat_a):
            eq.add(f'j_Li_a{i}', self.F_j_Li_a[i])

        for i in range(var.n_mat_c):
            eq.add(f'j_Li_c{i}', self.F_j_Li_c[i])

        # Add Dirchlet boundary conditions
        if cell.has_collectors:
            eq.add_boundary_conditions('phi_s_cc', self.bcs_ntab)
        else:
            eq.add_boundary_conditions('phi_s', self.bcs_ntab)

    def phi_e_equation(self, var, dx, component, scale_factor=1):
        """
        Implements variational formulation equation for electrolyte potential phi_e

        Parameters
        ----------
        var: ProblemVariables
            Object that store the preprocessed problem variables.
            Contains:

            * phi_e : dolfinx.fem.Function - Electrolyte Potential
            * test.phi_e : TestFunction - Electrolyte Potential Equation
              Test Function
            * j_Li : dolfinx.fem.Function - Intercalation reaction Flux
              (Optional)
            * c_e : dolfinx.fem.Function - Electrolyte Concentration

        dx: Measures
            Measure of the domain over the integral must integrate
        component: BaseCellComponent
            Object where component parameters are preprocessed and
            stored. Contains:

            * kappa : Constant or similar - Effective electric
              conductivity of the electrolyte
            * kappa_D : Constant or similar - Effective concentration
              induced conductivity (More or less)
            * grad : dolfinx.fem.Function - python function that returns
              the UFL gradient of the argument
            * L : Constant - Thickness used to normalize the domain, by
              default None

        Returns
        -------
        Form
            Electrolyte Potential Equation
        """
        test = var.test.phi_e
        domain_grad = component.grad
        label = component._label_
        j_Li = var(f"j_Li_{label}_term").total if label != 's' else None

        if dx.subdomain_id() in dx.subdomain_data().values:
            F_phi = (scale_factor * component.L * component.kappa
                     * inner(domain_grad(var.phi_e), domain_grad(test))
                     * dx(metadata={"quadrature_degree": 0})
                     + (scale_factor * component.L * component.kappa_D / var.c_e)
                     * inner(domain_grad(var.c_e), domain_grad(test))
                     * dx(metadata={"quadrature_degree": 2}))
            if j_Li is not None:
                F_phi -= (scale_factor * component.L * j_Li * test
                          * dx(metadata={"quadrature_degree": 2}))
            return F_phi
        else:
            return 0

    # def phi_e_continuity(self, phi_e, c_e, test, dS, n, kappa, kappa_D, L):
    #     weak_form = 0
    #     for res in ['-', '+']:
    #         weak_form += (kappa(res) / L(res) * inner(n(res), grad(phi_e(res))) * test(res) * dS
    #                     + kappa_D(res) / L(res) / c_e(res) * inner(n(res), grad(c_e(res)))
    #                     * test(res) * dS)
    #     return weak_form

    def phi_s_equation(self, var, d, cell, component):
        """
        Implements variational formulation for electrode potential phi_s

        Parameters
        ----------
        var : ProblemVariables
            Object that store the preprocessed problem variables.
            Contains:

            * phi_s : dolfinx.fem.Function - Electrode Potential
            * j_Li : dolfinx.fem.Function - Intercalation reaction Flux
            * test : TestFunction - Electrode Potential Equation Test
              Function
            * lm_phi_s : dolfinx.fem.Function - Lagrange multiplier
              (Optional)

        d : Measures
            * dx : Measure - Measure of the domain over the integral
              must integrate
            * dS : Measure - Measure of the boundary domain over the
              integral must integrate

        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.

        component: BaseCellComponent
            Object where component parameters are preprocessed and
            stored. Contains:

            * sigma : Constant or similar - Effective electric
              conductivity of the electrode material
            * grad : dolfinx.fem.Function - python function that returns
              the UFL gradient of the argument
            * L : Constant - Thickness used to normalize the domain, by
              default None

        Returns
        -------
        Form
            Electrode Potential Equation
        """

        label = component.label
        phi_s, test, j_Li, scale_factor = (
            (var.phi_s, var.test.phi_s, var(f'j_Li_{label}_term').total, 1) if label in ('a', 'c')
            else (var.phi_s_cc, var.test.phi_s_cc, None, component.sigma_ratio))

        dx = d[f'x_{label}']
        domain_grad = component.grad

        # TODO: check whether this 'if' is actually needed
        if dx.subdomain_id() in dx.subdomain_data().values:
            F_phi_s = (scale_factor * component.L * component.sigma
                       * inner(domain_grad(phi_s), domain_grad(test)) * dx)
            # Adds source term if there is any
            if j_Li is not None and j_Li != 0:  # if j_Li:
                F_phi_s += scale_factor * component.L * j_Li * test * dx
            if cell.has_collectors:
                if label in ('a', 'c'):
                    dS = d['S_a_ncc'] if label == 'a' else d['S_c_pcc']
                    F_phi_s += self.phi_s_interface(var.lm_phi_s, dS,
                                                    phi_s_test=test, scale_factor=scale_factor)
                elif label in ('ncc', 'pcc'):
                    dS = d['S_ncc_a'] if label == 'ncc' else d['S_pcc_c']
                    F_phi_s -= self.phi_s_interface(var.lm_phi_s, dS,
                                                    phi_s_cc_test=test,
                                                    scale_factor=scale_factor)
                else:
                    raise ValueError("Invalid interface condition")
            return F_phi_s
        else:
            return 0

    def phi_s_bc(self, var, cell, ds):
        """
        Implements boundary conditions for electrode potential equation phi_s

        Parameters
        ----------
        var : ProblemVariables
            Object that store the preprocessed problem variables. Contains:

            * lm_app : Constant, Function or similar - Current applied
              to the boundary of the electrode
            * test : TestFunction - Electrode Potential Equation Test
              Function

        cell : BatteryCell
            Object where cell parameters are preprocessed and stored.
        ds : Measure
            Measure of the boundary domain over the integral must integrate

        Returns
        -------
        Form
            Electrode Potential boundary condition Equation
        """
        test = var.test.phi_s_cc if cell.has_collectors else var.test.phi_s
        scale_factor = cell.positiveCC.sigma_ratio if cell.has_collectors else 1
        return scale_factor * var.lm_app * test * ds

    def phi_s_interface(self, lagrange_multiplier, dS, phi_s_test=None,
                        phi_s_cc_test=None, scale_factor=1):
        """
        Implements the interface for electrode potential phi_s

        Parameters
        ----------
        lagrange_multiplier : dolfinx.fem.Function
            Lagrange multiplier on the interface between the elctrode
            and the cc associated to the continuity of the potential
            in the solid face
        dS : Measure
            Measure of the boundary domain over the integral must integrate
        phi_s_test : TestFunction
            Electrode Potential Equation Test Function
        phi_s_cc_test : TestFunction
            Electrode Potential and cc Equation Test Function

        Returns
        -------
        Form
            Electrode Potential interface Equation
        """
        int_dir = dS.metadata()['direction']
        if phi_s_test is not None:
            interface_bc = scale_factor * lagrange_multiplier(int_dir) * phi_s_test(int_dir) * dS
        elif phi_s_cc_test is not None:
            interface_bc = (scale_factor * lagrange_multiplier(int_dir)
                            * phi_s_cc_test(int_dir) * dS)
        else:
            raise Exception("Invalid interface condition")
        return interface_bc

    def phi_s_continuity(self, var, dS_el, dS_cc):
        """
        Implements the continuity equation for electrode potential phi_s

        Parameters
        ----------
        phi_s_electrode: dolfinx.fem.Function
            Electrode Potential Equation
        phi_s_cc : dolfinx.fem.Function
            Current Collector Potential Equation
        lm_test : TestFunction
            Test Function
        dS_el : Measure
            Measure of the boundary domain over the integral must
            integrate in the electrolyte
        dS_cc : Measure
            Measure of the boundary domain over the integral must
            integrate in the cc

        Returns
        -------
        Form
            Potential Continuity Equation
        """
        el_dir = dS_el.metadata()['direction']
        cc_dir = dS_cc.metadata()['direction']
        lm_test = var.test.lm_phi_s
        return (var.phi_s(el_dir) * lm_test(el_dir) * dS_el
                - var.phi_s_cc(cc_dir) * lm_test(cc_dir) * dS_cc)

    def c_e_equation(self, var, dx, DT, cell, component, scale_factor=1):
        """
        Implements variational formulation for electrolyte concentration
        c_e

        Parameters
        ----------
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
        var: ProblemVariables
            Object that store the preprocessed problem variables.
            Requires:

            * c_e_0 : dolfinx.fem.Function - Electrode Potential last
              timestep
            * c_e : dolfinx.fem.Function - Electrode Potential
            * test : TestFunction - Electrode Potential Equation Test
              Function
            * j_Li : dolfinx.fem.Function - Intercalation reaction Flux

        d : Measures
            Measure of the domain over the integral must integrate
        DT : TimeScheme
            Instance of the TimeScheme class
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
            Requires:

            * F : Constant or similar - Faraday Constant
            * electrolyte.t_p : Constant or similar - Transference
              number for the electrolyte

        component: BaseCellComponent
            Object where component parameters are preprocessed and
            stored. Contains:

            * D_e : Constant or similar - Effective Diffusivity of the
              electrolyte
            * eps_e : Constant or similar - Volume fraction occupied by
              the electrolyte in the domain
            * grad : dolfinx.fem.Function - python function that returns
              the UFL gradient of the argument
            * L : Constant - Thickness used to normalize the domain, by
              default None

        Returns
        -------
        Form
            Electrolyte Concentration Equation
        """
        test = var.test.c_e
        domain_grad = component.grad
        label = component.label
        j_Li = var(f'j_Li_{label}_term').total if label != 's' else None

        if dx.subdomain_id() in dx.subdomain_data().values:
            F_c = ((scale_factor * component.L * component.eps_e)
                   * DT.dt(var.f_0.c_e, var.c_e) * test * dx
                   + (scale_factor * component.L * component.D_e)
                   * inner(domain_grad(var.c_e), domain_grad(test)) * dx)
            if j_Li is not None:
                F_c -= (scale_factor * component.L * (1 - cell.electrolyte.t_p) / cell.F) * j_Li * test * dx
            return F_c
        else:
            return 0

    # def c_e_continuity(self, c_e, test, dS, n, D_e, L, scale_factor=1):
    #     weak_form = 0
    #     for res in ['-', '+']:
    #         weak_form += (scale_factor / L(res) * D_e(res) * inner(n(res), grad(c_e(res)))
    #                       * test(res) * dS)
    #     return weak_form

    def i_n_equation(self, k, c_e, c_s, c_s_max, alpha):
        """
        Implements the interface for electrode potential phi_s

        Parameters
        ----------
        lagrange_multiplier : dolfinx.fem.Function
            Lagrange multiplier on the interface between the elctrode
            and the cc associated to the continuity of the potential
            in the solid face
        k : Constant or similar
            Kinetic constant of the active material
        c_e : dolfinx.fem.Function
            Electrolyte concentration
        c_s : dolfinx.fem.Function
            Electrode concentration
        c_s_max : TestFunction
            Maximum Electrode concentration
        alpha : Constant or similar
            Charge transfer coefficient

        Returns
        -------
        ufl.Operator
            Regularisation of i_0
        """

        f_c_e, f_c_s, f_c_s_max = 1, 1, 1
        regularization = (exp(-f_c_s / c_s**(1 / alpha)) * exp(-f_c_e / c_e**(1 / (1 - alpha)))
                          * exp(-f_c_s_max / (c_s_max - c_s)**(1 / (1 - alpha))))
        i_0 = k * c_e**(1 - alpha) * c_s**alpha * (c_s_max - c_s)**(1 - alpha)
        i_n = conditional(
            lt(c_e, 0),
            0,
            conditional(lt(c_s, 0), 0, conditional(lt(c_s_max - c_s, 0), 0, i_0 * regularization)))
        return i_n

    def ButlerVolmer_equation(self, alpha, F, R, T, eta):
        """
        Implements the Butler-Volmer equation

        Parameters
        ----------
        alpha : Constant or similar
            Charge transfer coefficient
        F : Constant or similar
            Farady's constant
        R : Constant or similar
            Universal Gas constant
        T : Constant or similar
            Absolute temperature
        eta : dolfinx.fem.Function
            Activation overpotential

        Returns
        -------
        ufl.Operator
            Right hand side
        """
        return exp((1 - alpha) * F / (R * T) * eta) - exp(-alpha * F / (R * T) * eta)
        # return 2 * sinh((alpha * F / R) * eta / T)

    def j_Li_equation(self, var, dx, cell, material):
        """
        Exchange between the electrolyte and the electrode by lithium intercalation

        Parameters
        ----------
        var: ProblemVariables
            Object that store the preprocessed problem variables.
            Requires:
        d : Measures
            Measure of the domain over the integral must integrate
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
        material: BaseCellComponent
            Object where active material parameters are preprocessed and
            stored.

        Returns
        -------
        Form
            Butler-Volmer equation of the specified material
        """

        label = material.electrode.label
        idx = material.index
        j_Li, j_Li_test = var.f_1(f'j_Li_{label}{idx}'), var.test(f'j_Li_{label}{idx}')
        overpotential = var(f'overpotential_{label}')
        c_s_surf = var(f'c_s_{label}_surf')

        i_n = self.i_n_equation(material.k_0, var.c_e, c_s_surf[idx],
                                material.c_s_max, material.alpha)
        BV = self.ButlerVolmer_equation(material.alpha, cell.F, cell.R, var.temp,
                                        overpotential[idx])
        j_li = cell.F * i_n * BV
        F_j_Li = (j_Li - j_li) * j_Li_test * dx
        return F_j_Li

    def explicit_update(self, problem) -> None:
        """
        This method updates some stuff after the implicit timestep is
        performed.

        Parameters
        ----------
        problem: Problem
            Object that handles the battery cell simulation.
        """
        # TODO: Create ModelHandler.advance_problem and move this
        if 'capacity' in problem._WH._requested_outputs['globals']:
            self.Q_out -= self.get_current() * self.problem.get_timestep() / 3600
