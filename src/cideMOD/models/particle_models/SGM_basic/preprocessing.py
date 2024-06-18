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
import numpy as np
from numpy.polynomial.legendre import legder, legmulx, legval, legint, legmul

from cideMOD.helpers.miscellaneous import get_spline
from cideMOD.numerics.fem_handler import BlockFunction
from cideMOD.numerics.time_scheme import TimeScheme
from cideMOD.cell.components import BatteryCell, BaseCellComponent
from cideMOD.cell.variables import ProblemVariables
from cideMOD.models.base import BaseCellModelPreprocessing


class ParticleModelSGMPreprocessing(BaseCellModelPreprocessing):
    """
    Base mixin class that contains the mandatory methods to be overrided
    related to the preprocessing of the model inputs.
    """

    # ******************************************************************************************* #
    # ***                                       Problem                                       *** #
    # ******************************************************************************************* #

    def set_state_variables(self, state_vars: list, mesher, V, V_vec, problem) -> None:
        """
        This method sets the state variables of the SGM particle model.

        Parameters
        ----------
        state_vars : List(Tuple(str, numpy.ndarray, dolfinx.fem.functionspace))
            List of tuples, each one containing the name, the
            subdomain and the function space of the state variable.

        mesher : BaseMesher
            Object that contains the mesh information.

        V : dolfinx.fem.functionspace
            Common functionspace to be used for each model.

        V_vec : dolfinx.fem.VectorFunctionSpace
            Common VectorFunctionSpace to be used for each model.

        Examples
        --------
        >>> res = mesher.get_restrictions()
        >>> state_vars.append(('new_var', res.electrolyte, V.clone()))
        """
        cell = problem.cell_parser
        res = mesher.get_restrictions()
        for electrode in [cell.anode, cell.cathode]:
            electrode_res = res._asdict()[electrode.tag]
            for k in range(electrode.n_mat):
                for j in range(self.order):
                    var_name = f'c_s_{j}_{electrode.label}{k}'
                    state_vars.append((var_name, V.clone(), electrode_res))

        # To be used later:
        state_var_names = [var_name for var_name, _, _ in state_vars]
        self._state_vars = state_var_names[state_var_names.index(f'c_s_0_{cell.anode.label}0'):]

    def set_problem_variables(self, var: ProblemVariables, DT: TimeScheme, problem) -> None:
        """
        This method sets the problem variables of the SGM particle
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
        # NOTE: As we need the stoichiometry to evaluate some non linear parameters, it is
        #       mandatory to build it here. The problem is that we also need access to the
        #       maximum concentration in the active materials, usually computed inside BatteryCell.
        #       Thats why we need to setup c_s_max first as in self.set_active_material_parameters.
        #       In phase 2, there should be no problem in setting up a CellParameter in this stage,
        #       but in this phase, if we would need to preprocess the parameter it must be done
        #       explicitly here.

        # c_s surface
        var.x_a_surf = []
        var.c_s_a_surf = []
        for am_idx, am in enumerate(problem.cell_parser.anode.active_materials):
            # NOTE: No further preprocessing is needed
            c_s_max = am.maximum_concentration.get_value(problem)
            c_s_am_surf = var.f_1(f'c_s_0_a{am_idx}')
            x_am = c_s_am_surf / c_s_max
            var.c_s_a_surf.append(c_s_am_surf)
            var.x_a_surf.append(x_am)

        var.x_c_surf = []
        var.c_s_c_surf = []
        for am_idx, am in enumerate(problem.cell_parser.cathode.active_materials):
            c_s_max = am.maximum_concentration.get_value(problem)
            c_s_am_surf = var.f_1(f'c_s_0_c{am_idx}')
            x_am = c_s_am_surf / c_s_max
            var.c_s_c_surf.append(c_s_am_surf)
            var.x_c_surf.append(x_am)

        # c_s r-averaged
        weights = self._leg_volume_integral()

        var.x_a_avg = []
        var.c_s_a_avg = []
        for am_idx, am in enumerate(problem.cell_parser.anode.active_materials):
            c_s_max = am.maximum_concentration.get_value(problem)
            c_s_am_avg = weights[0] * var.f_1(f'c_s_0_a{am_idx}')
            for i in range(1, self.order):
                c_s_am_avg += (weights[i] - weights[0]) * var.f_1(f'c_s_{i}_a{am_idx}')
            # NOTE: Already divided by the particle volume
            var.c_s_a_avg.append(3 * c_s_am_avg)
            var.x_a_avg.append(3 * c_s_am_avg / c_s_max)

        var.x_c_avg = []
        var.c_s_c_avg = []
        for am_idx, am in enumerate(problem.cell_parser.cathode.active_materials):
            c_s_max = am.maximum_concentration.get_value(problem)
            c_s_am_avg = weights[0] * var.f_1(f'c_s_0_c{am_idx}')
            for i in range(1, self.order):
                c_s_am_avg += (weights[i] - weights[0]) * var.f_1(f'c_s_{i}_c{am_idx}')
            # NOTE: Already divided by the particle volume
            var.c_s_c_avg.append(3 * c_s_am_avg)
            var.x_c_avg.append(3 * c_s_am_avg / c_s_max)

    def _leg_volume_integral(self):
        weights = np.zeros(self.order)
        for n in range(self.order):
            L_n = np.zeros(2 * self.order, dtype=int)
            L_n[2 * n] = 1  # Only pair polinomials used
            L_nxx = legmulx(legmulx(L_n))  # r*r*L
            weights[n] = legval(1.0, legint(L_nxx))  # integral(0, 1, r^2*L_n)
        return weights

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

        # c_s initial
        # NOTE: The first coefficient represents the average value in the particle
        for electrode in [cell.anode, cell.cathode]:
            for k, am in enumerate(electrode.active_materials):
                c_s_surf = f(f'c_s_0_{electrode.label}{k}')  # = c_s_0 as the others are set to 0
                P1_map.interpolate({electrode.tag: am.c_s_ini}, c_s_surf)

    def _build_legendre(self):
        """
        Builds mass matrix, stiffness matrix and boundary vector using
        Legendre Polinomials. The domain used is [0,1] and only pair
        Legendre polinomials are used to enforce zero flux at x=0.

        Args:
            order (int): number of Legendre polinomials to use

        Returns:
            tuple: Mass matrix, Stiffness matrix, boundary vector
        """
        # Init matrix and vector
        M = np.zeros((self.order, self.order))
        K = np.zeros((self.order, self.order))
        P = np.zeros(self.order)
        for n in range(self.order):
            L_n = np.zeros(2 * self.order, dtype=int)
            L_n[2 * n] = 1  # Only pair polinomials used

            D_n = legder(L_n)  # dL/dr
            L_nx = legmulx(L_n)  # r*L
            D_nx = legmulx(D_n)  # r*dL/dr

            P[n] = legval(1.0, L_n)  # L(1)

            for m in range(self.order):
                L_m = np.zeros(2 * self.order, dtype=int)
                L_m[2 * m] = 1

                D_m = legder(L_m)
                L_mx = legmulx(L_m)
                D_mx = legmulx(D_m)

                # integral(0, 1, r^2*L_n*L_m)
                M[n, m] = legval(1.0, legint(legmul(L_nx, L_mx)))
                # integral(0, 1, r^2*dL_n/dr*dL_m/dr)
                K[n, m] = legval(1.0, legint(legmul(D_nx, D_mx)))

        self.M = M
        self.K = K
        self.P = P

    # ******************************************************************************************* #
    # ***                                     BatteryCell                                     *** #
    # ******************************************************************************************* #
    def set_active_material_parameters(self, am: BaseCellComponent, problem) -> None:
        """
        This method preprocesses the active material parameters of the
        SGM particle model.

        Parameters
        ----------
        am: BaseCellComponent
            Object where active material parameters are preprocessed and
            stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        D_s = am.parser.diffusion_constant
        if 'inc' in am.tag and D_s.dtype == 'expression' and 'y' in D_s.user_value:
            raise NotImplementedError

        var = problem._vars
        x_surf = var.x_a_surf if am.electrode_tag == 'anode' else var.x_c_surf
        vars_dic = {'y': x_surf[am.index], 'temp': var.temp}

        am.D_s = D_s.get_value(problem, R=am.cell.R, **vars_dic)
