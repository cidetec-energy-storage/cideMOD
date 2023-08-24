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
from cideMOD.numerics.fem_handler import BlockFunctionSpace
from cideMOD.numerics.time_scheme import TimeScheme
from cideMOD.cell.components import BatteryCell
from cideMOD.cell.equations import ProblemEquations
from cideMOD.cell.variables import ProblemVariables
from cideMOD.mesh.base_mesher import BaseMesher
from cideMOD.models.base.base_models import BaseCellModelEquations


class ParticleModelSGMEquations(BaseCellModelEquations):

    def get_solvers_info(self, solvers_info, problem) -> None:
        """
        This method get the solvers information that concerns the SGM
        particle model.

        Parameters
        ----------
        solvers_info: dict
            Dictionary containing solvers information.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        # TODO: Activate the stationary equations
        solvers_info['solver']['state_variables'].extend(self._state_vars)
        # solvers_info['solver_transitory']['state_variables'].extend([])
        # solvers_info['solver_stationary']['state_variables'].extend(self._state_vars)

    def build_weak_formulation(self, eq: ProblemEquations, var: ProblemVariables,
                               cell: BatteryCell, mesher: BaseMesher, DT: TimeScheme,
                               W: BlockFunctionSpace, problem) -> None:
        """
        This method builds the weak formulation of the SGM particle
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

        for component in cell._components_.values():
            if not component.name == 'electrode':  # or not component.is_active:
                continue
            label = component.label
            dx = d[f'x_{label}'](metadata={"quadrature_degree": 2})
            for am_idx, am in enumerate(component.active_materials):
                c_s_0 = var.f_1(f'c_s_0_{label}{am_idx}')
                c_s_0_prev = var.f_0(f'c_s_0_{label}{am_idx}')
                j_li = var.f_1(f'j_Li_{label}{am_idx}')
                for j in range(self.order):
                    F_c_s = 0
                    test = var.test(f'c_s_{j}_{label}{am_idx}')
                    F_c_s += self.M[0, j] * DT.dt(c_s_0_prev, c_s_0) * test * dx
                    for i in range(1, self.order):
                        c_s_i = var.f_1(f'c_s_{i}_{label}{am_idx}')
                        c_s_i_prev = var.f_0(f'c_s_{i}_{label}{am_idx}')
                        F_c_s -= self.M[0, j] * DT.dt(c_s_i_prev, c_s_i) * test * dx
                        F_c_s += self.M[i, j] * DT.dt(c_s_i_prev, c_s_i) * test * dx
                        F_c_s += (am.D_s / am.R_s ** 2) * self.K[i, j] * c_s_i * test * dx

                    F_c_s += (1. / am.R_s) * (1. / cell.F) * self.P[j] * j_li * test * dx
                    eq.add(f'c_s_{j}_{label}{am_idx}', F_c_s)

    def build_weak_formulation_transitory(
        self, eq: ProblemEquations, var: ProblemVariables, cell: BatteryCell,
        mesher: BaseMesher, W: BlockFunctionSpace, problem
    ):
        """
        This method builds and adds the weak formulation of the SGM
        particle model that will be used to solve the transitory
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
        # c_s
        # NOTE: c_s should remain the same as in the previous timestep. Thats why it is not added.

    def build_weak_formulation_stationary(
        self, eq: ProblemEquations, var: ProblemVariables, cell: BatteryCell,
        mesher: BaseMesher, W: BlockFunctionSpace, problem
    ):
        """
        This method builds and adds the weak formulation of the SGM
        particle model that will be used to solve the stationary
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
        # FIXME: Assumes var.f_0.c_s_0_{domain}{am} has been initialized to a constant value
        #        througout the whole domain.
        d = mesher.get_measures()._asdict()

        for component in cell._components_.values():
            if not component.name == 'electrode':  # or not component.is_active:
                continue
            label = component.label
            dx = d[f'x_{label}']
            for am_idx in range(component.n_mat):
                c_s_0_prev = var.f_0(f'c_s_0_{label}{am_idx}')
                c_s_0 = var.f_1(f'c_s_0_{label}{am_idx}')
                test = var.test(f'c_s_0_{label}{am_idx}')
                eq.add(f'c_s_0_{label}{am_idx}', (c_s_0 - c_s_0_prev) * test * dx)
                for j in range(1, self.order):
                    c_s_j = var.f_1(f'c_s_{j}_{label}{am_idx}')
                    test = var.test(f'c_s_{j}_{label}{am_idx}')
                    eq.add(f'c_s_{j}_{label}{am_idx}', c_s_j * test * dx)
