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
from ufl import exp

from cideMOD.numerics.fem_handler import BlockFunctionSpace
from cideMOD.numerics.time_scheme import TimeScheme
from cideMOD.cell.components import BatteryCell
from cideMOD.mesh.base_mesher import BaseMesher
from cideMOD.cell import ProblemEquations
from cideMOD.cell import ProblemVariables
from cideMOD.models.PXD.base_model import BasePXDModelEquations


class DiffusionSEIModelEquations(BasePXDModelEquations):

    def get_solvers_info(self, solvers_info, problem) -> None:
        """
        This method get the solvers information that concerns the
        SEI model.

        Parameters
        ----------
        solvers_info: dict
            Dictionary containing solvers information.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        solvers_info['solver']['state_variables'].extend(self._state_vars)
        solvers_info['solver_transitory']['state_variables'].extend(
            self._state_vars[:self._state_vars.index('delta_sei_a0')])

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
        d = mesher.get_measures()
        self.F_jSEI, self.F_deltaSEI = [], []
        for am_idx, am in enumerate(cell.anode.active_materials):
            j_sei = var.f_1(f'j_sei_a{am_idx}')
            j_sei_test = var.test(f'j_sei_a{am_idx}')
            delta = var.f_1(f'delta_sei_a{am_idx}')
            delta_0 = var.f_0(f'delta_sei_a{am_idx}')
            delta_test = var.test(f'delta_sei_a{am_idx}')
            c_EC = var.f_1(f'c_EC_0_a{am_idx}')
            i_0s = cell.F * c_EC * cell.anode.SEI.porous.k_f_s
            self.F_jSEI.append(
                self.j_SEI(j_sei, j_sei_test, d.x_a, i_0s, cell.anode.SEI.porous.beta,
                           var.overpotential_sei[am_idx], var.temp, cell.F, cell.R))
            self.F_deltaSEI.append(
                self.delta_growth(delta_0, delta, j_sei, cell.F, cell.anode.SEI.porous.rho,
                                  cell.anode.SEI.porous.M, delta_test, d.x_a, DT))

        self.F_EC_ret = []
        for k in range(var.n_mat_a):
            F_EC_am = []
            c_EC_prev = ([var.f_0(f'c_EC_{j}_a{k}') for j in range(self.order)]
                         + [cell.anode.SEI.porous.c_EC_sln * cell.anode.SEI.porous.eps])
            c_EC = ([var.f_1(f'c_EC_{j}_a{k}') for j in range(self.order)]
                    + [cell.anode.SEI.porous.c_EC_sln * cell.anode.SEI.porous.eps])
            c_EC_test = ([var.test(f'c_EC_{j}_a{k}') for j in range(self.order)]
                         + [cell.anode.SEI.porous.c_EC_sln * cell.anode.SEI.porous.eps])
            delta_0 = var.f_0(f'delta_sei_a{k}')
            j_SEI = var.f_1(f'j_sei_a{k}')
            delta = var.f_1(f'delta_sei_a{k}')
            self.K = (1 / delta**2 * cell.anode.SEI.porous.D_EC * self.K2
                      + 1 / delta * DT.dt(delta_0, delta) * self.K1)
            for j in range(self.order):
                F_EC = 0
                for i in range(self.order + 1):
                    F_EC += (self.D[i, j] * DT.dt(c_EC_prev[i], c_EC[i]) * c_EC_test[j]
                             * d.x_a(metadata={"quadrature_degree": 2}))
                    F_EC += (self.K[i, j] * c_EC[i] * c_EC_test[j]
                             * d.x_a(metadata={"quadrature_degree": 2}))

                F_EC -= (1 / delta * j_SEI / cell.F * self.P[j]
                         * c_EC_test[j] * d.x_a(metadata={"quadrature_degree": 2}))
                F_EC_am.append(F_EC)
            self.F_EC_ret.append(F_EC_am)

        for am_idx in range(var.n_mat_a):
            eq.add(f'j_sei_a{am_idx}', self.F_jSEI[am_idx])
            eq.add(f'delta_sei_a{am_idx}', self.F_deltaSEI[am_idx])
            for j in range(self.order):
                eq.add(f'c_EC_{j}_a{am_idx}', self.F_EC_ret[am_idx][j])

    def build_weak_formulation_transitory(
            self, eq: ProblemEquations, var: ProblemVariables, cell: BatteryCell, mesher:
            BaseMesher, W: BlockFunctionSpace, problem):
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
        for am_idx in range(var.n_mat_a):
            eq.add(f'j_sei_a{am_idx}', self.F_jSEI[am_idx])

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
        # TODO: Implement this
        # raise NotImplementedError(f"Stationary solver not implemented yet")

    def j_SEI(self, j_SEI, test, dx, i_0s, beta, overpotential, T, F, R):
        return (
            j_SEI * test * dx
            + i_0s * exp(-(beta * F / R) * overpotential / T) * test * dx
        )

    def delta_growth(self, delta_0, delta_1, j_SEI, F, rho, M, test, dx, DT):
        return (
            DT.dt(delta_0, delta_1) * test * dx
            + j_SEI * M / (2 * F * rho) * test * dx
        )

    def explicit_update(self, problem) -> None:
        """
        This method updates some stuff after the implicit timestep is
        performed.

        Parameters
        ----------
        problem: Problem
            Object that handles the battery cell simulation.
        """
        if 'Q_sei_a' in problem._WH._requested_outputs['globals']:
            Q_sei_instant = self.get_Q_sei_instant()
            for k in range(problem.cell.anode.n_mat):
                self.Q_sei[k] += Q_sei_instant[k]
