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
from cideMOD.mesh.base_mesher import BaseMesher
from cideMOD.cell import ProblemEquations
from cideMOD.cell import ProblemVariables
from cideMOD.models.PXD.base_model import BasePXDModelEquations


class MigrationSEIModelEquations(BasePXDModelEquations):

    def get_solvers_info(self, solvers_info, problem) -> None:
        """
        This method get the solvers information that concerns the
        compact SEI model.

        Parameters
        ----------
        solvers_info: dict
            Dictionary containing solvers information.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        solvers_info['solver']['state_variables'].extend(self._state_vars)
        solvers_info['solver_transitory']['state_variables'].extend(
            self._state_vars[:self._state_vars.index('delta_porous_sei_a0')])

    def build_weak_formulation(self, eq: ProblemEquations, var: ProblemVariables,
                               cell: BatteryCell, mesher: BaseMesher, DT: TimeScheme,
                               W: BlockFunctionSpace, problem) -> None:
        """
        This method builds the weak formulation of the compact SEI
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
        self.F_jSEI, self.F_delta_porousSEI, self.F_delta_compactSEI = [], [], []
        for am_idx in range(var.n_mat_a):
            j_sei = var.f_1(f'j_sei_a{am_idx}')
            j_sei_test = var.test(f'j_sei_a{am_idx}')
            delta_porous = var.f_1(f'delta_porous_sei_a{am_idx}')
            delta_0_porous = var.f_0(f'delta_porous_sei_a{am_idx}')
            delta_test_porous = var.test(f'delta_porous_sei_a{am_idx}')
            delta_compact = var.f_1(f'delta_compact_sei_a{am_idx}')
            delta_0_compact = var.f_0(f'delta_compact_sei_a{am_idx}')
            delta_test_compact = var.test(f'delta_compact_sei_a{am_idx}')
            self.F_delta_porousSEI.append(
                self.delta_growth(delta_0_porous, delta_porous,
                                  cell.anode.SEI.compact.f * j_sei, cell.F,
                                  cell.anode.SEI.porous.rho, cell.anode.SEI.porous.M,
                                  delta_test_porous, d.x_a, DT))
            self.F_delta_compactSEI.append(
                self.delta_growth(delta_0_compact, delta_compact,
                                  (1 - cell.anode.SEI.compact.f) * j_sei, cell.F,
                                  cell.anode.SEI.compact.rho, cell.anode.SEI.compact.M,
                                  delta_test_compact, d.x_a, DT))
            self.F_jSEI.append(self.j_SEI(j_sei, j_sei_test, d.x_a, cell.anode.SEI.compact.kappa,
                                          var.overpotential_sei[am_idx], delta_compact))

        for i in range(var.n_mat_a):
            eq.add(f'j_sei_a{i}', self.F_jSEI[i])
            eq.add(f'delta_porous_sei_a{i}', self.F_delta_porousSEI[i])
            eq.add(f'delta_compact_sei_a{i}', self.F_delta_compactSEI[i])

    def build_weak_formulation_transitory(
            self, eq: ProblemEquations, var: ProblemVariables, cell: BatteryCell, mesher:
            BaseMesher, W: BlockFunctionSpace, problem):
        """
        This method builds and adds the weak formulation of the compact
        SEI model that will be used to solve the stationary
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
        for i in range(var.n_mat_a):
            eq.add(f'j_sei_a{i}', self.F_jSEI[i])

    def build_weak_formulation_stationary(
        self, eq: ProblemEquations, var: ProblemVariables, cell: BatteryCell,
        mesher: BaseMesher, W: BlockFunctionSpace, problem
    ):
        """
        This method builds and adds the weak formulation of the compact
        SEI model that will be used to solve the stationary problem.

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

    def j_SEI(self, j_SEI, test, dx, kappa, overpotential, delta):

        return (j_SEI - kappa * overpotential / delta) * test * dx

    def delta_growth(self, delta_0, delta_1, j_SEI, F, rho, M, test, dx, DT):

        return (DT.dt(delta_0, delta_1) * test * dx
                + j_SEI * M / (2 * F * rho) * test * dx)

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
