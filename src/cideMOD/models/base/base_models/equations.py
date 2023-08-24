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
from abc import ABC, abstractmethod
from cideMOD.numerics.fem_handler import BlockFunctionSpace
from cideMOD.numerics.time_scheme import TimeScheme
from cideMOD.cell.components import BatteryCell
from cideMOD.cell.equations import ProblemEquations
from cideMOD.cell.variables import ProblemVariables
from cideMOD.mesh.base_mesher import BaseMesher


class BaseCellModelEquations(ABC):
    """
    Base mixin class that contains the mandatory methods to be overrided
    related to the implementation of the model equations
    """

    @abstractmethod
    def get_solvers_info(self, solvers_info, problem) -> None:
        """
        This method get the solvers information that concerns this
        specific model.

        Parameters
        ----------
        solvers_info: dict
            Dictionary containing solvers information.
        problem: Problem
            Object that handles the battery cell simulation.

        Examples
        --------
        To add new state variables:

        >>> state_vars = ['var1', 'var2', 'var3']
        >>> solvers_info['solver']['state_variables'].extend(state_vars)

        To change the solver options:

        >>> options = {...}
        >>> solvers_info['solver_stationary']['options'].update(options)
        """

    @abstractmethod
    def build_weak_formulation(self, equations: ProblemEquations, var: ProblemVariables,
                               cell: BatteryCell, mesher: BaseMesher, DT: TimeScheme,
                               W: BlockFunctionSpace, problem) -> None:
        """
        This method builds the weak formulation of this specific model.

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
        # TODO: Extend this documentation with examples explaining how to add equations and bcs.

    def build_weak_formulation_transitory(
        self, equations: ProblemEquations, var: ProblemVariables, cell: BatteryCell,
        mesher: BaseMesher, W: BlockFunctionSpace, problem
    ) -> None:
        """
        This method builds the weak formulation of this specific model
        that will be used to solve the transitory problem.

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
        # TODO: Extend this documentation with examples explaining how to add equations and bcs.

    def build_weak_formulation_stationary(
        self, equations: ProblemEquations, var: ProblemVariables, cell: BatteryCell,
        mesher: BaseMesher, W: BlockFunctionSpace, problem
    ) -> None:
        """
        This method builds the weak formulation of this specific model
        that will be used to solve the stationary problem.

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
        # TODO: Extend this documentation with examples explaining how to add equations and bcs.

    def explicit_update(self, problem) -> None:
        """
        This method updates this specific explicit model (if so) once
        the implicit timestep is performed.

        Parameters
        ----------
        problem: Problem
            Object that handles the battery cell simulation.
        """
