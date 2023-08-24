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
from typing import Tuple

from cideMOD.numerics.fem_handler import BlockFunction


class ProblemVariables:
    """
    This class is responsible for storing the information of each
    problem variable. They will be used in the pre-processing, equations
    and post-processing modules. It allows models to create variables
    and modify those created by other models.
    """

    def __init__(self, problem):

        # Independent variables
        self.time = problem._time
        # self.x = problem.mesher.get_spatial_coordinate()

    def __call__(self, variable):
        return getattr(self, variable)

    def setup(self, problem):
        """
        This method sets up the control and state variables of each
        model.

        Parameters
        ----------
        problem: Problem
            Object that handles the battery cell simulation.
        """

        # State variables
        state_vars = problem._f_0.var_names
        self.f_0 = self.StateVariables(state_vars, problem._f_0.functions, problem._DA)
        self.f_1 = self.StateVariables(state_vars, problem._f_1.functions, problem._DA)
        self.test = problem.test
        for var_name, var_value in self.f_1.items():
            setattr(self, var_name, var_value)

        # Control variables and more
        problem._models.set_problem_variables(self, problem._DT, problem)

    class StateVariables(BlockFunction):

        def __init__(self, var_names: Tuple[str], functions: list, DA) -> None:
            # NOTE: Assume that the given functions are dimensionless
            self.N = len(var_names)
            self.functions = []
            self.var_names = var_names + tuple(f"{name}_" for name in var_names)
            for i, name in enumerate(self.var_names):
                if i >= self.N:
                    # Set non dimensional values
                    value = functions[i - self.N]
                else:
                    # Set dimensional values
                    value = DA.unscale_variable(name, functions[i])
                self.functions.append(value)
                setattr(self, name, value)
