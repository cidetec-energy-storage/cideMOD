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
from typing import List
import dolfinx as dfx


class ProblemEquations(dict):
    """
    This class will be in charge of storing the equations of the problem
    and identifying them with the corresponding state variable.
    """

    def __init__(self, state_variables: List[str]):

        self.state_variables = state_variables
        self.bcs = dict()

        for variable in self.state_variables:
            self[variable] = None
            self.bcs[variable] = list()

    def add(self, state_variable, equation):
        """
        This method adds the equation of the specified state variable.

        Parameters
        ----------
        state_variable: str
            Name of the state variable.
        equation: dolfinx.fem.Form
            Integral form of the equation.

        Examples
        --------
        To relate a variable with its equation:

        >>> state_variables = ['var1', 'var2', 'var3']
        >>> equations = ProblemEquations(state_variables)
        >>> equations.add('var1', var1_equation)
        """
        # TODO: Implement additional checks.
        if state_variable in self.state_variables:
            self[state_variable] = equation
        else:
            raise ValueError(f"Unrecognized state variable '{state_variable}'")

    def add_boundary_conditions(self, state_variable, boundary):
        """
        This method adds the dirichlet boundary condition of the
        specified state variable.

        Parameters
        ----------
        state_variable: str
            Name of the state variable.
        boundary: dolfinx.fem.DirichletBCMetaClass
            Dirichlet boundary condition of this variable.

        Examples
        --------
        To relate a variable with its boundary condition:

        >>> equations.add_boundary_condition('var1', var1_bc)

        To access the boundary condition:

        >>> equations.bcs['var1']
        [var1_bc]

        """
        if isinstance(boundary, dfx.fem.DirichletBC):
            self.bcs[state_variable].append(boundary)
        else:
            raise TypeError(f"The boundary condition is not a DirichletBC instance")

    def get_boundary_conditions(self):
        bcs = []
        for state_var in self.state_variables:
            bc = self.bcs[state_var]
            if bc:
                bcs.extend(bc)
        return bcs

    def update():
        raise NotImplementedError

    def check(self):
        """
        Once the state variables are initialised and with the
        corresponding equations added, this method is used to check that
        all equations have been added.
        """
        for variable in self.state_variables:
            if self[variable] is None:
                raise RuntimeError(f"The '{variable}' equation has not been added yet")

    def print(self, state_variable=None):
        """
        This method print the equations and boundary conditions
        associated with each state variable.
        """
        if state_variable is None:
            for var in self.state_variables:
                self.print(var)
        elif state_variable not in self.state_variables:
            raise ValueError(f"Unrecognized state variables '{state_variable}'")
        else:
            # TODO: Replace the original names of Constants and TestFunctions
            print(f"State variable '{state_variable}'")
            print("\tequation:")
            print(f"\t\t{self[state_variable]}")
            # if self.bcs[state_variable]:
            #     print("\tboundary conditions")
            #     for bc in self.bcs[state_variable]:
            #         print(f"\t\t{bc}")
