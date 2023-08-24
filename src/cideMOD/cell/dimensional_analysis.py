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

from cideMOD.cell.parser import CellParser


class DimensionalAnalysis:
    """
    This class performs the dimensional analysis. First obtains the
    reference parameters and then compute the dimensionless numbers that
    arise from the dimensional analysis. It also allows to scale and
    unscale some variables.

    Parameters
    ----------
    cell: CellParser
        Parser of the cell dictionary.
    model_options: BaseModelOptions
        Object containing the simulation options.
    """

    def __init__(self, cell: CellParser, model_options) -> None:

        self.dimensionless = False  # FIXME: model_options.dimensionless
        self._models = model_options._get_model_handler()

        # Perform the dimensional analysis
        if self.dimensionless:
            self._models.build_dimensional_analysis(self, cell)

    def scale_variable(self, name: str, value):
        """
        This method scales the given variable.

        Parameters
        ----------
        name: str
            Name of the variable to be scaled.
        value: Any
            Value to be scaled.

        Returns
        -------
        Any
            Scaled value of the variable.

        Examples
        --------
        >>> models.scale_variable('c_e', 1000)
        0
        """

        return self.models.scale_variable(name, value) if self.dimensionless else value

    def unscale_variable(self, name: str, value):
        """
        This method unscales the given variable.

        Parameters
        ----------
        name: str
            Name of the variable to be unscaled.
        value: Any
            Value to be unscaled.

        Returns
        -------
        Any
            Unscaled value of the variable.

        Examples
        --------
        >>> models.unscale_variable('c_e', 0)
        1000
        """
        return self.models.unscale_variable(name, value) if self.dimensionless else value

    def scale_variables(self, variables: dict):
        """
        This method scales the given variables.

        Parameters
        ----------
        variables: Dict[str, Any]
            Dictionary containing the names and the values of the
            variables to be scaled.

        Returns
        -------
        dict
            Dictionary containing the scaled variables.

        Examples
        --------
        >>> variables = {'c_e': 1000, 'c_s_a': 28700}
        >>> models.scale_variables(variables)
        {'c_e': 0, 'c_s_a': 1}
        """
        return self.models.scale_variables(variables) if self.dimensionless else variables

    def unscale_variables(self, variables: dict):
        """
        This method unscales the given variables.

        Parameters
        ----------
        variables: Dict[str, Any]
            Dictionary containing the names and the values of the
            variables to be unscaled.

        Returns
        -------
        dict
            Dictionary containing the unscaled variables.

        Examples
        --------
        >>> variables = {'c_e': 0, 'c_s_a': 1}
        >>> models.unscale_variables(variables)
        {'c_e': 1000, 'c_s_a': 28700}
        """
        return self.models.scale_variables(variables) if self.dimensionless else variables
