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

from cideMOD.cell.parser import CellParser


class BaseCellModelInputs(ABC):
    """
    Base mixin class that contains the mandatory methods to be overrided
    related to the inputs of :class:`cideMOD.models.BaseCellModel`.
    """
    # ******************************************************************************************* #
    # ***                                   User interface                                    *** #
    # ******************************************************************************************* #

    @classmethod
    @abstractmethod
    def is_active_model(cls, model_options) -> bool:
        """
        This method checks the model options configured by the user to
        evaluate if this model should be added to the cell model.

        Parameters
        ----------
        model_options: BaseModelOptions
            Model options already configured by the user.

        Returns
        -------
        bool
            Whether or not this model should be added to the cell model.
        """

    # ******************************************************************************************* #
    # ***                                       Problem                                       *** #
    # ******************************************************************************************* #

    def set_cell_state(self, problem, **kwargs) -> None:
        """
        This method set the current state of the cell.

        Parameters
        ----------
        problem: Problem
            Object that handles the battery cell simulation.
        kwargs: dict
            Dictionary containing the variables that defines the cell
            state. They must be defined by the active models.

        Notes
        -----
        Notice that :class:`Problem` will call this method at least once
        to set the default values.

        Examples
        --------
        >>> problem.SoC_ini = SoC_ini
        """

    # ******************************************************************************************* #
    # ***                                     CellParser                                      *** #
    # ******************************************************************************************* #

    def build_cell_components(self, cell: CellParser) -> None:
        """
        This method builds the components of the cell that fit our model
        type, e.g. electrodes, separator, current collectors, etc.

        Parameters
        ----------
        cell: CellParser
            Parser of the cell dictionary.

        Examples
        --------
        >>> cell.set_component('anode', ElectrodeParser)

        It is also possible to create the class dinamically:

        >>> cell.set_component('anode', 'ElectrodeParser')
        """

    def parse_cell_structure(self, cell: CellParser) -> None:
        """
        This method parse the cell structure. If there are any component
        this model does not know, then this method should return the
        list of unrecognized components. Maybe this components has been
        defined by other models, so this task should be delegated to
        these model.

        Parameters
        ----------
        cell: CellParser
            Parser of the cell dictionary.

        Returns
        -------
        Optional[Union[bool, list]]
            Whether or not the cell structure is valid. If there are any
            component this model does not know, then this method should
            return the list of unrecognized components.

            If this model has not added any component, then return None.
        """

    def compute_reference_cell_properties(self, cell: CellParser) -> None:
        """
        This method computes the general reference cell properties of
        this specific model.

        Parameters
        ----------
        cell: CellParser
            Parser of the cell dictionary.

        Notes
        -----
        This method is called once the cell parameters has been parsed.
        """

    # def parse_component_parameters(self, component) -> None:
    #     """
    #     This method parses the component parameters of this specific
    #     model.

    #     Parameters
    #     ----------
    #     component: BaseComponentParser
    #         Object that parses the component parameters.

    #     Notes
    #     -----
    #     This method is just a template for the family of methods
    #     :meth:`parse_{component}_parameters`. The component name
    #     will be given by the components added during the registration
    #     process.

    #     Examples
    #     --------
    #     To parse a new component parameter call:

    #     >>> component.new_parameter = component.parse_value(
    #             'json name', default=1)
    #     """
