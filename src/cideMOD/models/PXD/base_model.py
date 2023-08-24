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

from abc import ABC
from cideMOD.helpers.miscellaneous import constant_expression
from cideMOD.models.base import (
    BaseCellModel,
    BaseCellModelEquations,
    BaseCellModelInputs,
    BaseCellModelOutputs,
    BaseCellModelPreprocessing
)

__all__ = [
    "BasePXDModel",
    "BasePXDModelInputs",
    "BasePXDModelOutputs",
    "BasePXDModelPreprocessing",
    "BasePXDModelEquations"
]

BasePXDModelEquations = BaseCellModelEquations
BasePXDModelOutputs = BaseCellModelOutputs


class BasePXDModelInputs(BaseCellModelInputs, ABC):

    # ******************************************************************************************* #
    # ***                                     CellParser                                      *** #
    # ******************************************************************************************* #

    def parse_cell_parameters(self, cell) -> None:
        """
        This method parses the cell parameters of this specific model.

        Parameters
        ----------
        cell: CellParser
            Parser of the cell dictionary.

        Examples
        --------
        To parse a new cell parameter call:

        >>> cell.new_parameter = cell.parse_value(
                'json name', default=1)
        """

    def parse_electrode_parameters(self, electrode) -> None:
        """
        This method parses the electrode parameters of this specific
        model.

        Parameters
        ----------
        electrode: BaseComponentParser
            Object that parses the electrode parameters.

        Examples
        --------
        To parse a new electrode parameter call:

        >>> electrode.new_parameter = electrode.parse_value(
                'json name', default=1)
        """

    def parse_active_material_parameters(self, am) -> None:
        """
        This method parses the active material parameters of this
        specific model.

        Parameters
        ----------
        am: BaseComponentParser
            Object that parses the active material parameters.

        Examples
        --------
        To parse a new active material parameter call:

        >>> am.new_parameter = am.parse_value('json name', default=1)
        """

    def parse_current_collector_parameters(self, cc) -> None:
        """
        This method parses the current collector parameters of this
        specific model.

        Parameters
        ----------
        cc: BaseComponentParser
            Object that parses the current collector parameters.

        Examples
        --------
        To parse a new current collector parameter call:

        >>> cc.new_parameter = cc.parse_value('json name', default=1)
        """

    def parse_separator_parameters(self, separator) -> None:
        """
        This method parses the separator parameters of this specific
        model.

        Parameters
        ----------
        separator: BaseComponentParser
            Object that parses the separator parameters.

        Examples
        --------
        To parse a new separator parameter call:

        >>> separator.new_parameter = separator.parse_value(
                'json name', default=1)
        """

    def parse_electrolyte_parameters(self, electrolyte) -> None:
        """
        This method parses the electrolyte parameters of this specific
        model.

        Parameters
        ----------
        electrolyte: BaseComponentParser
            Object that parses the electrolyte parameters.

        Examples
        --------
        To parse a new electrolyte parameter call:

        >>> electrolyte.new_parameter = electrolyte.parse_value(
                'json name', default=1)
        """


class BasePXDModelPreprocessing(BaseCellModelPreprocessing):

    def set_cell_parameters(self, cell, problem) -> None:
        """
        This method preprocesseses the cell parameters of this specific
        model.

        Parameters
        ----------
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """

    def set_electrode_parameters(self, electrode, problem) -> None:
        """
        This method preprocesses the electrode parameters of this
        specific model.

        Parameters
        ----------
        electrode: BaseCellComponent
            Object where electrode parameters are preprocessed and
            stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """

    def set_active_material_parameters(self, am, problem) -> None:
        """
        This method preprocesses the active material parameters of this
        specific model.

        Parameters
        ----------
        am: BaseCellComponent
            Object where active material parameters are preprocessed and
            stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """

    def set_separator_parameters(self, separator, problem) -> None:
        """
        This method preprocesses the separator parameters of this
        specific model.

        Parameters
        ----------
        separator: BaseCellComponent
            Object where separator parameters are preprocessed and
            stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """

    def set_current_collector_parameters(self, cc, problem) -> None:
        """
        This method preprocesses the current collector parameters of
        this specific model.

        Parameters
        ----------
        cc: BaseCellComponent
            Object where the current collector parameters are
            preprocessed and stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """

    def set_electrolyte_parameters(self, electrolyte, problem) -> None:
        """
        This method preprocesses the electrolyte parameters of this
        specific model.

        Parameters
        ----------
        electrolyte: BaseCellComponent
            Object where the electrolyte parameters are
            preprocessed and stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """

    def _effective_value(self, value, porosity, bruggeman=None, tortuosity=None, use="tortuosity"):
        assert value is not None
        assert porosity is not None
        assert use in ["tortuosity", "bruggeman"], "Correction must be tortuosity or bruggeman"
        if use == "bruggeman":
            assert bruggeman, "Selected bruggeman correction but not provided a value"
            return value * porosity**bruggeman
        if use == "tortuosity":
            assert tortuosity, "Selected tortuosity correction but not provided a value"
            return value * porosity / tortuosity

    def get_brug_e(self, porous_component, dic, problem, **kwargs):
        """
        Calculate Bruggeman constant for the liquid phase in the component.
        """
        return self.get_brug(porous_component, dic, problem, 'e', **kwargs)

    def get_brug_s(self, porous_component, dic, problem, **kwargs):
        """
        Calculate Bruggeman constant for the solid phase in the component.
        """
        return self.get_brug(porous_component, dic, problem, 's', **kwargs)

    def get_brug(self, porous_component, dic, problem, phase, **kwargs):
        if not isinstance(dic, dict):
            dic = {'value': dic, 'effective': True}
        if isinstance(dic["value"], str):
            x = constant_expression(dic["value"], **{**problem._vars.f_1._asdict(), **kwargs})
        else:
            x = dic["value"]
        if dic["effective"]:
            return x
        elif phase == 's':
            eps_s = sum([porous_component.active_materials[i].eps_s
                         for i in range(len(porous_component.active_materials))])
            return self._effective_value(x, eps_s, porous_component.bruggeman,
                                         porous_component.tortuosity_s, dic["correction"])
        elif phase == 'e':
            return self._effective_value(x, porous_component.eps_e, porous_component.bruggeman,
                                         porous_component.tortuosity_e, dic["correction"])
        else:
            raise ValueError(f"Unrecognized phase '{phase}'. Available options: 'e' 's'")


class BasePXDModel(
    BasePXDModelInputs,
    BasePXDModelPreprocessing,
    BaseCellModel,
    ABC
):
    """
    Abstract base class for PXD models

    Parameters
    ----------
    name : str
        Name of the model. Defaults to 'Unnamed model'
    time_scheme : str
        Type of time scheme. It must be either 'implicit' or 'explicit'
    root : bool
        Whether or not is a root model
    hierarchy : int
        Hierarchy level. The lower the hierarchy the greater its priority. Notice that
        `root model > implicit models > explicit models` is always true regardless of
        its hierarchy level

    .. note::

        Notice that some model methods may be overrided by other models with higher hierarchy level

    """

    _mtype_ = 'PXD'
