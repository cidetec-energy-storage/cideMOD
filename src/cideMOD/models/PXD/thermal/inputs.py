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
import pydantic

from cideMOD.cell.parser import CellParser, BaseComponentParser
from cideMOD.models import BaseModelOptions, register_model_options
from cideMOD.models.PXD.base_model import BasePXDModelInputs
from cideMOD.models.PXD.thermal import __model_name__


@register_model_options(__model_name__)
class ThermalModelOptions(pydantic.BaseModel):
    """
    Thermal Model
    -------------
    solve_thermal: bool
        Whether or not to solve the thermal model
    """
    solve_thermal: bool = False


class ThermalModelInputs(BasePXDModelInputs):

    # ******************************************************************************************* #
    # ***                                    ModelOptions                                     *** #
    # ******************************************************************************************* #

    @classmethod
    def is_active_model(cls, model_options: BaseModelOptions) -> bool:
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
        # TODO: Ensure that model_options has been extended with ThermalModelOptions
        return model_options.solve_thermal

    # ******************************************************************************************* #
    # ***                                     CellParser                                      *** #
    # ******************************************************************************************* #

    def parse_cell_parameters(self, cell: CellParser) -> None:
        """
        This methods parses the cell parameters of the thermal model.

        Parameters
        ----------
        cell: CellParser
            Parser of the cell dictionary.
        """
        cell.set_parameters(__cell_parameters__['cell'])

    def parse_electrode_parameters(self, electrode: BaseComponentParser):
        """
        This method parses the electrode parameters of the thermal
        model.

        Parameters
        ----------
        electrode: BaseComponentParser
            Object that parses the electrode parameters.
        """
        electrode.set_parameters(__cell_parameters__['porous_component'])
        electrode.set_parameters(__cell_parameters__['electrode'])

    def parse_separator_parameters(self, separator: BaseComponentParser) -> None:
        """
        This method parses the separator parameters of the thermal
        model.

        Parameters
        ----------
        separator: BaseComponentParser
            Object that parses the separator parameters.
        """
        # TODO: maybe include metaclasses
        separator.set_parameters(__cell_parameters__['porous_component'])
        separator.set_parameters(__cell_parameters__['separator'])

    def parse_current_collector_parameters(self, cc: BaseComponentParser) -> None:
        """
        This method parses the current collector parameters of the
        thermal model.

        Parameters
        ----------
        cc: BaseComponentParser
            Object that parses the current collector parameters.
        """
        cc.set_parameters(__cell_parameters__['current_collector'])

    def parse_electrolyte_parameters(self, electrolyte: BaseComponentParser) -> None:
        """
        This method parses the electrolyte parameters of the thermal
        model.

        Parameters
        ----------
        electrolyte: BaseComponentParser
            Object that parses the electrolyte parameters.
        """
        # FIXME: Fix the thermal model, make this parameter not being optional. Add density too.
        electrolyte.set_parameters(__cell_parameters__['electrolyte'])


__cell_parameters__ = {
    'cell': {
        'heat_convection': {'element': 'properties', 'aliases': ['heatConvection']},
        'thermal_expansion_rate': {'element': 'properties', 'is_optional': True,
                                   'aliases': ['thermalExpansionRate']}
    },
    'porous_component': {
        'thermal_conductivity': {'can_effective': True, 'aliases': ['thermalConductivity']},
        'specific_heat': {'can_effective': True, 'aliases': ['c_p', 'specificHeat']},
        'required_parameters': ['density']
    },
    'electrode': {},
    'separator': {},
    'current_collector': {
        'thermal_conductivity': {'aliases': ['thermalConductivity']},
        'specific_heat': {'aliases': ['c_p', 'specificHeat']},
        'required_parameters': ['density']
    },
    'electrolyte': {
        'thermal_conductivity': {'is_optional': True, 'aliases': ['thermalConductivity']},
        'specific_heat': {'is_optional': True, 'aliases': ['c_p', 'specificHeat']}
    },
}
