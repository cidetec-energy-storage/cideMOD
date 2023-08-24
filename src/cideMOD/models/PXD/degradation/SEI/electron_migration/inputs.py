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

from cideMOD.models.model_options import BaseModelOptions
from cideMOD.models.PXD.base_model import BasePXDModelInputs


class MigrationSEIModelInputs(BasePXDModelInputs):

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
        # TODO: Ensure that model_options has been extended with SEIModelOptions
        return model_options.solve_SEI and model_options.SEI_model == 'electron_migration'

    # ******************************************************************************************* #
    # ***                                     CellParser                                      *** #
    # ******************************************************************************************* #

    def parse_SEI_parameters(self, SEI) -> None:
        """
        This method parses the electrode parameters of the SEI model.

        Parameters
        ----------
        SEI: BaseComponentParser
            Object that parses the SEI parameters.
        """
        if not SEI.has_compact:
            raise KeyError("Parameters of the compact SEI are missing")

    def parse_compactSEI_parameters(self, compact) -> None:
        """
        This method parses the electrode parameters of the
        compact SEI model.

        Parameters
        ----------
        compactSEI: BaseComponentParser
            Object that parses the compact SEI parameters.
        """
        compact.set_parameters(__cell_parameters__['compact'])


__cell_parameters__ = {
    'compact': {
        'reference_voltage': {'aliases': ['referenceVoltage']},
        'charge_transfer_coefficient': {'default': 0.5, 'is_optional': True,
                                        'aliases': ['chargeTransferCoefficient']},
        'electron_conductivity': {'aliases': ['electronConductivity']},
    }
}
