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

from pydantic import BaseModel, validator

from cideMOD.models import register_model_options
from cideMOD.models.model_options import BaseModelOptions
from cideMOD.models.PXD.base_model import BasePXDModelInputs
from cideMOD.models.PXD.degradation.SEI.solvent_diffusion import __model_name__


@register_model_options(__model_name__)
class DiffusionSEIModelOptions(BaseModel):
    """
    SEI model: solvent-diffusion
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    sei_particle_order: int
        Order of spectral finite elements interpolation in particle.
        Defaults to 2
    """

    sei_particle_order: int = 2

    @validator("sei_particle_order")
    def validate_particle_order(cls, v):
        if v <= 0:
            raise ValueError("Particle order must be a non-zero positive integer")
        return v


class DiffusionSEIModelInputs(BasePXDModelInputs):

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
        return model_options.solve_SEI and model_options.SEI_model == 'solvent_diffusion'

    # ******************************************************************************************* #
    # ***                                     CellParser                                      *** #
    # ******************************************************************************************* #

    def parse_porousSEI_parameters(self, porous) -> None:
        """
        This method parses the electrode parameters of the
        SEI model.

        Parameters
        ----------
        SEI: BaseComponentParser
            Object that parses the SEI parameters.
        """
        porous.set_parameters(__cell_parameters__['porous'])


__cell_parameters__ = {
    'porous': {
        'reference_voltage': {'aliases': ['referenceVoltage']},
        'charge_transfer_coefficient': {'default': 0.5, 'is_optional': True,
                                        'aliases': ['chargeTransferCoefficient']},
        'solvent_diffusion': {'aliases': ['solventDiffusion']},
        'solvent_porosity': {'aliases': ['solventPorosity']},
        'solvent_surf_concentration': {'aliases': ['solventSurfConcentration']},
        'rate_constant': {'aliases': ['rateConstant']}
    }
}
