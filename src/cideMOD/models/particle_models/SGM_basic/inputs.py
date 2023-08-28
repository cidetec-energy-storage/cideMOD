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

from cideMOD.cell.parser import BaseComponentParser
from cideMOD.models import register_model_options, __mtypes__
from cideMOD.models.PXD.base_model import BasePXDModelInputs
from cideMOD.models.model_options import BaseModelOptions
from cideMOD.models.particle_models.SGM_basic import __model_name__


@register_model_options(__model_name__)
class ParticleModelSGMOptions(BaseModel):
    """
    Particle Model. SGM
    -------------------
    particle_order: int
        Order of spectral finite elements interpolation in particle.
        Defaults to 2
    """
    particle_order: int = 2

    @validator("particle_order")
    def validate_particle_order(cls, v):
        if v <= 0:
            raise ValueError("Particle order must be a non-zero positive integer")
        return v


class ParticleModelSGMInputs(BasePXDModelInputs):

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
        return model_options.particle_model == 'SGM'

    # ******************************************************************************************* #
    # ***                                     CellParser                                      *** #
    # ******************************************************************************************* #

    def parse_active_material_parameters(self, am: BaseComponentParser) -> None:
        """
        This method parses the active material parameters of the
        SGM particle model.

        Parameters
        ----------
        am: BaseComponentParser
            Object that parses the active material parameters.
        """
        am.set_parameters(__cell_parameters__['active_material'])


__cell_parameters__ = {
    'active_material': {
        'diffusion_constant': {
            'dtypes': ('real', 'expression', 'spline'),
            'can_arrhenius': True,
            'aliases': ['D_s', 'diffusionConstant']}
    }
}
