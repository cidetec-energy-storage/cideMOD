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

__model_name__ = 'PM_SGM'
__mtype__ = 'PXD'
__root_model__ = False
__hierarchy__ = 0

from cideMOD.models import register_model
from cideMOD.models.model_options import BaseModelOptions
from cideMOD.models.PXD.base_model import BasePXDModel
from cideMOD.models.particle_models.SGM_basic.inputs import ParticleModelSGMInputs
from cideMOD.models.particle_models.SGM_basic.preprocessing import ParticleModelSGMPreprocessing
from cideMOD.models.particle_models.SGM_basic.equations import ParticleModelSGMEquations
from cideMOD.models.particle_models.SGM_basic.outputs import ParticleModelSGMOutputs


@register_model
class ParticleModelSGM(
    ParticleModelSGMInputs,
    ParticleModelSGMPreprocessing,
    ParticleModelSGMEquations,
    ParticleModelSGMOutputs,
    BasePXDModel
):

    _name_ = __model_name__
    _mtype_ = __mtype__
    _root_ = __root_model__
    _hierarchy_ = __hierarchy__

    def __init__(self, options: BaseModelOptions):
        super().__init__(options)
        self.order = options.particle_order
        self._build_legendre()
