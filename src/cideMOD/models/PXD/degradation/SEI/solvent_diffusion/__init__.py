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

__model_name__ = 'diffusionSEI'
__mtype__ = 'PXD'
__root_model__ = False
__hierarchy__ = 310

from cideMOD.models import register_model
from cideMOD.models.PXD.base_model import BasePXDModel
from cideMOD.models.model_options import BaseModelOptions

from .inputs import DiffusionSEIModelInputs
from .preprocessing import DiffusionSEIModelPreprocessing
from .equations import DiffusionSEIModelEquations
from .outputs import DiffusionSEIModelOutputs


# TODO: Make SEI submodels inherit from the BaseSEIModel
@register_model
class DiffusionSEIModel(
    DiffusionSEIModelInputs,
    DiffusionSEIModelPreprocessing,
    DiffusionSEIModelEquations,
    DiffusionSEIModelOutputs,
    BasePXDModel
):
    _name_ = __model_name__
    _mtype_ = __mtype__
    _root_ = __root_model__
    _hierarchy_ = __hierarchy__

    def __init__(self, options: BaseModelOptions):
        super().__init__(options)
        self.order = options.sei_particle_order
        self._build_lagrange()
