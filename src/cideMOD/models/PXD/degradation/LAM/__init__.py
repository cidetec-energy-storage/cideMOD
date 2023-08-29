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
__model_name__ = 'LAM'
__mtype__ = 'PXD'
__root_model__ = False
__hierarchy__ = 400


from cideMOD.models import register_model
from cideMOD.models.PXD.base_model import BasePXDModel
# from cideMOD.models.PXD.degradation.LAM.inputs import LAMModelInputs
# from cideMOD.models.PXD.degradation.LAM.preprocessing import LAMModelPreprocessing
# from cideMOD.models.PXD.degradation.LAM.equations import LAMModelEquations
# from cideMOD.models.PXD.degradation.LAM.outputs import LAMModelOutputs


# @register_model
# class LAMModel(
#     LAMModelInputs,
#     LAMModelPreprocessing,
#     LAMModelEquations,
#     LAMModelOutputs,
#     BasePXDModel
# ):
#     _name_ = __model_name__
#     _mtype_ = __mtype__
#     _root_ = __root_model__
#     _hierarchy_ = __hierarchy__
