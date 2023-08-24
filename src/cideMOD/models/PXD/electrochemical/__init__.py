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

__model_name__ = 'Electrochemical'
__mtype__ = 'PXD'
__root_model__ = True
__hierarchy__ = 0

from cideMOD.models import register_model, register_model_type
from cideMOD.models.PXD.base_model import BasePXDModel
from cideMOD.models.PXD.electrochemical.inputs import ElectrochemicalModelInputs
from cideMOD.models.PXD.electrochemical.preprocessing import ElectrochemicalModelPreprocessing
from cideMOD.models.PXD.electrochemical.equations import ElectrochemicalModelEquations
from cideMOD.models.PXD.electrochemical.outputs import ElectrochemicalModelOutputs

register_model_type(__mtype__, aliases=['P2D', 'P3D', 'P4D'])


@register_model
class ElectrochemicalModel(
    ElectrochemicalModelInputs,
    ElectrochemicalModelPreprocessing,
    ElectrochemicalModelEquations,
    ElectrochemicalModelOutputs,
    BasePXDModel
):

    _name_ = __model_name__
    _mtype_ = __mtype__
    _root_ = __root_model__
    _hierarchy_ = __hierarchy__

    def __init__(self, options):
        super().__init__(options)
        self._T_ext = None
        self._T_ini = None
