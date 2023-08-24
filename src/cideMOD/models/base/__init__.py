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

from cideMOD.models.base.base_particle_models import StrongCoupledPM, WeakCoupledPM
from cideMOD.models.base.base_models import (
    BaseCellModel,
    BaseCellModelInputs,
    BaseCellModelOutputs,
    BaseCellModelPreprocessing,
    BaseCellModelEquations
)

__all__ = [
    "StrongCoupledPM",
    "WeakCoupledPM",
    "BaseCellModel",
    "BaseCellModelInputs",
    "BaseCellModelOutputs",
    "BaseCellModelPreprocessing",
    "BaseCellModelEquations"
]

# TODO: Deprecate base_particle_models and make base be base_models
