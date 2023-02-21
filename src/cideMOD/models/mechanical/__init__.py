#
# Copyright (c) 2022 CIDETEC Energy Storage.
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
from cideMOD.models.mechanical.liquid_electrolyte import LiquidElectrolyteMechanicalModel

def mechanical_model(cell):
    if cell.electrolyte.type == 'liquid':
        raise NotImplementedError("Mechanical model for liquid electrolite cell is not implemented yet")
        #return LiquidElectrolyteMechanicalModel(cell)
    elif cell.electrolyte.type in ['solid', 'polymer']:
        raise NotImplementedError("Solid electrolytes not supported in this version")

__all__ = [
    "mechanical_model"
]