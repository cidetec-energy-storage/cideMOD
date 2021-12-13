#
# Copyright (c) 2021 CIDETEC Energy Storage.
#
# This file is part of PXD.
#
# PXD is free software: you can redistribute it and/or modify
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
from PXD.models.base.base_nondimensional import BaseModel

class MechanicModel(BaseModel):
    def _unscale_mechanical_variables(self, variables_dict):
        return {}

    def _scale_mechanical_variables(self, variables_dict):
        return {}
    
    def _calc_mechanic_dimensionless_parameters(self):
        pass
