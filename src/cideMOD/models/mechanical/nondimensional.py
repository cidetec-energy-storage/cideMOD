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
from cideMOD.models.base.base_nondimensional import BaseModel
from cideMOD.helpers.config_parser import electrode

class MechanicModel(BaseModel):
    def _unscale_mechanical_variables(self, variables_dict):
        return {}

    def _scale_mechanical_variables(self, variables_dict):
        return {}
    
    def _calc_mechanic_dimensionless_parameters(self):
        self.E_a_ref = [am.young for am in self.cell.negative_electrode.active_materials]
        self.E_c_ref = [am.young for am in self.cell.positive_electrode.active_materials]

    def _material_mechanical_parameters(self, material):
        c_s_max = material.maximumConcentration if isinstance(material, electrode.active_material) else material.c_s_max
        delta_stress_h = 2/9*material.omega*c_s_max/(1-material.poisson)
        return {'delta_stress_h':delta_stress_h}