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
from cideMOD.models.model_options import ModelOptions


class BaseModel:
    def __init__(self, cell, model_options: ModelOptions):
        self.cell = cell
        self.solve_thermal = model_options.solve_thermal
        self.solve_mechanic = model_options.solve_mechanic
        self.solve_sei = model_options.solve_SEI
        self.solve_lam = model_options.solve_LAM
        self.calc_dimensionless_parameters()

    def dimensional_variables(self, f_1):
        fields = f_1._fields
        new_f = []
        for i, fname in enumerate(fields):
            unscaled_field = self.unscale_variables({fname: f_1[i]}).get(fname, f_1[i])
            new_f.append(unscaled_field)
        return new_f

    def scale_variables(self, variables_dict):
        scaled_dict = self._scale_electrochemical_variables(variables_dict)
        if self.solve_thermal:
            scaled_dict_th = self._scale_thermal_variables(variables_dict)
            scaled_dict = {**scaled_dict, **scaled_dict_th}
        if self.solve_mechanic:
            scaled_dict_mec = self._scale_mechanical_variables(variables_dict)
            scaled_dict = {**scaled_dict, **scaled_dict_mec}
        if self.solve_sei:
            scaled_dict_sei = self._scale_sei_variables(variables_dict)
            scaled_dict = {**scaled_dict, **scaled_dict_sei}
        if self.solve_lam:
            scaled_dict_lam = self._scale_lam_variables(variables_dict)
            scaled_dict = {**scaled_dict, **scaled_dict_lam}
        return scaled_dict

    def unscale_variables(self, variables_dict):
        unscaled_dict = self._unscale_electrochemical_variables(variables_dict)
        if self.solve_thermal:
            unscaled_dict_th = self._unscale_thermal_variables(variables_dict)
            unscaled_dict = {**unscaled_dict, **unscaled_dict_th}
        if self.solve_mechanic:
            unscaled_dict_mec = self._unscale_mechanical_variables(variables_dict)
            unscaled_dict = {**unscaled_dict, **unscaled_dict_mec}
        if self.solve_sei:
            unscaled_dict_sei = self._unscale_sei_variables(variables_dict)
            unscaled_dict = {**unscaled_dict, **unscaled_dict_sei}
        if self.solve_lam:
            unscaled_dict_lam = self._unscale_lam_variables(variables_dict)
            unscaled_dict = {**unscaled_dict, **unscaled_dict_lam}
        return unscaled_dict

    def _parse_cell_value(self, value):
        if isinstance(value, dict):
            assert "value" in value.keys()
            return value["value"]
        else:
            return value

    def calc_dimensionless_parameters(self):
        self._calc_electrochemical_dimensionless_parameters()
        if self.solve_thermal:
            self._calc_thermal_dimensionless_parameters()
        if self.solve_mechanic:
            self._calc_mechanic_dimensionless_parameters()
        if self.solve_sei:
            self._calc_sei_dimensionless_parameters()
        if self.solve_lam:
            self._calc_lam_dimensionless_parameters()

    def material_parameters(self, material):
        pars = self._material_electrochemical_parameters(material)
        if self.solve_thermal:
            therm = self._material_thermal_parameters(material)
            pars = {**pars, **therm}
        if self.solve_mechanic:
            mech = self._material_mechanic_parameters(material)
            pars = {**pars, **mech}
        if self.solve_sei:
            sei = self._material_sei_parameters(material)
            pars = {**pars, **sei}
        if self.solve_lam:
            lam = self._material_lam_parameters(material)
            pars = {**pars, **lam}
        return pars

