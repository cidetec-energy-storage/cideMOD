#
# Copyright (c) 2021 CIDETEC Energy Storage.
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
from dolfin import grad, inner

from cideMOD.models.base.base_nondimensional import BaseModel
from cideMOD.models.cell_components import CurrentColector, Electrode, Separator


class ThermalModel(BaseModel):
    def _scale_thermal_variables(self, variables_dict):
        scaled_dict = {}
        for key, value in variables_dict.items():
            if key == 'dU/dT':
                def scaled_entropy_coefficient(entropy_coefficient_function):
                    def entropy_coefficient(c_s, current = None):
                        return entropy_coefficient_function(c_s, current)*self.thermal_gradient/self.thermal_potential
                    return entropy_coefficient
                scaled_dict[key] = scaled_entropy_coefficient(value)
            if key in ['T', 'temp']:
                scaled_dict[key] = (value - self.T_ref) / self.thermal_gradient
        return scaled_dict

    def _unscale_thermal_variables(self, variables_dict):
        unscaled_dict = {}
        for key, value in variables_dict.items():
            if key in ['T', 'temp']:
                unscaled_dict[key] = self.T_ref + self.thermal_gradient * value
        return unscaled_dict

    def _calc_thermal_dimensionless_parameters(self):
        self.k_t_reff = sum([self._parse_cell_value(el.thermalConductivity)*el.thickness for el in [self.cell.positive_electrode, self.cell.separator, self.cell.negative_electrode]])/self.L_0
        self.rho_ref = sum([el.density*el.thickness for el in [self.cell.positive_electrode, self.cell.separator, self.cell.negative_electrode]])/self.L_0
        self.c_p_ref = sum([self._parse_cell_value(el.specificHeat)*el.thickness for el in [self.cell.positive_electrode, self.cell.separator, self.cell.negative_electrode]])/self.L_0

        # NOTE: This should be equal to the expected change of temperature in time.
        #       Otherwise one can selecy the thermal gradient across cell, but it is generally too small.
        #       A very small value stagnates the solver calculating more precission than the needed.  
        #       Using thermal gradien along cell = self.I_0 * self.L_0 * self.thermal_potential / self.k_t_reff
        #           - tau_T = self.t_c * self.k_t_reff / (self.rho_ref * self.c_p_ref * self.L_0**2) 
        #       Using thermal gradient over time = self.I_0 * self.t_c * self.thermal_potential / (self.L_0 * self.rho_ref * self.c_p_ref)
        #           - delta_k = (self.L_0 ** 2 * self.rho_ref * self.c_p_ref) / (self.t_c * self.k_t_reff)

        self.thermal_gradient = self.I_0 * self.t_c * self.thermal_potential / (self.L_0 * self.rho_ref * self.c_p_ref) 
        self.delta_k = (self.L_0 ** 2 * self.rho_ref * self.c_p_ref) / (self.t_c * self.k_t_reff)

    def _material_thermal_parameters(self, material):
        return {}

    def T_equation(self, domain, DT, T, T_0, test, f_1, c_s_surf, current, dx, **kwargs):
        if 'ncc' in self.cell.structure:
            domain_scale = 2 * self._parse_cell_value(self.cell.negative_curent_colector.thermalConductivity) / (self.cell.separator.height + self.cell.separator.width)
        else:
            domain_scale = 2 * self.k_t_reff / (self.cell.separator.height + self.cell.separator.width)
        accumulation_term = (domain.rho*domain.c_p / (self.rho_ref*self.c_p_ref*domain_scale)) * DT.dt(T_0,T) * test * dx
        diffusion_term = domain.k_t/(self.delta_k * self.k_t_reff*domain_scale) * inner(grad(T), grad(test)) * dx(metadata={"quadrature_degree":0})
        source_term = 0
        # Heat generated in electrolyte
        if isinstance(domain, (Electrode, Separator)):
            source_term += domain.kappa/(self.delta_K*self.K_eff_ref*domain_scale) * inner(grad(f_1.phi_e),grad(f_1.phi_e))*test*dx(metadata={"quadrature_degree":1})
            source_term -= domain.kappa/(self.delta_K_D*self.K_eff_ref*domain_scale) * (1+self.thermal_gradient/self.T_ref * T)/(1+self.delta_c_e_ref/self.c_e_0 * f_1.c_e) *inner(grad(f_1.c_e),grad(f_1.phi_e))*test*dx
        # Heat generated in solid
        if isinstance(domain, Electrode):
            source_term += domain.sigma/(self.delta_sigma*self.sigma_ref*(self.thermal_potential/self.solid_potential)*domain_scale) * inner(grad(f_1.phi_s),grad(f_1.phi_s))*test*dx(metadata={"quadrature_degree":1})
        if isinstance(domain, CurrentColector):
            source_term += domain.sigma/(self.delta_sigma*self.sigma_ref*(self.thermal_potential/self.solid_potential)*domain_scale) * inner(grad(f_1.phi_s_cc),grad(f_1.phi_s_cc))*test*dx(metadata={"quadrature_degree":1})
        # Heat generated at the active material interface
        if isinstance(domain, Electrode):
            for i, material in enumerate(domain.active_material):
                j_Li_index = f_1._fields.index(f"j_Li_{domain.tag}{i}")
                entropy = self.scale_variables({'dU/dT': material.delta_S}).get('dU/dT',lambda *args,**kwargs: 0)
                source_term+= f_1[j_Li_index]/domain_scale * (self.T_ref/self.thermal_gradient + T) * entropy(c_s_surf[i], current) * test * dx
                eta = self.overpotential(material, f_1.phi_s, f_1.phi_e, current, c_s_surf[i], T, **kwargs)
                source_term+= f_1[j_Li_index]/domain_scale * eta * test * dx
        return accumulation_term + diffusion_term - source_term

    def T_bc_equation(self, T, T_ext, h_t, test, ds):
        if 'ncc' in self.cell.structure:
            domain_scale = 2 * self._parse_cell_value(self.cell.negative_curent_colector.thermalConductivity) / (self.cell.separator.height + self.cell.separator.width)
        else:
            domain_scale = 2 * self.k_t_reff / (self.cell.separator.height + self.cell.separator.width)     
        return h_t*self.L_0/(self.k_t_reff*self.thermal_gradient*self.delta_k*domain_scale) * (self.T_ref-T_ext+self.thermal_gradient*T)*test*ds
