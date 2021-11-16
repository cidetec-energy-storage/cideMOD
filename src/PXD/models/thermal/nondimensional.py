from dolfin import inner, grad
from PXD.models.cell_components import Electrode, Separator, CurrentColector
from PXD.models.base.base_nondimensional import BaseModel

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
        accumulation_term = (domain.rho*domain.c_p / (self.rho_ref*self.c_p_ref)) * DT.dt(T_0,T) * test * dx
        diffusion_term = domain.k_t/(self.delta_k * self.k_t_reff) * inner(grad(T), grad(test)) * dx(metadata={"quadrature_degree":0})
        source_term = 0
        # Heat generated in electrolyte
        if isinstance(domain, (Electrode, Separator)):
            source_term += domain.eps_e * domain.kappa/(self.delta_K*self.K_eff_ref) * inner(grad(f_1.phi_e),grad(f_1.phi_e))*test*dx(metadata={"quadrature_degree":1})
            source_term -= domain.eps_e * domain.kappa/(self.delta_K_D*self.K_eff_ref) * (1+self.thermal_gradient/self.T_ref * T)/(1+self.delta_c_e_ref/self.c_e_0 * f_1.c_e) *inner(grad(f_1.c_e),grad(f_1.phi_e))*test*dx
        # Heat generated in solid
        if isinstance(domain, Electrode):
            source_term += (1-domain.eps_e) * domain.sigma/(self.delta_sigma*self.sigma_ref*(self.thermal_potential/self.solid_potential)) * inner(grad(f_1.phi_s),grad(f_1.phi_s))*test*dx(metadata={"quadrature_degree":1})
        if isinstance(domain, CurrentColector):
            source_term += domain.sigma/(self.delta_sigma*self.sigma_ref*(self.thermal_potential/self.solid_potential)) * inner(grad(f_1.phi_s_cc),grad(f_1.phi_s_cc))*test*dx(metadata={"quadrature_degree":1})
        # Heat generated at the active material interface
        if isinstance(domain, Electrode):
            for i, material in enumerate(domain.active_material):
                j_Li_index = f_1._fields.index(f"j_Li_{domain.tag}{i}")
                entropy = self.scale_variables({'dU/dT': material.delta_S})['dU/dT']
                source_term+= f_1[j_Li_index] * (self.T_ref/self.thermal_gradient + T) * entropy(c_s_surf[i], current) * test * dx
                eta = self.overpotential(material, f_1.phi_s, f_1.phi_e, current, c_s_surf[i], kwargs=kwargs)
                source_term+= f_1[j_Li_index] * eta * test * dx
        return accumulation_term + diffusion_term - source_term

    def T_bc_equation(self, T, T_ext, h_t, test, ds):
        return h_t*self.L_0/(self.k_t_reff*self.thermal_gradient) * (self.T_ref-T_ext+self.thermal_gradient*T)*test*ds