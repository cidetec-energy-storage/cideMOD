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
from dolfin import Constant, conditional, grad, inner, lt

import re
from typing import List

import numpy as np
from ufl.operators import exp, sinh

from cideMOD.helpers.config_parser import CellParser, electrode
from cideMOD.helpers.miscellaneous import constant_expression, get_spline
from cideMOD.models.base.base_nondimensional import BaseModel
from cideMOD.models.cell_components import CurrentColector, Electrode, Separator

# from numerics.spline import spline

class ElectrochemicalModel(BaseModel):
    def _scale_electrochemical_variables(self, variables_dict):
        scaled_dict = {}
        for key, value in variables_dict.items():
            if key == 'c_e':
                scaled_dict['c_e'] =  (value - self.c_e_0) / self.delta_c_e_ref
            if key == 'phi_e':
                scaled_dict['phi_e'] = (value - self.phi_e_ref) / self.liquid_potential 
            if key == 'phi_s':
                scaled_dict['phi_s'] = (value - self.phi_s_ref) / self.solid_potential
            if key == 'phi_s_cc':
                scaled_dict['phi_s_cc'] = (value - self.phi_s_ref) / self.solid_potential
            if key.startswith('c_s'):
                mat_index = int(key[-1])
                electrode_index = key[-2]
                c_s_ref = self.c_s_a_max if electrode_index == 'a' else self.c_s_c_max
                scaled_dict[key] = value / c_s_ref[mat_index]
            if key.startswith('j_Li'):
                scaled_dict[key] = self.L_0/self.I_0 * value
            if key == 'OCV':
                if callable(value):
                    def scaled_OCV( OCV_function):
                        def OCV(c_s, current = None):
                            return (OCV_function(c_s, current) - self.OCV_ref)/self.thermal_potential
                        return OCV
                    scaled_dict['OCV'] = scaled_OCV(value)
                else:
                    scaled_dict['OCV'] = (value - self.OCV_ref)/self.thermal_potential
            if key == 'x':
                scaled_dict['x'] = value / self.L_0
            if key == 'T':
                scaled_dict['T'] = (value-self.T_ref) / self.thermal_gradient
        return scaled_dict

    def _unscale_electrochemical_variables(self, variables_dict):
        unscaled_dict = {}
        for key, value in variables_dict.items():
            if key == 'c_e':
                unscaled_dict['c_e'] =  self.c_e_0 + self.delta_c_e_ref * value
            if key == 'phi_e':
                unscaled_dict['phi_e'] = self.phi_e_ref + self.liquid_potential * value 
            if key == 'phi_s':
                unscaled_dict['phi_s'] = self.phi_s_ref + self.solid_potential * value
            if key == 'phi_s_cc':
                unscaled_dict['phi_s_cc'] = self.phi_s_ref + self.solid_potential * value
            if key.startswith('c_s'):
                mat_index = int(key[-1])
                electrode_index = key[-2]
                c_s_ref = self.c_s_a_max if electrode_index == 'a' else self.c_s_c_max
                unscaled_dict[key] = value * c_s_ref[mat_index]
            if key.startswith('j_Li'):
                unscaled_dict[key] =  value * self.I_0/self.L_0
            if key == 'x':
                unscaled_dict['x'] = value * self.L_0
            if key == 'T':
                unscaled_dict['T'] = self.T_ref + value * self.thermal_gradient
        return unscaled_dict

    def _calc_electrochemical_dimensionless_parameters(self):
        self.t_c = 3600
        self.T_ref = 298.15
        self.phi_e_ref = 0

        self.I_0 = (self.cell.capacity/3600)*self.t_c/self.cell.area 
        
        self.c_e_0 = self.cell.electrolyte.initialConcentration

        def get_ocv(ocv, x):
            if isinstance(ocv['value'],dict):
                U_eq = get_spline(np.loadtxt(ocv['value']['discharge']),spline_type=ocv['spline_type'], return_fenics=False)
            else:
                U_eq = get_spline(np.loadtxt(ocv['value']),spline_type=ocv['spline_type'], return_fenics=False)
            return U_eq([x])
        
        phi_s_a_ref = max([get_ocv(material.openCircuitPotential, material.stoichiometry1) for material in self.cell.negative_electrode.active_materials] + [get_ocv(material.openCircuitPotential, material.stoichiometry0) for material in self.cell.negative_electrode.active_materials]) 

        phi_s_c_ref = max([get_ocv(material.openCircuitPotential, material.stoichiometry1) for material in self.cell.positive_electrode.active_materials] + [get_ocv(material.openCircuitPotential, material.stoichiometry0) for material in self.cell.positive_electrode.active_materials]) 

        self.phi_s_ref = max(phi_s_a_ref, phi_s_c_ref)[0]

        self.OCV_ref = self.phi_s_ref - self.phi_e_ref

        self.thermal_potential = self.cell.R * self.T_ref / (self.cell.alpha * self.cell.F)

        # Parametros dependientes de la estructura
        self.L_0 = self.cell.separator.thickness
        if 'a' in self.cell.structure:
            self.L_0 += self.cell.negative_electrode.thickness 
        if 'c' in self.cell.structure:
            self.L_0 += self.cell.positive_electrode.thickness

        self.D_e_eff_ref = constant_expression(self._parse_cell_value(self.cell.electrolyte.diffusionConstant), c_e=self.c_e_0, temp=self.T_ref)

        self.K_eff_ref = constant_expression(self._parse_cell_value(self.cell.electrolyte.ionicConductivity), c_e=self.c_e_0, temp=self.T_ref)

        self.sigma_ref = np.min([self._parse_cell_value(self.cell.negative_electrode.electronicConductivity), 
            self._parse_cell_value(self.cell.positive_electrode.electronicConductivity)])
        
        self.solid_potential = self.I_0 * self.L_0 / self.sigma_ref 
        # self.liquid_potential = self.I_0 * self.L_0 / self.K_eff_ref

        # self.solid_potential = self.thermal_potential 
        self.liquid_potential = self.thermal_potential        

        self.tau_e = self.D_e_eff_ref * self.t_c / (self.L_0 ** 2)

        t_p = constant_expression(str(self.cell.electrolyte.transferenceNumber), c_e=self.c_e_0, temp=self.T_ref, T_0=self.T_ref)
        a_D = constant_expression(str(self.cell.electrolyte.activityDependence), c_e=self.c_e_0, temp=self.T_ref, T_0=self.T_ref, t_p=t_p)

        self.delta_c_e_ref = self.I_0 * self.L_0 * (1-t_p) / (self.D_e_eff_ref * self.cell.F)

        self.delta_K = self.L_0 * self.I_0 / (self.K_eff_ref * self.liquid_potential)
        
        self.delta_K_D = self.delta_K * self.liquid_potential * self.c_e_0 / ( 2 * self.cell.alpha * self.thermal_potential * a_D * (1-t_p) * self.delta_c_e_ref)
        
        self.delta_sigma = self.L_0 * self.I_0 / (self.sigma_ref * self.solid_potential)

        self.c_s_a_max = [ material.maximumConcentration for material in self.cell.negative_electrode.active_materials ]
        # Parametros dependientes del electrodo
        def material_parameters(materials:List[electrode.active_material]):
            d_s_ref = [ np.mean([eval(str(mat.diffusionConstant),{"x":i}) for i in np.linspace(mat.stoichiometry0,mat.stoichiometry1,num=10)]) for mat in materials ]
            tau_s = [ d_s_ref[mat.index] * self.t_c / mat.particleRadius**2 for mat in materials]
            S = [ mat.particleRadius**2 * self.I_0 / (3*mat.volumeFraction*d_s_ref[mat.index]*mat.maximumConcentration*self.cell.F*self.L_0 ) for mat in materials]
            k_0 = [ (mat.volumeFraction*3/mat.particleRadius)*self.cell.F*mat.kineticConstant*self.L_0 * self.c_e_0**0.5 * mat.maximumConcentration / self.I_0 for mat in materials]
            return d_s_ref, tau_s, S, k_0
            

        if 'a' in self.cell.structure:
            self.D_s_a_ref, self.tau_s_a, self.S_a, self.k_0_a = material_parameters(self.cell.negative_electrode.active_materials)

        self.c_s_c_max = [ material.maximumConcentration for material in self.cell.positive_electrode.active_materials ]
        if 'c' in self.cell.structure:
            self.D_s_c_ref, self.tau_s_c, self.S_c, self.k_0_c = material_parameters(self.cell.positive_electrode.active_materials)

        if 'pcc' in self.cell.structure or 'ncc' in self.cell.structure:
            self.sigma_cc_ref = np.max([self.cell.positive_curent_colector.electronicConductivity, self.cell.negative_curent_colector.electronicConductivity])
            self.conductor_potential = self.I_0*self.L_0/self.sigma_cc_ref
        else:
            self.sigma_cc_ref = self.sigma_ref
            self.conductor_potential = 1

        self.thermal_gradient = 1

    def _material_electrochemical_parameters(self, material):
        if isinstance(material, electrode.active_material):
            c_s_max = material.maximumConcentration
            D_s_ref = np.mean(constant_expression(material.diffusionConstant, x = np.linspace(material.stoichiometry0,material.stoichiometry1,num=10)))
            tau_s = D_s_ref * self.t_c / material.particleRadius ** 2
            S = material.particleRadius**2 * self.I_0 / (3*material.volumeFraction*D_s_ref*c_s_max*self.cell.F*self.L_0 )
            k_0 = (3*material.volumeFraction/material.particleRadius)*self.cell.F*material.kineticConstant*self.L_0 * self.c_e_0**0.5 * c_s_max / self.I_0
        else:
            c_s_max = material.c_s_max
            D_s_ref = np.mean(constant_expression(material.D_s, x = np.linspace(material.stoichiometry0,material.stoichiometry1,num=10)))
            tau_s = D_s_ref * self.t_c / material.R_s ** 2
            S = material.R_s**2 * self.I_0 / (3*material.eps_s*D_s_ref*c_s_max*self.cell.F*self.L_0 )
            k_0 = (3*material.eps_s/material.R_s)*self.cell.F*material.k_0*self.L_0 * self.c_e_0**0.5 * c_s_max / self.I_0
        return {
            'D_s_ref': D_s_ref,
            'tau_s': tau_s,
            'S': S,
            'k_0': k_0
        }

    def phi_e_equation(self, domain, phi_e, c_e, test, j_li, T, dx):
        migration_term = 1/(self.delta_K*self.K_eff_ref) * domain.kappa * inner(grad(phi_e), grad(test)) * dx(metadata={"quadrature_degree":0}) 
        diffusion_term = 1/(self.delta_K_D*self.K_eff_ref) * domain.kappa * (1+(self.thermal_gradient/self.T_ref) * T) / (1+(self.delta_c_e_ref/self.c_e_0) * c_e) * inner(grad(c_e), grad(test)) * dx
        weak_form = migration_term - diffusion_term
        if j_li:
            reaction_term = j_li*test*dx
            weak_form -= reaction_term
        return weak_form

    def phi_s_electrode_equation(self, domain, phi_s, test, j_li, dx, lagrange_multiplier=None, dS=None):
        weak_form = 1/(self.delta_sigma) * inner(grad(phi_s), grad(test)) * dx(metadata={"quadrature_degree":0})
        if j_li:
            weak_form += self.sigma_ref/domain.sigma * j_li*test*dx
        if lagrange_multiplier and dS:
            weak_form += self.sigma_ref/domain.sigma * self.phi_s_interface(lagrange_multiplier, phi_s_test = test, dS=dS)
        return weak_form

    def phi_s_conductor_equation(self, domain, phi_s_cc, test, dx, lagrange_multiplier=None, dS=None):
        sigma_ratio = self.sigma_ref/self.sigma_cc_ref
        weak_form = domain.sigma * (sigma_ratio / (self.delta_sigma * self.sigma_ref)) * inner(grad(phi_s_cc), grad(test)) * dx(metadata={"quadrature_degree":0})
        if lagrange_multiplier and dS:
            weak_form -= sigma_ratio*self.phi_s_interface(lagrange_multiplier, phi_s_cc_test = test, dS=dS)
        return weak_form

    def phi_s_bc(self, i_app, test, ds, area_ratio=1, eq_scaling=1):
        sigma_ratio = self.sigma_ref/self.sigma_cc_ref
        return sigma_ratio*eq_scaling*area_ratio*i_app*test*ds

    def phi_s_interface(self, lagrange_multiplier, dS, phi_s_test = None, phi_s_cc_test=None):
        if phi_s_test:
            interface_bc = lagrange_multiplier(dS.metadata()['direction'])*phi_s_test(dS.metadata()['direction'])*dS
        elif phi_s_cc_test:
             interface_bc = lagrange_multiplier(dS.metadata()['direction'])*phi_s_cc_test(dS.metadata()['direction'])*dS
        return  interface_bc
    
    def phi_s_continuity(self, phi_s_electrode, dS_el, phi_s_cc, lm_test, dS_cc):
        return phi_s_electrode(dS_el.metadata()['direction']) * lm_test(dS_el.metadata()['direction']) * dS_el - phi_s_cc(dS_cc.metadata()['direction']) * lm_test(dS_cc.metadata()['direction']) * dS_cc 

    def c_e_equation(self, domain, DT, c_e, c_e_0, test, j_li, dx):
        accumulation_term = (1/self.tau_e) * domain.eps_e * DT.dt(c_e_0, c_e) * test* dx
        diffusion_term = (1/self.D_e_eff_ref) * domain.D_e * inner(grad(c_e), grad(test)) * dx(metadata={"quadrature_degree":0})
        weak_form = accumulation_term + diffusion_term
        if j_li:
            reaction_term = j_li * test * dx 
            weak_form -= reaction_term
        return weak_form

    def j_Li_equation(self, material, c_e, c_s_surf, phi_s, phi_e, T, current, **kwargs):
        mat_dp = self.material_parameters(material)
        f_c_e, f_c_s, f_c_s_max = [1,1/material.c_s_max**2,1/material.c_s_max**2]
        regularization = exp(-f_c_s/c_s_surf**2) * exp(-f_c_e/(self.c_e_0+self.delta_c_e_ref*c_e)**2) * exp(-f_c_s_max/(1 - c_s_surf)**2)    
        i_0 = mat_dp['k_0'] * c_s_surf **0.5 * (1-c_s_surf)**0.5 * (1+self.delta_c_e_ref/self.c_e_0 * c_e) ** 0.5
        i_n = conditional(lt(self.c_e_0+self.delta_c_e_ref*c_e, 0), 0, conditional(lt(c_s_surf,0),0, conditional(lt(1-c_s_surf,0), 0, i_0 * regularization)))
        eta = self.overpotential(material, phi_s, phi_e, current, c_s_surf, kwargs=kwargs) / (1+self.thermal_gradient/self.T_ref * T)
        j_li = i_n * 2 * sinh(eta)
        return j_li

    def overpotential(self, material, phi_s, phi_e, current, c_s_surf, **kwargs):
        ocv = self.scale_variables({'OCV': material.U})['OCV']
        eta = (self.solid_potential/self.thermal_potential*phi_s - self.liquid_potential/self.thermal_potential*phi_e - ocv(c_s_surf, current)) 
        return eta

    def get_j_total(self, f, material):
        regex = r"j_\w*({tag}{i})".format(tag=material.electrode.tag, i=material.index)
        indexes = [i for i, field in enumerate(f._fields) if re.search(regex,field)]
        j_total = sum(field for i, field in enumerate(f) if i in indexes)
        return j_total
