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
from .degradation.nondimensional import SolventLimitedSEIModel
from .electrochemical.nondimensional import ElectrochemicalModel
from .mechanical.nondimensional import MechanicModel
from .thermal.nondimensional import ThermalModel

from cideMOD.helpers.miscellaneous import project_onto_subdomains
from dolfin import grad, inner

class NondimensionalModel(ThermalModel, MechanicModel, SolventLimitedSEIModel, ElectrochemicalModel):
    def physical_variables(self, problem):
        c_e = self.c_e_0 + self.delta_c_e_ref * problem.P1_map.generate_vector({'anode':problem.f_1.c_e,'separator':problem.f_1.c_e,'cathode':problem.f_1.c_e})
        phi_e = self.phi_e_ref + self.liquid_potential * problem.P1_map.generate_vector({'anode':problem.f_1.phi_e,'separator':problem.f_1.phi_e,'cathode':problem.f_1.phi_e})
        phi_s_el = self.phi_s_ref + self.solid_potential * problem.P1_map.generate_vector({'anode':problem.f_1.phi_s,'cathode':problem.f_1.phi_s})
        if 'phi_s_cc' in problem.f_1._fields:
            phi_s_cc = self.phi_s_ref + self.solid_potential * problem.P1_map.generate_vector({'positiveCC':problem.f_1.phi_s_cc,'negativeCC':problem.f_1.phi_s_cc})
        else:
            phi_s_cc = 0
        c_s_a_index = problem.f_1._fields.index('c_s_0_a0')
        c_s_a = [problem.P1_map.generate_vector({'anode':problem.f_1[c_s_a_index + i]}) * c_s_max for i, c_s_max in enumerate(self.c_s_a_max) ]
        x_a = [problem.P1_map.generate_vector({'anode':problem.f_1[c_s_a_index + i]}) for i, c_s_max in enumerate(self.c_s_a_max) ]
        c_s_c_index = problem.f_1._fields.index('c_s_0_c0')
        c_s_c = [problem.P1_map.generate_vector({'cathode':problem.f_1[c_s_c_index + i]}) * c_s_max for i, c_s_max in enumerate(self.c_s_c_max) ]
        x_c = [problem.P1_map.generate_vector({'cathode':problem.f_1[c_s_c_index + i]}) for i, c_s_max in enumerate(self.c_s_c_max) ]
        j_Li_a_index = problem.f_1._fields.index('j_Li_a0')
        j_Li_a = sum([self.I_0/self.L_0 * problem.P1_map.generate_vector({'anode':problem.f_1[j_Li_a_index + i]}) for i in range(len(self.cell.negative_electrode.active_materials))]) 
        j_Li_c_index = problem.f_1._fields.index('j_Li_c0')
        j_Li_c = sum([self.I_0/self.L_0 * problem.P1_map.generate_vector({'cathode':problem.f_1[j_Li_c_index + i]}) for i in range(len(self.cell.positive_electrode.active_materials))])
        T = self.T_ref + self.thermal_gradient*problem.f_1.temp
        vars = {
            'c_e': problem.P1_map.generate_function({'anode':c_e,'separator':c_e,'cathode':c_e}),
            'phi_e': problem.P1_map.generate_function({'anode':phi_e,'separator':phi_e,'cathode':phi_e}),
            'phi_s': problem.P1_map.generate_function({'anode':phi_s_el,'cathode':phi_s_el,'positiveCC':phi_s_cc,'negativeCC':phi_s_cc}),
            'c_s_a': [problem.P1_map.generate_function({'anode': c_s}) for c_s in c_s_a],
            'x_a': [problem.P1_map.generate_function({'anode': c_s}) for c_s in x_a],
            'c_s_c': [problem.P1_map.generate_function({'cathode': c_s}) for c_s in c_s_c],
            'x_c': [problem.P1_map.generate_function({'cathode': c_s}) for c_s in x_c],
            'j_Li': problem.P1_map.generate_function({'anode':j_Li_a,'cathode':j_Li_c}),
            'T': T,
            'temp': T
        }

        unscaled_vars = self.unscale_variables(problem.f_1._asdict())
        for physical_var in ['electric_current', 'q_ohmic_e', 'q_ohmic_s', 'q_rev_a', 'q_rev_c', 'q_irrev_a', 'q_irrev_c']:

            if not any([ (var if isinstance(var, str) else var[0]) == physical_var for var in problem.internal_storage_order ]):
                continue
            
            elif physical_var == 'electric_current':
                phi_s, phi_s_cc = unscaled_vars['phi_s'], unscaled_vars['phi_s_cc']
                electric_current = lambda domain, phis: - domain.sigma / self.L_0 * grad(phis)
                source_dict = {
                    'anode': electric_current(problem.anode, phi_s), 
                    'cathode': electric_current(problem.cathode, phi_s),
                    'negativeCC': electric_current(problem.negativeCC, phi_s_cc),
                    'positiveCC': electric_current(problem.positiveCC, phi_s_cc)
                }
                vars['electric_current'] = project_onto_subdomains(source_dict, problem, vector = True)

            elif physical_var == 'q_ohmic_e':
                temp, phi_e, c_e = unscaled_vars['temp'], unscaled_vars['phi_e'], unscaled_vars['c_e']
                q_ohmic_e = lambda domain : domain.kappa / self.L_0**2 * inner(grad(phi_e), grad(phi_e)) \
                            - 2*domain.kappa*self.cell.R*temp/self.cell.F*(1-problem.t_p)*problem.activity/c_e*inner(grad(c_e), grad(phi_e))/self.L_0**2 
                vars['q_ohmic_e'] =  project_onto_subdomains({domain: q_ohmic_e(getattr(problem,domain)) for domain in ['anode','cathode','separator']}, problem)
            
            elif physical_var == 'q_ohmic_s':
                phi_s, phi_s_cc = unscaled_vars['phi_s'], unscaled_vars['phi_s_cc']
                q_ohmic_s = lambda domain, phis : domain.sigma / self.L_0**2 * inner(grad(phis), grad(phis))
                source_dict = {
                    'anode': q_ohmic_s(problem.anode, phi_s), 
                    'cathode': q_ohmic_s(problem.cathode, phi_s),
                    'negativeCC': q_ohmic_s(problem.negativeCC, phi_s_cc),
                    'positiveCC': q_ohmic_s(problem.positiveCC, phi_s_cc)
                }
                vars['q_ohmic_s'] = project_onto_subdomains(source_dict, problem)

            elif physical_var in ['q_irrev_a', 'q_irrev_c']:      
                vars[physical_var] = list()
                subdomain = 'anode' if physical_var == 'q_irrev_a' else 'cathode'
                domain = getattr(problem, subdomain)
                temp, phi_s, phi_e = unscaled_vars['temp'], unscaled_vars['phi_s'], unscaled_vars['phi_e']

                for i, material in enumerate(domain.active_material):
                    j_Li = self.I_0/self.L_0 * problem.f_1._asdict()[f'j_Li_{domain.tag}{i}'] # == a_s_i * j_Li_i
                    c_s_surf = problem.f_1._asdict()[f'c_s_0_{domain.tag}{i}']
                    ocv = material.U(c_s_surf) + material.delta_S(c_s_surf) * ((temp - material.U.T_ref) )
                    eta = phi_s - phi_e - ocv
                    vars[physical_var].append( project_onto_subdomains({subdomain:j_Li * eta}, problem))

            elif physical_var in ['q_rev_a', 'q_rev_c']:
                vars[physical_var] = list()
                subdomain = 'anode' if physical_var == 'q_rev_a' else 'cathode'
                domain = getattr(problem, subdomain)
                temp = unscaled_vars['temp']
                for i, material in enumerate(domain.active_material):
                    j_Li = self.I_0/self.L_0 * problem.f_1._asdict()[f'j_Li_{domain.tag}{i}'] # == a_s_i * j_Li_i
                    c_s_surf = problem.f_1._asdict()[f'c_s_0_{domain.tag}{i}']
                    entropy = material.delta_S(c_s_surf)
                    vars[physical_var].append( project_onto_subdomains({subdomain: j_Li * temp * entropy }, problem))

        return vars
