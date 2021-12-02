from .degradation.nondimensional import SolventLimitedSEIModel
from .electrochemical.nondimensional import ElectrochemicalModel
from .mechanical.nondimensional import MechanicModel
from .thermal.nondimensional import ThermalModel


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
        j_Li_c = sum([self.I_0/self.L_0 * problem.P1_map.generate_vector({'cathode':problem.f_1[j_Li_c_index + i]}) for i in range(len(self.cell.negative_electrode.active_materials))])
        T = self.T_ref + self.thermal_gradient*problem.f_1.temp
        return {
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
