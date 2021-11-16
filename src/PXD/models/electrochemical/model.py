from .equations import *
from dolfin import grad

class ElectrochemicalModel:
    def fields(self, domains):
        fields = ['c_e', 'phi_e', 'phi_s', 'lm_app']
        for domain in domains:
            if domain.tag in ('a','c'):
                for i in range(len(domain.active_material)):
                    fields.append('j_Li_{tag}{number}'.format(tag=domain.tag, number=i))
        return fields

    def function_spaces(self, P1, P2, LM, mesher, domains):
        E_c_e = (P1, mesher.electrolyte)
        E_phi_e = (P1, mesher.electrolyte)
        # LM_phi = (LM, None)
        E_phi_s = (P1, mesher.solid_conductor)
        LM_app = (LM, None)
        E_electrochemical = [E_c_e, E_phi_e, E_phi_s, LM_app]
        for d in domains:
            if d.tag in ('a','c'):
                for i, material in enumerate(d.active_material):
                    E_electrochemical.append((P1, mesher.anode if d.tag == 'a' else mesher.cathode))
        return E_electrochemical

    def transport_wf(self, f_0, f_1, test, domain, measures, time_scheme, constants, scale = 1):
        self.scale = scale 
        F_c_e, F_phi_e, F_lm_phi, F_phi_s, F_phi_s_bc, F_lm_app = 0, 0, 0, 0, 0, 0
        # Electrolyte mass and charge balance
        if domain.tag in ('a','c','s'):
            F_c_e = self.c_e_form(f_0, f_1, test, domain, measures, time_scheme, constants)
            F_phi_e = self.phi_e_form(f_1, test, domain, measures)
            F_phi_e += f_1.lm_phi*test.phi_e*domain.dx(measures)
            F_lm_phi = test.lm_phi*f_1.phi_e*domain.dx(measures)
        # Solid charge balance
        if domain.tag in ('a','c','pcc','ncc'):
            F_phi_s = self.phi_s_form(f_1, test, domain, measures)
        # Solid charge BC
        if domain.tag in ('a','c'):
            F_phi_s_bc = self.phi_s_bc(f_1, test, domain, measures)
            
        return [F_c_e, F_phi_e, F_lm_phi, F_phi_s + F_phi_s_bc]

    def reaction_wf(self, f_1, c_s_surf_a, c_s_surf_c, test, domain, measures, constants):
        F_j_Li = []
        if domain.tag in ('a','c'):
            if domain.tag == 'a':
                F_j_Li = self.j_Li_form(f_1, c_s_surf_a, test, domain, measures, constants)
            else:
                F_j_Li = self.j_Li_form(f_1, c_s_surf_c, test, domain, measures, constants)
        return F_j_Li

    def operation_wf(self, f_1, test, v_app, i_app, beta, measures):
        F_lm_app = 0
        # Charge BC LM
        F_lm_app = self.lm_app_form(f_1, test, v_app, i_app, beta, measures)
        return [F_lm_app]


    def _j_Li_total(self, f_1, domain):
        if domain.tag in ('a','c'):
            j_Li = sum((material.a_s*f_1[f_1._fields.index('j_Li_{tag}{number}'.format(tag=domain.tag, number=i))] for i, material in enumerate(domain.active_material)))
        else:
            j_Li = None
        return j_Li

    def phi_e_form(self, f_1, test, domain, measures):
        j_Li = self._j_Li_total(f_1, domain)
        if isinstance(self.scale, list):
            return phi_e_equation(f_1.phi_e, test.phi_e, domain.dx(measures), f_1.c_e, j_Li, domain.kappa, domain.kappa_D, domain.grad, domain.L)
        else:
            return phi_e_equation(f_1.phi_e, test.phi_e, domain.dx(measures), f_1.c_e, j_Li, domain.kappa, domain.kappa_D, grad, None, self.scale)

    def phi_s_form(self, f_1, test, domain, measures):
        j_Li = self._j_Li_total(f_1, domain)
        if isinstance(self.scale, list):
            return phi_s_equation(f_1.phi_s, test.phi_s, domain.dx(measures), j_Li, domain.sigma, domain.grad, domain.L)
        else:
            return phi_s_equation(f_1.phi_s, test.phi_s, domain.dx(measures), j_Li, domain.sigma, grad, None, self.scale)

    def phi_s_bc(self, f_1, test, domain, measures):
        sign = 1 if domain.tag in ('a','pcc') else -1
        if isinstance(self.scale, list):
            return sign*f_1.lm_app*test.phi_s*domain.ds(measures)(metadata={"quadrature_degree":1})
        else:
            return sign/self.scale*f_1.lm_app*test.phi_s*domain.ds(measures)(metadata={"quadrature_degree":1})

    def c_e_form(self, f_0, f_1, test, domain, measures, time_scheme, constants):
        j_Li = self._j_Li_total(f_1, domain)
        if isinstance(self.scale, list):
            return c_e_equation(f_0.c_e, f_1.c_e, test.c_e, domain.dx(measures), time_scheme, j_Li, domain.D_e, domain.eps_e, domain.t_p, constants.F, domain.grad, domain.L)
        else:
            return c_e_equation(f_0.c_e, f_1.c_e, test.c_e, domain.dx(measures), time_scheme, j_Li, domain.D_e, domain.eps_e, domain.t_p, constants.F, grad, None, self.scale)

    def j_Li_form(self, f_1, c_s_surf, test, domain, measures, constants, amplification=1):
        F_j_Li = []
        for i, material in enumerate(domain.active_material):
            j_li_index = f_1._fields.index('j_Li_{tag}{number}'.format(tag=domain.tag, number=i))
            j_li = j_Li_equation(material.k_0, f_1.c_e, c_s_surf[i], material.c_s_max, constants.alpha, f_1.phi_s, f_1.phi_e, material.U, constants.F, constants.R, f_1.temp)
            F_j_Li.append(
                amplification * (f_1[j_li_index] - j_li) * test[j_li_index] * domain.dx(measures)
            )
        return F_j_Li

    def lm_app_form(self, f_1, test, v_app, i_app, beta, measures):
        F = ((1 - beta) * (f_1.lm_app - i_app) - beta * v_app ) * test.lm_app * measures.s_c(metadata={"quadrature_degree":0}) + ((1 - beta) * (f_1.lm_app - i_app) - beta * v_app ) * test.lm_app * measures.s_a(metadata={"quadrature_degree":0}) +\
            beta * f_1.phi_s * test.lm_app * measures.s_c(metadata={"quadrature_degree":1}) - beta * f_1.phi_s * test.lm_app * measures.s_a(metadata={"quadrature_degree":1})
        return F