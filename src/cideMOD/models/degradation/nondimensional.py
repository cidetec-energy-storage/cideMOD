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
from dolfin import Constant, exp, interpolate
from multiphenics import assign

import numpy
from numpy.polynomial.polynomial import *

from cideMOD.helpers.config_parser import electrode
from cideMOD.numerics.polynomials import Lagrange
from cideMOD.models.base.base_nondimensional import BaseModel


class SolventLimitedSEIModel(BaseModel):
    def _unscale_sei_variables(self, variables_dict):
        res = {}
        for key, value in variables_dict.items():
            if 'j_sei' in key:
                res[key] = value*self.I_0/self.L_0
            if 'delta_sei' in key:
                if key[-2] == 'a':
                    res[key] = value*self.delta_sei_a[int(key[-1])]
                if key[-2] == 'c':
                    res[key] = value*self.delta_sei_c[int(key[-1])]
            if 'c_EC' in key:
                if key[-2] == 'a':
                    res[key] = value*self.c_sei_a[int(key[-1])]
                if key[-2] == 'c':
                    res[key] = value*self.c_sei_c[int(key[-1])]
        return res

    def _scale_sei_variables(self, variables_dict):
        res = {}
        for key, value in variables_dict.items():
            if 'j_sei' in key:
                res[key] = value*self.L_0/self.I_0
            if 'delta_sei' in key:
                if key[-2] == 'a':
                    res[key] = value/self.delta_sei_a[int(key[-1])]
                if key[-2] == 'c':
                    res[key] = value/self.delta_sei_c[int(key[-1])]
            if 'c_EC' in key:
                if key[-2] == 'a':
                    res[key] = value/self.c_sei_a[int(key[-1])]
                if key[-2] == 'c':
                    res[key] = value/self.c_sei_c[int(key[-1])]
        return res

    def _calc_sei_dimensionless_parameters(self):
        self.delta_sei_a = []
        self.c_sei_a = []
        for i, mat in enumerate(self.cell.negative_electrode.active_materials):
            params = self._material_sei_parameters(mat)
            self.delta_sei_a.append(params['delta_ref_sei'])
            self.c_sei_a.append(params['delta_c_sei'])
        self.delta_sei_c = []
        self.c_sei_c = []
        for i, mat in enumerate(self.cell.positive_electrode.active_materials):
            params = self._material_sei_parameters(mat)
            self.delta_sei_c.append(params['delta_ref_sei'])
            self.c_sei_c.append(params['delta_c_sei'])
        

    def _material_sei_parameters(self, material):
        SEI = material.electrode.SEI
        if isinstance(material, electrode.active_material):
            a_s = 3*material.volumeFraction/material.particleRadius
            if SEI:
                ref_thickness_change = self.t_c*self.I_0*SEI.molecularWeight/(2*self.cell.F*a_s*self.L_0*SEI.density)
                delta_c = SEI.solventSurfConcentration*SEI.EC_eps
            else:
                ref_thickness_change = 1
                delta_c = 1
            return {'delta_ref_sei': ref_thickness_change, 'delta_c_sei':delta_c}
        else:
            a_s = 3*material.eps_s/material.R_s
            if SEI:
                ref_thickness_change = self.t_c*self.I_0*SEI.M/(2*self.cell.F*a_s*self.L_0*SEI.rho)
                delta_c = SEI.c_EC_sln*SEI.eps
            else:
                ref_thickness_change = 1
                delta_c = 1
            return {'delta_ref_sei': ref_thickness_change, 'delta_c_sei':delta_c}


    def j_SEI(self, material, i_0s, phi_s, phi_e, T, J, delta):
        eta_sei = self.SEI_overpotential(material, phi_s, phi_e, J, delta) / (1+self.thermal_gradient/self.T_ref * T)
        return -i_0s * self.L_0*material.a_s/self.I_0 * exp(-eta_sei)

    def SEI_overpotential(self, material, phi_s, phi_e, J, delta):
        SEI = material.electrode.SEI
        mat_dp = self.material_parameters(material)
        sei_ocv = self.scale_variables({'OCV': SEI.U})['OCV']
        eta_sei = self.solid_potential/self.thermal_potential*phi_s - self.liquid_potential/self.thermal_potential*phi_e - sei_ocv
        sei_resistance = SEI.R+mat_dp['delta_ref_sei']*delta/SEI.kappa 
        eta_sei -= sei_resistance/self.thermal_potential * J * self.I_0/(self.L_0*material.a_s) 
        return eta_sei

    def overpotential(self, material, phi_s, phi_e, current, c_s_surf, T, **kwargs):
        SEI = material.electrode.SEI
        mat_dp = self.material_parameters(material)
        ocv_ref = self.scale_variables({'OCV': material.U})['OCV']
        delta_S = self.scale_variables({'dU/dT': material.delta_S}).get('dU/dT',lambda *args,**kwargs: 0)
        ocv = ocv_ref(c_s_surf, current)-delta_S(c_s_surf, current)*(T+(self.T_ref-material.U.T_ref)/self.thermal_gradient)
        eta = self.solid_potential/self.thermal_potential*phi_s - self.liquid_potential/self.thermal_potential*phi_e - ocv
        if all(key in kwargs for key in ('delta','J')):
            sei_resistance = SEI.R+mat_dp['delta_ref_sei']*kwargs['delta']/SEI.k 
            eta -= sei_resistance/self.thermal_potential * kwargs['J'] * self.I_0/(self.L_0*material.a_s)        
        return eta

    def delta_growth(self, DT, delta_0, delta_1, j_SEI, test, dx):
        if DT:
            return DT.dt(delta_0, delta_1) * test * dx + j_SEI * test * dx
        else:
            return (delta_0 - delta_1) * test * dx

    def SEI_equations(self, f_0, f_1, test, dx, active_material, DT = None):
        spectral_model = self.SpectralLagrangeModel_EC()
        F_SEI = []
        for i, material in enumerate(active_material):
            SEI = material.electrode.SEI
            j_sei_index = f_1._fields.index(f'j_sei_a{i}')
            delta_index = f_1._fields.index(f'delta_a{i}')
            c_EC_index = f_1._fields.index(f'c_EC_0_a{i}')
            j_int_index = f_1._fields.index(f'j_Li_a{i}')
            
            i_0s = self.cell.F * self.unscale_variables({f'c_EC_0_a{i}': f_1[c_EC_index]})[f'c_EC_0_a{i}'] * SEI.k_f_s
            J = f_1[j_int_index] + f_1[j_sei_index] 
            F_SEI.append((f_1[j_sei_index] - self.j_SEI(material, i_0s, f_1.phi_s, f_1.phi_e, f_1.temp, J, f_1[delta_index]))*test[j_sei_index]*dx)
            F_SEI.append(self.delta_growth(DT, f_0[delta_index], f_1[delta_index], f_1[j_sei_index], test[delta_index], dx))

        if DT:
            F_SEI.extend(spectral_model.wf(f_0, f_1, test, dx, DT, active_material, self))
        else:
            F_SEI.extend(spectral_model.wf_0(f_0, f_1, test, dx))
        
        return F_SEI


    class SpectralLagrangeModel_EC():
        
        def __init__(self, order=2):
            self.order = order
            self.poly = Lagrange(order)
            self.f = self.poly.f
            self.df = self.poly.df
            self.xf = self.poly.xf
            self.xdf = self.poly.xdf
            self.build_matrix()

        def _get_n_mat(self, f):
            n = 0
            for name in f._fields:
                if name.startswith('c_EC_0_a'):
                    n += 1
            return n
        
        def fields(self, n_mat):
            field_list =  ['c_EC_{}_a{}'.format( order, material) for material in range(n_mat) for order in range(self.order)]
            return field_list

        def initial_guess(self, f_0, c_0):
            n_mat = self._get_n_mat(f_0)
            if n_mat!=0:
                for material in range(n_mat):
                    for j in range(self.order):
                        c_EC_index = f_0._fields.index(f'c_EC_{j}_a{material}')
                        assign(f_0[c_EC_index], interpolate(Constant(c_0), f_0[c_EC_index].function_space()))

        def build_matrix(self):
            J = numpy.zeros((self.order+1, self.order+1))
            K = numpy.zeros((self.order+1, self.order+1))
            L = numpy.zeros((self.order+1, self.order+1))
            M = numpy.zeros((self.order+1, self.order+1))
            N = numpy.zeros((self.order+1, self.order+1))
            P = numpy.zeros(self.order+1)

            for i in range(self.order+1):

                P[i] = polyval(0, self.f[i])

                for j in range(self.order+1):

                    J[i,j] = polyval(0, polymul(self.f[i], self.f[j]))
                    K[i,j] = polyval(1,polyint(polymul(self.f[i], self.f[j])))
                    L[i,j] = polyval(1,polyint(polymul(self.df[i], self.f[j])))
                    M[i,j] = polyval(1,polyint(polymul(self.xdf[i], self.f[j])))
                    N[i,j] = polyval(1,polyint(polymul(self.df[i], self.df[j])))

            J_d = J[0:-1,0:-1]
            K_d = K[0:-1,0:-1]
            L_d = L[0:-1,0:-1]
            M_d = M[0:-1,0:-1]
            N_d = N[0:-1,0:-1]
            P_d = P[0:-1]

            self.D = K_d
            self.K1 = L_d - M_d - J_d
            self.K2 = N_d
            self.P = P_d

        def wf_0(self, f_0, f_1, test, dx):
            n_mat = self._get_n_mat(f_0)
            F_EC_0 = []
            for material in range(n_mat):
                for j in range(self.order):
                    c_EC_index = f_0._fields.index(f'c_EC_{j}_a{material}')
                    F_EC_0.append([(f_1[c_EC_index] - f_0[c_EC_index]) * test[c_EC_index] * dx])
            return F_EC_0

        def wf(self, f_0, f_1, test, dx, DT, materials, nd_model):

            F_EC_ret = []
            for k, material in enumerate(materials):
                SEI = material.electrode.SEI
                c_EC_index = f_0._fields.index(f'c_EC_0_a{k}')
                j_SEI_index = f_0._fields.index(f'j_sei_a{k}')
                delta_index = f_0._fields.index(f'delta_a{k}')
                self.K = 1 / f_1[delta_index] * DT.dt(f_0[delta_index], f_1[delta_index]) * self.K1 + SEI.D_EC * nd_model.t_c / (nd_model.delta_sei_a[k]**2 *f_1[delta_index]**2) * self.K2
                for j in range(self.order):
                    F_EC = 0
                    for i in range(self.order):
                        F_EC += self.D[i, j] * DT.dt(f_0[c_EC_index+i], f_1[c_EC_index+i]) * test[c_EC_index+j] * dx(metadata={"quadrature_degree":2})
                        F_EC += self.K[i, j] * f_1[c_EC_index+i] * test[c_EC_index+j] * dx(metadata={"quadrature_degree":2})
                        F_EC -= self.K[i, j] * nd_model.scale_variables({f'c_EC_{j}_a{k}': SEI.c_EC_sln * SEI.eps})[f'c_EC_{j}_a{k}'] * test[c_EC_index+j] * dx(metadata={"quadrature_degree":2})

                    F_EC -= 2*SEI.rho/ (SEI.M*nd_model.c_sei_a[k]) / f_1[delta_index] * f_1[j_SEI_index] * self.P[j] * test[c_EC_index+j] * dx(metadata={"quadrature_degree":2})
                    
                    F_EC_ret.append(F_EC)

            return F_EC_ret
