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
from dolfin import *

from typing import List

import numpy
from numpy.polynomial.legendre import *

from cideMOD.numerics.polynomials import Lagrange
from numpy.polynomial.polynomial import *
from ufl.operators import exp, sinh

__all__= [
    "SEI",
]

def _get_n_mat(f):
        n = 0
        for name in f._fields:
            if name.startswith('c_EC_0_a'):
                n += 1
        return n

class SEI:
    """SEI (Solid-Electrolyte Interphase) growth model limited by solvent diffusion through the SEI.

    Args:
        order: Order of the inner spectral model for solvent diffusion, default is 2.

    References:
        1: Safari et al. - 2009 - Multimodal Physics-Based Aging Model for Life Prediction of Li-Ion Batteries
        
    """
    def __init__(self, order=2):
        self.order = order
        self.SLagM = self.SpectralLagrangeModel_EC(order)

    def fields(self, n_mat):    
        field_list =  ['c_EC_{}_a{}'.format( order, material) for material in range(n_mat) for order in range(self.order)]
        return field_list

    def initial_guess(self, f_0, c_0):
        n_mat = _get_n_mat(f_0)
        if n_mat!=0:
            for material in range(n_mat):
                c_EC_index = f_0._fields.index('c_EC_0_a{}'.format(material))
                for j in range(self.order):
                    assign(f_0[c_EC_index+j], interpolate(Constant(c_0), f_0[c_EC_index+j].function_space()))

    def j_SEI(self, j_SEI, test, dx, i_0s, phi_s, phi_e, T, J, G_film, SEI, F, R):

        eta = self.SEI_overpotential(phi_s, phi_e, J, G_film, SEI)

        return j_SEI * test * dx + i_0s * exp(-(SEI.beta * F / R) * eta / T) * test * dx

    def SEI_overpotential(self, phi_s, phi_e, J, G_film, SEI):

        return phi_s - phi_e - SEI.U - J * G_film

    def delta_growth(self, DT, delta_0, delta_1, j_SEI, SEI, F, test, dx):
        
        if DT:
            return DT.dt(delta_0, delta_1) * test * dx + j_SEI * SEI.M / (2 * F * SEI.rho) * test * dx
        else:
            return (delta_0 - delta_1) * test * dx

    def equations(self, f_0, f_1, test, dx, active_material, F, R, DT = None):
        if len(active_material)>0:
            assert hasattr(active_material[0].electrode,'SEI')
        SEI = active_material[0].electrode.SEI
        F_SEI = []
        for i, material in enumerate(active_material):
            j_sei_index = f_1._fields.index(f'j_sei_a{i}')
            delta_index = f_1._fields.index(f'delta_a{i}')
            c_EC_index = f_1._fields.index(f'c_EC_0_a{i}')
            j_int_index = f_1._fields.index(f'j_Li_a{i}')
            
            i_0s = F * f_1[c_EC_index] * SEI.k_f_s
            G_film = SEI.R + f_1[delta_index] / SEI.kappa
            J = f_1[j_int_index] + f_1[j_sei_index] 
            F_SEI.append(self.j_SEI(f_1[j_sei_index], test[j_sei_index], dx, i_0s, f_1.phi_s, f_1.phi_e, f_1.temp, J, G_film, SEI, F, R))
            F_SEI.append(self.delta_growth(DT, f_0[delta_index], f_1[delta_index], f_1[j_sei_index], SEI, F ,test[delta_index], dx))

        if DT:
            F_SEI.extend(self.SLagM.wf(f_0, f_1, test, dx, DT, active_material, SEI, F))
        else:
            F_SEI.extend(self.SLagM.wf_0(f_0, f_1, test, dx))
        
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

            n_mat = _get_n_mat(f_0)
            F_EC_0 = []
            for material in range(n_mat):
                c_EC_index = f_0._fields.index('c_EC_0_a{}'.format(material))
                for j in range(self.order):
                    F_EC_0.append([(f_1[c_EC_index+j] - f_0[c_EC_index+j]) * test[c_EC_index+j] * dx])
            return F_EC_0

        def wf(self, f_0, f_1, test, dx : Measure, DT, materials:List, SEI, F):

            F_EC_ret = []
            for k, material in enumerate(materials):
                c_EC_index = f_0._fields.index('c_EC_0_a{}'.format(k))
                j_SEI_index = f_0._fields.index('j_sei_a{}'.format(k))
                delta_index = f_0._fields.index('delta_a{}'.format(k))
                self.K = 1 / f_1[delta_index] * DT.dt(f_0[delta_index], f_1[delta_index]) * self.K1 + 1 / f_1[delta_index]**2 * SEI.D_EC * self.K2
                for j in range(self.order):
                    F_EC = 0
                    for i in range(self.order):

                        F_EC += self.D[i, j] * DT.dt(f_0[c_EC_index+i], f_1[c_EC_index+i]) * test[c_EC_index+j] * dx(metadata={"quadrature_degree":2})
                        F_EC += self.K[i, j] * f_1[c_EC_index+i] * test[c_EC_index+j] * dx(metadata={"quadrature_degree":2})
                        F_EC -= self.K[i, j] * SEI.c_EC_sln * SEI.eps * test[c_EC_index+j] * dx(metadata={"quadrature_degree":2})

                    F_EC -= 1 / f_1[delta_index] * f_1[j_SEI_index] / F * self.P[j] * test[c_EC_index+j] * dx(metadata={"quadrature_degree":2})
                    
                    F_EC_ret.append(F_EC)

            return F_EC_ret
