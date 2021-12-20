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

# from cideMOD.helpers.miscellaneous import Lagrange
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

    def __init__(self, order=2):
        self.order = order
        # self.SLM = self.SpectralLegendreModel_EC(order)
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

    def set_up(self, anode, F, R):
        SEI = anode.SEI
        self.ohm_SEI = SEI.R
        self.delta_0_SEI = SEI.delta0
        self.kappa_SEI = SEI.k
        self.M_SEI = SEI.M
        self.rho_SEI = SEI.rho
        self.i_0_SEI = SEI.i_0
        self.U_SEI = SEI.U
        self.D_EC = SEI.D_EC
        self.eps_SEI = SEI.eps
        self.c_EC_sln = SEI.c_EC_sln
        self.k_f_s = SEI.k_f_s

        self.F = F
        self.R = R
        self.alpha = SEI.beta
        self.L = anode.L
        self.A = anode.area
        # self.FDM = self.finiteDifferenceMethod(self)

    def j_SEI(self, j_SEI, test, dx, i_0s, phi_s, phi_e, T, J, G_film):

        eta = self.SEI_overpotential(phi_s, phi_e, J, G_film)

        return j_SEI * test * dx + i_0s * exp(-self.alpha * self.F / self.R / T * eta) * test * dx

    def SEI_overpotential(self, phi_s, phi_e, J, G_film):

        return phi_s - phi_e - self.U_SEI - J * G_film

    def delta_growth(self, DT, delta_0, delta_1, j_SEI, test, dx):
        
        if DT:
            return DT.dt(delta_0, delta_1) * test * dx + j_SEI * self.M_SEI / (2 * self.F * self.rho_SEI) * test * dx
        else:
            return (delta_0 - delta_1) * test * dx

    def equations(self, f_0, f_1, test, dx, active_material, DT = None):

        F_SEI = []
        for i, material in enumerate(active_material):
            j_sei_index = f_1._fields.index(f'j_sei_a{i}')
            delta_index = f_1._fields.index(f'delta_a{i}')
            c_EC_index = f_1._fields.index(f'c_EC_0_a{i}')
            j_int_index = f_1._fields.index(f'j_Li_a{i}')
            
            i_0s = self.F * f_1[c_EC_index] * self.k_f_s
            G_film = self.ohm_SEI + f_1[delta_index] / self.kappa_SEI
            J = f_1[j_int_index] + f_1[j_sei_index] 
            F_SEI.append(self.j_SEI(f_1[j_sei_index], test[j_sei_index], dx, i_0s, f_1.phi_s, f_1.phi_e, f_1.temp, J, G_film))
            F_SEI.append(self.delta_growth(DT, f_0[delta_index], f_1[delta_index], f_1[j_sei_index], test[delta_index], dx))

        # if DT:
        #     F_SEI.extend(self.SLM.wf_implicit_coupling(f_0, f_1, test, dx, DT, active_material, self))
        # else:
        #     F_SEI.extend(self.SLM.wf_0(f_0, f_1, test, dx))

        if DT:
            F_SEI.extend(self.SLagM.wf(f_0, f_1, test, dx, DT, active_material, self))
        else:
            F_SEI.extend(self.SLagM.wf_0(f_0, f_1, test, dx))
        
        return F_SEI

    class finiteDifferenceMethod():

        def __init__(self, SEI, start=0, stop=1, nStep=20):
            self.SEI = SEI
            self.y_t = [self.SEI.eps_SEI*self.SEI.c_EC_sln for i in range(nStep+1)]
            self.nStep = nStep
            self.h = (stop-start) / nStep

            self.K = numpy.zeros((nStep+1, nStep+1))
            self.b = numpy.zeros(nStep+1)
            self.x = numpy.zeros(nStep+1)

            self.K[nStep, nStep] = 1
            self.x[0] = 0; self.x[nStep] = 1
            self.b[nStep] = self.SEI.eps_SEI*self.SEI.c_EC_sln

        def build_linear_system(self):
            for i in range(0, self.nStep):
                self.x[i] = self.x[i-1] + self.h
                if i!=0:
                    self.K[i, i-1] = 1 + self.h*self.A/2*(1-self.x[i])
                self.K[i, i] = - (2 + self.h**2*self.B + (i==0) * self.A*self.h * (2 + self.A*self.h))
                self.K[i, i+1] = 1 - self.h*self.A/2*(1-self.x[i]) + (i==0) * (1 + self.h*self.A/2)
                self.b[i] = - self.h**2 * self.B *self.y_t[i] + (i==0) * (2*self.h*self.D + self.h**2*self.A*self.D)


        def set_constants(self, f_1, f_0, dx, materials, DT):

            for k, material in enumerate(materials):
                j_SEI_index = f_0._fields.index('j_sei_a{}'.format(k))
                delta_index = f_0._fields.index('delta_a{}'.format(k))
                v = - f_1[j_SEI_index] / 2 / self.SEI.F * self.SEI.M_SEI / self.SEI.rho_SEI
                self.A = assemble(f_1[delta_index] / self.SEI.D_EC * v * dx)
                self.B = assemble(f_1[delta_index]**2 / self.SEI.D_EC / DT.delta_t * dx)
                self.D = assemble(- f_1[delta_index] * f_1[j_SEI_index] / self.SEI.F / self.SEI.D_EC * dx)
                self.build_linear_system()
        
        def solve(self, f_1, f_0, dx, materials, DT):
            self.set_constants(f_1, f_0, dx, materials, DT)
            AA = self.K.T @ self.K
            bA = self.b @ self.K
            D, U = numpy.linalg.eigh(AA)
            Ap = (U * numpy.sqrt(D)).T
            bp = bA @ U / numpy.sqrt(D)
            self.y_t = numpy.linalg.lstsq(Ap, bp)[0]
            # self.y_t = numpy.linalg.solve(self.K, self.b) 
            # self.y_t = scipy.sparse.linalg.spsolve(self.K, self.b)
            return self.y_t


    class SpectralLegendreModel_EC():
        """Particle Intercalation resolved with Legendre Polinomials.
        Diffusion is modeled with Fick's law.
        """
        def __init__(self, order):
            
            self.order = order
            self.build_legendre(order)

        def wf_0(self, f_0, f_1, test, dx):

            n_mat = _get_n_mat(f_0)
            F_EC_0 = []
            for material in range(n_mat):
                c_EC_index = f_0._fields.index('c_EC_0_a{}'.format(material))
                for j in range(self.order):
                    F_EC_0.append([f_1[c_EC_index+j] * test[c_EC_index+j] * dx])
            return F_EC_0

        def c_EC_surf(self, f, c_0):

            n_mat = _get_n_mat(f)
            c_EC_surf = []
            for material in range(n_mat):
                c_EC_surf_i = 0
                c_EC_index = f._fields.index('c_EC_0_a{}'.format(material))
                for i in range(self.order):
                    c_EC_surf_i += f[c_EC_index+i] * self.P[i]
                c_EC_surf.append(c_EC_surf_i+ c_0)

            return c_EC_surf

        def c_EC_surf_out(self, f, c_0):

            n_mat = _get_n_mat(f)
            c_EC_surf = []
            for material in range(n_mat):
                c_EC_surf_i = 0
                c_EC_index = f._fields.index('c_EC_0_a{}'.format(material))
                for i in range(self.order):
                    c_EC_surf_i += f[c_EC_index+i] * self.Q[i]
                c_EC_surf.append(c_EC_surf_i +c_0)

            return c_EC_surf

        def build_legendre(self, order):

            """Builds mass matrix, stiffness matrix and boundary vector using Legendre Polinomials.
            The domain used is [0,1] and only pair Legendre polinomials are used to enforce zero flux at x=0.

            Args:
                order (int): number of Legendre polinomials to use

            Returns:
                tuple: Mass matrix, Stiffness matrix, boundary vector
            """
            # Init matrix and vector
            K = numpy.zeros((order, order))
            L = numpy.zeros((order, order))
            M = numpy.zeros((order, order))
            N = numpy.zeros((order, order))
            P = numpy.zeros(order)
            Q = numpy.zeros(order)       

            for n in range(order):
                L_n = numpy.zeros(2*order, dtype=int)
                L_n[2*n+1] = 1  # Only even polinomials used
                D_n = legder(L_n)  # dL/dr
                D_nx = legmulx(D_n)  # r*L

                P[n] = legval(-1.0, L_n)  # L(-1)
                Q[n] = legval(0.0, L_n)  # L(0)

                for m in range(order):
                    L_m = numpy.zeros(2*order, dtype=int)
                    L_m[2*m+1] = 1  # Only even polinomials used
                    D_m = legder(L_m)
                    # integral(-1, 0, L_n*L_m)
                    K[n, m] = legval(0.0, legint(legmul(L_n, L_m), lbnd=-1))
                    # integral(-1, 0, dL_n/dr*dL_m/dr)
                    L[n, m] = legval(0.0, legint(legmul(D_n, D_m), lbnd=-1))
                    # integral(-1, 0, r*L_n*dL_m/dr)
                    M[n, m] = legval(0.0, legint(legmul(D_nx, L_m), lbnd=-1))+legval(0.0, legint(legmul(D_n, L_m), lbnd=-1))
                    # integral(-1, 0, r*L_n*dL_m/dr)
                    N[n, m] = legval(0.0, legint(legmul(D_n, L_m), lbnd=-1))

            self.K = K
            self.L = L
            self.M = M
            self.N = N
            self.P = P
            self.Q = Q
    
        def wf_implicit_coupling(self, f_0, f_1, test, dx : Measure, DT, materials:List, SEI):

            F_EC_ret = []
            for k, material in enumerate(materials):
                c_EC_index = f_0._fields.index('c_EC_0_a{}'.format(k))
                j_SEI_index = f_0._fields.index('j_sei_a{}'.format(k))
                delta_index = f_0._fields.index('delta_a{}'.format(k))
                v = - f_1[j_SEI_index] / 2 / SEI.F * SEI.M_SEI / SEI.rho_SEI
                for j in range(self.order):
                    F_EC = 0
                    for i in range(self.order):

                        F_EC += self.K[i, j] * f_1[delta_index]**2 * DT.dt(f_0[c_EC_index+i], f_1[c_EC_index+i])* test[c_EC_index+j] * dx(metadata={"quadrature_degree":2})
                        F_EC -= f_1[delta_index] * v * f_1[c_EC_index+i] * self.M[i, j] * test[c_EC_index+j] * dx(metadata={"quadrature_degree":2})
                        F_EC += self.L[i, j] * SEI.D_EC * f_1[c_EC_index+i] * test[c_EC_index+j] * dx(metadata={"quadrature_degree":2})
                        F_EC += f_1[delta_index] * v * f_1[c_EC_index+i] * self.N[i, j] * test[c_EC_index+j] * dx(metadata={"quadrature_degree":2})

                    F_EC += f_1[delta_index] * (f_1[j_SEI_index] / SEI.F - v * f_1[c_EC_index+i] * self.P[j]) * self.P[j] * test[c_EC_index+j] * dx(metadata={"quadrature_degree":2})
                    
                    F_EC_ret.append(F_EC)

            return F_EC_ret
        

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

        def wf(self, f_0, f_1, test, dx : Measure, DT, materials:List, SEI):

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
                        F_EC -= self.K[i, j] * SEI.c_EC_sln * SEI.eps_SEI * test[c_EC_index+j] * dx(metadata={"quadrature_degree":2})

                    F_EC -= 1 / f_1[delta_index] * f_1[j_SEI_index] / SEI.F * self.P[j] * test[c_EC_index+j] * dx(metadata={"quadrature_degree":2})
                    
                    F_EC_ret.append(F_EC)

            return F_EC_ret


###############################################################################
# It is also in helpers/miscellaneous but i cant import it with relative paths

class Lagrange():
    
    def __init__(self, order, interval=[0, 1]):
        self.order = order
        self.points = numpy.linspace(interval[0], interval[1], num=order+1)
        self.f_vector()
        self.df_vector()
        self.xf_vector()
        self.xdf_vector()
    
    def simple_poly(self, point):
        poly_c = [1]
        for i in self.points:
            if i!=point:
                poly_c = polymul(poly_c, [-i/(point-i), 1/(point-i)])
        return poly_c
    
    def getPolyFromCoeffs(self, c):
        assert len(c)==self.order+1, "The length of the coefficients list has to be: "+str(self.order+1)
        poly = Polynomial([0])
        for k in range(self.order+1):
            poly = polyadd(poly, c[k]*self.f[k])
        return poly

    def f_vector(self):
        self.f = []
        for k in range(self.order+1):
            self.f.append(self.simple_poly(self.points[k]))
        
    def xf_vector(self):
        self.xf = []
        for k in range(self.order+1):
            self.xf.append(polymul([0,1],self.simple_poly(self.points[k])))

    def df_vector(self):
        self.df = []
        for k in range(self.order+1):
            self.df.append(polyder(self.simple_poly(self.points[k])))

    def xdf_vector(self):
        self.xdf = []
        for k in range(self.order+1):
            self.xdf.append(polymul([0,1],polyder(self.simple_poly(self.points[k]))))
