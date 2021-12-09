#
# Copyright (c) 2021 CIDETEC Energy Storage.
#
# This file is part of PXD.
#
# PXD is free software: you can redistribute it and/or modify
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
# along with this program. If not, see <http://www.gnu.org/licenses/>.#
from dolfin import *

from typing import List

import numpy
from numpy.polynomial.legendre import *
from ufl.coefficient import Coefficient

from PXD.models.base.base_particle_models import StrongCoupledPM


class SpectralLegendreModel(StrongCoupledPM):
    """Particle Intercalation resolved with Legendre Polinomials.
    Diffusion is modeled with Fick's law.
    """
    def __init__(self, order):
        self.order = order
        self.build_legendre(order)

    def _get_domain(self, electrode):
        allowed = ('anode', 'cathode', 'particle')
        assert electrode in ('anode', 'cathode', 'particle') , "Keyword 'electrode' must be either 'anode' or 'cathode'"
        domain = ['a', 'c', ''][allowed.index(electrode)]
        return domain

    def _get_n_mat(self, f, domain):
        n = 0
        for name in f._fields:
            if name.startswith(f'c_s_0_{domain}'):
                n += 1
        return n
    
    def fields(self, n_mat, electrode):
        domain = self._get_domain(electrode)
        return ['c_s_{}_{}{}'.format( order, domain, material) for material in range(n_mat) for order in range(self.order)]

    def initial_guess(self, f_0, electrode, c_s_ini):
        domain = self._get_domain(electrode)
        n_mat = self._get_n_mat(f_0, domain)
        if n_mat!=0:
            assert isinstance(c_s_ini, (list, tuple)) , "Keyword 'c_s_ini' must be an iterable"
            assert len(c_s_ini) == n_mat , "Keyword 'c_s_ini' must be a list of length {}".format(n_mat)
            for i, name in enumerate(f_0._fields):
                for j, c_ini in enumerate(c_s_ini):
                    if name == f'c_s_0_{domain}{j}':
                        f_0[i].assign(project(c_ini,f_0[i].function_space()))

    def build_legendre(self, order):
        """Builds mass matrix, stiffness matrix and boundary vector using Legendre Polinomials.
        The domain used is [0,1] and only pair Legendre polinomials are used to enforce zero flux at x=0.

        Args:
            order (int): number of Legendre polinomials to use

        Returns:
            tuple: Mass matrix, Stiffness matrix, boundary vector
        """
        # Init matrix and vector
        M = numpy.zeros((order, order))
        K = numpy.zeros((order, order))
        P = numpy.zeros(order)
        for n in range(order):
            L_n = numpy.zeros(2*order, dtype=int)
            L_n[2*n] = 1  # Only pair polinomials used

            D_n = legder(L_n)  # dL/dr
            L_nx = legmulx(L_n)  # r*L
            D_nx = legmulx(D_n)  # r*dL/dr

            P[n] = legval(1.0, L_n)  # L(1)

            for m in range(order):
                L_m = numpy.zeros(2*order, dtype=int)
                L_m[2*m] = 1

                D_m = legder(L_m)
                L_mx = legmulx(L_m)
                D_mx = legmulx(D_m)

                # integral(0, 1, r^2*L_n*L_m)
                M[n, m] = legval(1.0, legint(legmul(L_nx, L_mx)))
                # integral(0, 1, r^2*dL_n/dr*dL_m/dr)
                K[n, m] = legval(1.0, legint(legmul(D_nx, D_mx)))

        self.M = M
        self.K = K
        self.P = P

    def wf_0(self, f_0, f_1, test, electrode, dx):
        domain = self._get_domain(electrode)
        n_mat = self._get_n_mat(f_0, domain)
        F_c_s_0 = []
        for material in range(n_mat):
            c_s_index = f_0._fields.index('c_s_0_{}{}'.format(domain, material))
            F_c_s_0.append([(f_1[c_s_index] - f_0[c_s_index]) * test[c_s_index] * dx])
            for j in range(1, self.order):
                F_c_s_0.append([f_1[c_s_index+j] * test[c_s_index+j] * dx])
        return F_c_s_0

    def wf_implicit_coupling(self, f_0, f_1, test, electrode, dx : Measure, DT, materials:List, F, R):
        domain = self._get_domain(electrode)
        F_c_s_ret = []
        for k, material in enumerate(materials):
            c_s_index = f_1._fields.index('c_s_0_{}{}'.format(domain, k))
            j_li_index = f_1._fields.index('j_Li_{}{}'.format(domain, k))
            D_s_eff = self.get_value(material.D_s, f_1, electrode, material) 
            if 'temp' in f_1._fields:
                D_s_eff= D_s_eff* exp(material.D_s_Ea*(1/material.D_s_Tref - 1/f_1.temp)/R)
            for j in range(self.order):
                F_c_s = 0
                F_c_s += self.M[0, j] * DT.dt(f_0[c_s_index], f_1[c_s_index]) * test[c_s_index+j] * dx(metadata={"quadrature_degree":2})
                for i in range(1, self.order):
                    F_c_s -= self.M[0, j] * DT.dt(f_0[c_s_index+i], f_1[c_s_index+i]) * test[c_s_index+j] * dx(metadata={"quadrature_degree":2})

                for i in range(1, self.order):

                    F_c_s += self.M[i, j] * DT.dt(f_0[c_s_index+i], f_1[c_s_index+i]) * test[c_s_index+j] * dx(metadata={"quadrature_degree":2})

                    F_c_s += (D_s_eff / material.R_s ** 2) * self.K[i, j] * f_1[c_s_index+i] * test[c_s_index+j] * dx(metadata={"quadrature_degree":2})

                F_c_s += (1. / material.R_s) * (1. / F) * self.P[j] * f_1[j_li_index] * test[c_s_index+j] * dx(metadata={"quadrature_degree":2})
                F_c_s_ret.append(F_c_s)

        return F_c_s_ret

    def wf_explicit_coupling(self, f_0, f_1, test, electrode, dx):
        domain = self._get_domain(electrode)
        n_mat = self._get_n_mat(f_0, domain)
        F_c_s_aux = []
        for material in range(n_mat):
            c_s_index = f_0._fields.index('c_s_0_{}{}'.format(domain, material))
            for j in range(0, self.order):
                F_c_s_aux.append(f_1[c_s_index+j] * test[c_s_index+j] * dx - f_0[c_s_index+j] * test[c_s_index+j] * dx)
        return F_c_s_aux

    def c_s_surf(self, f, electrode):
        domain = self._get_domain(electrode)
        n_mat = self._get_n_mat(f, domain)
        c_s_surf = []
        for material in range(n_mat):
            c_s_index = f._fields.index('c_s_0_{}{}'.format(domain, material))
            c_s_surf_i = f[c_s_index]
            c_s_surf.append(c_s_surf_i)
        return c_s_surf

    def Li_amount(self, f, electrode, materials:List, dx, volume_factor=1):
        domain = self._get_domain(electrode)
        #Build particle integral
        Int = numpy.zeros(self.order)
        for n in range(self.order):
            L_n = numpy.zeros(2*self.order, dtype=float)
            L_n[2*n] = 1  # Only pair polinomials used
            L_nxx = legmulx(legmulx(L_n)) # L_n*r^2
            Int[n] = legval(1.0, legint(L_nxx)) # integral(0,1, L_n*r^2)

        li_total = []
        for k, material in enumerate(materials):
            particle_total = 0
            c_s_index = f._fields.index('c_s_0_{}{}'.format(domain, k))
            for i in range(self.order):
                particle_total += f[c_s_index+i] * Int[i] # particle_integral(c_s)/ V
            c_s_total = assemble(volume_factor*particle_total * material.eps_s * dx ) 
            li_total.append(c_s_total)
        return sum(li_total)

    def get_value(self, value, f, electrode, mat):
        if isinstance(value, (int, float, Coefficient)):
            return value
        elif isinstance(value, str):
            x = self.c_s_surf(f, electrode)[mat.index] / mat.c_s_max
            return eval(value)
        elif callable(value):
            x = self.c_s_surf(f, electrode)[mat.index] / mat.c_s_max
            return value(x)
        else:
            raise Exception('Unknown type of parameter')


class NondimensionalSpectralModel(SpectralLegendreModel):
    def wf_implicit_coupling(self, f_0, f_1, test, electrode, dx : Measure, DT, materials:List, nd_model):
        domain = self._get_domain(electrode)
        F_c_s_ret = []
        for k, material in enumerate(materials):
            c_s_index = f_0._fields.index('c_s_0_{}{}'.format(domain, k))
            j_li_index = f_0._fields.index('j_Li_{}{}'.format(domain, k))
            mat_params = nd_model.material_parameters(material)
            T = nd_model.unscale_variables({'T': f_1.temp})['T']
            if material.D_s_Ea == 0:
                D_s_eff = self.get_value(material.D_s, material.index, f_1, electrode)
            else:
                D_s_eff = self.get_value(material.D_s, material.index, f_1, electrode) * exp(material.D_s_Ea*(1/material.D_s_Tref - 1/T)/nd_model.cell.R)
            for j in range(self.order):
                F_c_s = 0
                F_c_s += (1/mat_params['tau_s']) * self.M[0, j] * DT.dt(f_0[c_s_index], f_1[c_s_index]) * test[c_s_index+j] * dx(metadata={"quadrature_degree":2})
                for i in range(1, self.order):
                    F_c_s -= (1/mat_params['tau_s'])* self.M[0, j] * DT.dt(f_0[c_s_index+i], f_1[c_s_index+i]) * test[c_s_index+j] * dx(metadata={"quadrature_degree":2})
                for i in range(1, self.order):

                    F_c_s += (1/mat_params['tau_s']) * self.M[i, j] * DT.dt(f_0[c_s_index+i], f_1[c_s_index+i]) * test[c_s_index+j] * dx(metadata={"quadrature_degree":2})

                    F_c_s += (D_s_eff / mat_params['D_s_ref']) * self.K[i, j] * f_1[c_s_index+i] * test[c_s_index+j] * dx(metadata={"quadrature_degree":2})

                F_c_s += mat_params['S'] * self.P[j] * f_1[j_li_index] * test[c_s_index+j] * dx(metadata={"quadrature_degree":2})
                F_c_s_ret.append(F_c_s)

        return F_c_s_ret

    def get_value(self, value, index, f, electrode):
        if isinstance(value, (int, float, Coefficient)):
            return value
        elif isinstance(value, str):
            x = self.c_s_surf(f, electrode)[index]
            return eval(value)
        elif callable(value):
            x = self.c_s_surf(f, electrode)[index]
            return value(x)
        else:
            raise Exception('Unknown type of parameter')


class StressEnhancedSpectralModel(SpectralLegendreModel):
    def build_legendre(self, order):
        """Builds mass matrix, stiffness matrix and boundary vector using Legendre Polinomials.
        The domain used is [0,1] and only pair Legendre polinomials are used to enforce zero flux at x=0.

        Args:
            order (int): number of Legendre polinomials to use

        Returns:
            tuple: Mass matrix, Stiffness matrix, boundary vector
        """
        # Init matrix and vector
        M = numpy.zeros((order, order))
        K = numpy.zeros((order, order))
        S = numpy.zeros((order, order, order))
        P = numpy.zeros(order)
        avg = numpy.zeros(order)
        for n in range(order):
            L_n = numpy.zeros(2*order, dtype=float)
            L_n[2*n] = 1  # Only pair polinomials used

            D_n = legder(L_n)  # dL/dr
            L_nx = legmulx(L_n)  # r*L
            D_nx = legmulx(D_n)  # r*dL/dr

            P[n] = legval(1.0, L_n)  # L(1)
            avg[n] = legval(1.0, legint(L_n)) # integral(0,1, L_n)

            for m in range(order):
                L_m = numpy.zeros(2*order, dtype=float)
                L_m[2*m] = 1

                D_m = legder(L_m)
                L_mx = legmulx(L_m)
                D_mx = legmulx(D_m)

                # integral(0, 1, r^2*L_n*L_m)
                M[n, m] = legval(1.0, legint(legmul(L_nx, L_mx)))
                # integral(0, 1, r^2*dL_n/dr*dL_m/dr)
                K[n, m] = legval(1.0, legint(legmul(D_nx, D_mx)))

                for k in range(order):
                    L_k = numpy.zeros(2*order, dtype=float)
                    L_k[2*k] = 1
                    # integral(0, 1, r^2*dL_n/dr*dL_m/dr*L_k)
                    S[n, m, k] = legval(1.0, legint(legmul(legmul(D_nx, D_mx),L_k)))

        self.M = M
        self.K = K
        self.P = P
        self.S = S
        self.avg = avg

    def theta(self, material, R, T):
        if material.omega is not None and material.young is not None and material.poisson is not None:
            return Constant((material.omega*2*material.young*material.omega)/(R*9*(1-material.poisson)))/T
        else:
            raise Exception('Material {} does not have mechanical properties'.format(material.index))
            # print(f'Material {material.index} does not have mechanical properties \n Ommiting particle deformation ...')
            # return 0

    def wf_implicit_coupling(self, f_0, f_1, test, electrode, dx : Measure, DT, materials, F, R):
        domain = self._get_domain(electrode)
        F_c_s_ret = []
        for k, material in enumerate(materials):
            c_s_index = f_0._fields.index('c_s_0_{}{}'.format(domain, k))
            j_li_index = f_0._fields.index('j_Li_{}{}'.format(domain, k))
            theta = self.theta(material, R, f_1.temp)
            D_s_eff = self.get_value(material.D_s, material.index, f_1, electrode) * exp(material.D_s_Ea*(1/material.D_s_Tref - 1/f_1.temp)/R)
            for j in range(self.order):
                F_c_s = 0
                for i in range(self.order):
                    F_c_s += self.M[i, j] * DT.dt(f_0[c_s_index+i], f_1[c_s_index+i]) * test[c_s_index+j] * dx(metadata={"quadrature_degree":2})
                    F_c_s += (1-theta*material.c_s_ini)*(D_s_eff / material.R_s ** 2) * self.K[i, j] * f_1[c_s_index+i] * test[c_s_index+j] * dx(metadata={"quadrature_degree":2})
                    for l in range(self.order):
                        F_c_s += theta*(D_s_eff / material.R_s ** 2) * self.S[i, j, l] * f_1[c_s_index+i] * f_1[c_s_index+l] *test[c_s_index+j] * dx(metadata={"quadrature_degree":3})
                F_c_s += (1. / material.R_s) * (1. / F) * self.P[j] * f_1[j_li_index] * test[c_s_index+j] * dx(metadata={"quadrature_degree":2})
                F_c_s_ret.append(F_c_s)

        return F_c_s_ret

    def get_average_c_s(self, f, domain):
        assert domain.tag in ('a','c')
        c_s = []
        for i, materials in enumerate(domain.active_material):
            c_s_index = f._fields.index('c_s_0_{}{}'.format(domain.tag, i))
            F_avg = 0
            for j in range(self.order):
                F_avg += f[c_s_index+j]*self.avg[j]
            c_s.append(F_avg)
        return c_s
