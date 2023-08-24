#
# Copyright (c) 2023 CIDETEC Energy Storage.
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
import numpy
from numpy.polynomial.polynomial import *


class Lagrange():

    def __init__(self, order, interval=[0, 1]):
        self.order = order
        self.points = numpy.linspace(interval[0], interval[1], num=order + 1)
        self.f_vector()
        self.df_vector()
        self.xf_vector()
        self.xdf_vector()

    def simple_poly(self, point):
        poly_c = [1]
        for i in self.points:
            if i != point:
                poly_c = polymul(poly_c, [-i / (point - i), 1 / (point - i)])
        return poly_c

    def getPolyFromCoeffs(self, c):
        if len(c) != self.order + 1:
            raise ValueError(f"The length of the coefficients list must be: {self.order + 1}")
        poly = Polynomial([0])
        for k in range(self.order + 1):
            poly = polyadd(poly, Polynomial(c[k] * self.f[k]))
        return poly

    def f_vector(self):
        self.f = []
        for k in range(self.order + 1):
            self.f.append(self.simple_poly(self.points[k]))

    def xf_vector(self):
        self.xf = []
        for k in range(self.order + 1):
            self.xf.append(polymul([0, 1], self.simple_poly(self.points[k])))

    def df_vector(self):
        self.df = []
        for k in range(self.order + 1):
            self.df.append(polyder(self.simple_poly(self.points[k])))

    def xdf_vector(self):
        self.xdf = []
        for k in range(self.order + 1):
            self.xdf.append(polymul([0, 1], polyder(self.simple_poly(self.points[k]))))
