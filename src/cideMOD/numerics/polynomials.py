import numpy
from numpy.polynomial.polynomial import *

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
            poly = polyadd(poly, Polynomial(c[k]*self.f[k]))
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
