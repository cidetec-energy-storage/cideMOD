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
"""cell_components creates and initializes the corresponding battery components
attributes. This also includes the functions for the weak formulation.

_Date: 16/09/2020_

_Author: Clara Ganuza_

"""
from dolfin import *

import numpy
from ufl.core.operator import Operator

from cideMOD.helpers.miscellaneous import *


def get_arr(Ea, T_ref, problem):
        '''
        Calculate Arrhenius equation.
        '''
        if Ea != 0:
            if 'nd_model' in problem.__dict__.keys():
                return exp((Ea / problem.R)*(1/T_ref - 1/problem.dim_variables.temp))
            else:    
                return exp((Ea / problem.R)*(1/T_ref - 1/problem.f_1.temp))
        else:
            return 1


class Component:
    """
    Component class has the common properties for the anode, cathode and separator
    as part of a battery.
    """
    def __init__(self, tag, config):
        self.tag = tag
        self.config = config
        self.L = self.get_value(config.thickness)
        self.H = self.get_value(config.height)
        self.W = self.get_value(config.width)

    def setup(self, problem):
        # Build length coeffs
        self.norm = []
        self.norm.append(self.L)
        if problem.mesher.mesh.geometric_dimension() > 1:
            self.norm.append(self.H)
        if problem.mesher.mesh.geometric_dimension() > 2:
            self.norm.append(self.W)
    
    def dx(self, measures):
        return measures[measures._fields.index('x_{}'.format(self.tag))]

    def ds(self, measures):
        return measures[measures._fields.index('s_{}'.format(self.tag))]

    def dS(self, measures, domain):
        return measures[measures._fields.index('S_{tag}_{dtag}'.format(tag = self.tag, dtag=domain.tag))]

    def grad(self, arg):
        "Return normalized gradient for normalized domains"
        return as_vector([arg.dx(i)/self.norm[i] for i in range(arg.geometric_dimension())])

    def get_value(self, variable, problem = None, default = None):
        if variable is None:
            return default
        elif isinstance(variable,dict):
            return variable
        elif isinstance(variable,str) and problem is not None:
            if 'nd_model' in problem.__dict__.keys():
                return constant_expression(variable, **{**problem.dim_variables._asdict(), 'T_0':problem.T_ini, 't_p':problem.t_p} )
            else:
                return constant_expression(variable, **{**problem.f_1._asdict(), 'T_0':problem.T_ini, 't_p':problem.t_p} )
        else:
            return variable


class PorousComponent(Component):
    def __init__(self, tag, config):
        super().__init__(tag, config)
        self.eps_e = self.get_value(self.config.porosity)
        self.bruggeman = self.get_value(self.config.bruggeman)
        self.tortuosity_e = self.get_value(self.config.tortuosity)

    def get_brug_e(self, dic, problem):
        """
        Calculate Bruggeman constant for the liquid phase in the component.
        """
        if not isinstance(dic, dict):
            dic = {'value': dic, 'effective': True}
        # get value or expression
        if isinstance(dic["value"],str):
            if 'nd_model' in problem.__dict__.keys():
                x = constant_expression(dic["value"], **{**problem.dim_variables._asdict(), 'T_0':problem.T_ini, 't_p':problem.t_p} )
            else:
                x = constant_expression(dic["value"], **{**problem.f_1._asdict(), 'T_0':problem.T_ini, 't_p':problem.t_p} )
        else:
            x = dic["value"]
        # do corrections if necessary
        if dic["effective"]:
            return x
        else:
            if (dic["correction"] == "bruggeman" or self.tortuosity_e is None) and self.bruggeman is not None:
                tortuosity_e = self.eps_e ** (1 - self.bruggeman)
                if isinstance(x, Expression) or isinstance(x,Operator):
                    return x * Constant(self.eps_e / tortuosity_e)
                return Constant(x * self.eps_e / tortuosity_e)
            elif (dic["correction"] == "tortuosity" or self.bruggeman is None) and self.tortuosity_e is not None:
                if isinstance(x, Expression) or isinstance(x,Operator):
                    return x * Constant(self.eps_e / self.tortuosity_e)
                return Constant(x * self.eps_e / self.tortuosity_e)
            else:
                raise Exception("Cant convert to effective value")

    def get_brug_s(self, dic, problem):
        '''
        Calculate Bruggeman constant for the solid phase in the component.
        '''
        if not isinstance(dic, dict):
            dic = {'value': dic, 'effective': True}
        if isinstance(dic["value"],str):
            if 'nd_model' in problem.__dict__.keys():
                x = constant_expression(dic["value"], **{**problem.dim_variables._asdict(), 'T_0':problem.T_ini, 't_p':problem.t_p} )
            else:
                x = constant_expression(dic["value"], **{**problem.f_1._asdict(), 'T_0':problem.T_ini, 't_p':problem.t_p} )
        else:
            x = dic["value"]
        if dic["effective"]:
            return x
        else:
            eps_s = sum([self.active_material[i].eps_s for i in range(len(self.active_material))])
            tortuosity_s = self.tortuosity_s
            if (dic["correction"] == "bruggeman" or tortuosity_s is None) and self.bruggeman is not None:
                tortuosity_s = eps_s ** (1 - self.bruggeman)
                if isinstance(x, Expression) or isinstance(x,Operator):
                    return x * Constant(eps_s / tortuosity_s)
                return Constant(x * eps_s / tortuosity_s)
            elif (dic["correction"] == "tortuosity" or self.bruggeman is None) and self.tortuosity_s is not None:
                tortuosity_s = self.tortuosity_s
                if isinstance(x, Expression) or isinstance(x,Operator):
                    return x * Constant(eps_s / tortuosity_s)
                return Constant(x * eps_s / tortuosity_s)
            else:
                raise Exception("Cant convert to effective value")

    def build_brg(self, problem):
        """
        Calls the calculate Bruggeman and Arrhenius constants functions to storage
        them.
        """

        try:
            self.sigma = self.get_brug_s(self.sigma, problem)
        except:
            pass
        if self.eps_e is None:
            self.D_e = None
            self.kappa = None
            self.kappa_D = None
        else:
            self.D_e = self.get_brug_e(problem.D_e, problem) * get_arr(problem.D_e_Ea, problem.D_e_Tref, problem)
            self.kappa = self.get_brug_e(problem.kappa, problem) * get_arr(problem.kappa_Ea, problem.kappa_Tref, problem)
            self.kappa_D = (- 2 * problem.R / problem.F) * (1 - problem.t_p) * \
                        problem.f_1.temp * self.kappa * problem.activity

        self.k_t = self.get_brug_s(self.k_t, problem)
        self.c_p = self.get_brug_s(self.c_p, problem)


class electrolyteInterface:
    def __init__(self, EI):
        self.R = EI.resitance
        self.U = EI.referenceVoltage
        self.i_0 = EI.sideReactionExchangeCurrentDensity
        self.M = EI.molecularWeight
        self.rho = EI.density
        self.k = EI.conductivity
        self.delta0 = EI.delta0
        self.beta = EI.chargeTransferCoefficient
        self.D_EC = EI.EC_diffusion
        self.eps = EI.EC_eps
        self.c_EC_sln = EI.solventSurfConcentration
        self.k_f_s = EI.rateConstant

    def __bool__(self) -> bool :
        return bool(self.M) and bool(self.eps) and bool(self.rho) and bool(self.k_f_s)


class Electrode(PorousComponent):
    def __init__(self, tag, electrode):
        super().__init__(tag, electrode)
        self.active_material = [self.ActiveMaterial(am, self) for am in electrode.active_materials]
        self.area = self.get_value(electrode.area)
        self.C_dl = self.get_value(electrode.doubleLayerCapacitance)

        self.SEI = electrolyteInterface(electrode.SEI)

    class ActiveMaterial:
        def __init__(self, material, electrode):
            self.config = material
            self.electrode = electrode
            self.inclusions = [self.__class__(inc) for inc in material.inclusions]

        def setup(self, bruggeman, problem, SOC_ini=1):
            self.index = self.config.index
            self.R_s = self.config.particleRadius
            self.eps_s = self.config.volumeFraction
            self.porosity = self.config.porosity
            self.k_0 = Constant(self.config.kineticConstant, name='k_0')
            self.k_0_Ea = self.config.kineticConstant_Ea
            self.k_0_Tref = self.config.kineticConstant_Tref
            self.c_s_max = self.config.maximumConcentration
            self.stoichiometry0 = self.config.stoichiometry0
            self.stoichiometry1 = self.config.stoichiometry1
            self.c_s_ini = self.c_s_max * (SOC_ini * (self.stoichiometry1 - self.stoichiometry0) + self.stoichiometry0)
            self.D_s_Ea = self.config.diffusionConstant_Ea
            self.D_s_Tref = self.config.diffusionConstant_Tref
            self.omega = self.config.omega
            self.young = self.config.young
            self.poisson = self.config.poisson
            if self.young and self.poisson is not None and problem.mechanics is not None:
                self.C_mat = problem.mechanics.elasticity_tensor(self.young, self.poisson)
                self.elsheby_tensor = problem.mechanics.elsheby_tensor(self.poisson)
            else:
                self.C_mat = None
                self.elsheby_tensor = None
            
            if type(self.config.diffusionConstant) == int or type(self.config.diffusionConstant) == float:
                self.D_s = Constant(self.config.diffusionConstant, name='D_s')
            elif isinstance(self.config.diffusionConstant, dict):
                self.D_s = get_spline(self.config.diffusionConstant['value'])
            else:
                self.D_s = self.config.diffusionConstant

            if None in [self.eps_s, self.R_s]:
                self.a_s = None
                self.tortuosity_s = None
            else:
                self.a_s = 3. * self.eps_s / self.R_s
                self.tortuosity_s = self.eps_s ** (1 - (bruggeman or 1.5))
            if self.config.openCircuitPotential is None:
                self.U = None
                self._U_check = None
            elif self.config.openCircuitPotential['type'] == 'spline':
                ocp = self.config.openCircuitPotential
                if isinstance(ocp['value'], dict):
                    self.U_hysteresis = True
                    self.U = hysteresys_property({
                        'charge': get_spline(numpy.loadtxt(ocp['value']['charge']),spline_type=ocp['spline_type']),
                        'discharge': get_spline(numpy.loadtxt(ocp['value']['discharge']),spline_type=ocp['spline_type']),
                    })
                    self._U_check = hysteresys_property({
                        'charge': get_spline(numpy.loadtxt(ocp['value']['charge']),spline_type=ocp['spline_type'], return_fenics=False),
                        'discharge': get_spline(numpy.loadtxt(ocp['value']['discharge']),spline_type=ocp['spline_type'], return_fenics=False),
                    })
                else:
                    
                    self.U_hysteresis = False
                    self.U = hysteresys_property({
                        'charge': get_spline(numpy.loadtxt(ocp['value']),spline_type=ocp['spline_type']),
                        'discharge': get_spline(numpy.loadtxt(ocp['value']),spline_type=ocp['spline_type']),
                    })
                    self._U_check = hysteresys_property({
                        'charge': get_spline(numpy.loadtxt(ocp['value']),spline_type=ocp['spline_type'], return_fenics=False),
                        'discharge': get_spline(numpy.loadtxt(ocp['value']),spline_type=ocp['spline_type'], return_fenics=False),
                    })
            else:
                raise NameError('Unknown type of OCP for active_material {}'.format(self.config.name))

            if self.config.entropyCoefficient is None:
                def dummy_f(*args,**kwargs):
                    return 0
                self.delta_S = dummy_f
            elif self.config.entropyCoefficient['type'] == 'spline':
                ocp = self.config.entropyCoefficient
                if isinstance(ocp['value'], dict):
                    self.delta_S_hysteresis = True
                    self.delta_S = hysteresys_property({
                        'charge': get_spline(numpy.loadtxt(ocp['value']['charge']),spline_type=ocp['spline_type']),
                        'discharge': get_spline(numpy.loadtxt(ocp['value']['discharge']),spline_type=ocp['spline_type']),
                    })
                else:
                    self.delta_S_hysteresis = False
                    self.delta_S = hysteresys_property({
                        'charge': get_spline(numpy.loadtxt(ocp['value']),spline_type=ocp['spline_type']),
                        'discharge': get_spline(numpy.loadtxt(ocp['value']),spline_type=ocp['spline_type']),
                    })
            else:
                raise NameError('Unknown type of OCP for active_material {}'.format(self.config.name))

            for inc in self.inclusions:
                inc.setup(bruggeman, SOC_ini)
                
            try:
                self.k_0 = self.k_0 * get_arr(self.k_0_Ea, self.k_0_Tref, problem)
            except:
                pass
    

    def setup (self, problem, SOC_ini=1):
        super().setup(problem)
        if 'nd_model' in problem.__dict__.keys():
            self.sigma = self.get_value(constant_expression(self.config.electronicConductivity, **problem.dim_variables._asdict()))
        else:
            self.sigma = self.get_value(constant_expression(self.config.electronicConductivity, **problem.f_1._asdict()))
        self.k_t = self.get_value(self.config.thermalConductivity)
        self.rho = self.get_value(self.config.density)
        self.c_p = self.get_value(self.config.specificHeat)
        self.tortuosity_s = self.get_value(self.config.tortuosity_s)
        
        # Build active material
        for material in self.active_material:
            material.setup(self.bruggeman, problem, SOC_ini)

        self.build_brg(problem)


class Separator(PorousComponent):
    def __init__(self, separator):
        super().__init__('s', separator)

    def setup(self, problem):
        super().setup(problem)
        #Separator
        self.k_t = self.get_value(self.config.thermalConductivity)
        self.rho = self.get_value(self.config.density)
        self.c_p = self.get_value(self.config.specificHeat)

        self.build_brg(problem)


class CurrentColector(Component):
    def __init__(self, tag, current_colector):
        assert tag in ('positive', 'negative'), "Invalid Colector Tag"
        super().__init__(tag[0]+'cc', current_colector)

    def setup(self, problem):
        super().setup(problem)
        # Current colector
        self.sigma = self.get_value(self.config.electronicConductivity)
        self.k_t = self.get_value(self.config.thermalConductivity)
        self.rho = self.get_value(self.config.density)
        self.c_p = self.get_value(self.config.specificHeat)
        
        self.young = self.get_value(self.config.young)
        self.poisson = self.get_value(self.config.poisson)
