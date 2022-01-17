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
from dolfin import conditional, inner, lt

from ufl.operators import exp, sinh


def phi_e_equation(phi_e, test, dx, c_e, j_Li, kappa, kappa_D, domain_grad=None, L=None, scale_factor=1):
    """
    Implements variational formulation equation for electrolyte potential phi_e

    Parameters
    ----------
    phi_e : Function
        Electrolyte Potential
    test : TestFunction
        Electrolyte Potential Equation Test Function
    dx : Measure
        Measure of the domain over the integral must integrate
    c_e : Function
        Electrolyte Concentration
    j_Li : Function
        Intercalation reaction Flux
    kappa : Constant or similar
        Effective electric conductivity of the electrolyte
    kappa_D : Constant or similar
        Effective concentration induced conductivity (More or less)
    grad : function
        python function that returns the UFL gradient of the argument
    L : Constant
        Thickness used to normalize the domain, by default None

    Returns
    -------
    Form
        Electrolyte Potential Equation
    """
    if dx.subdomain_data().where_equal(dx.subdomain_id()):
        F_phi = scale_factor*L*kappa * inner(domain_grad(phi_e), domain_grad(test))*dx(metadata={"quadrature_degree":0}) + \
            (scale_factor*L*kappa_D/c_e)*inner(domain_grad(c_e), domain_grad(test))*dx(metadata={"quadrature_degree":2})
        if j_Li is not None:
            F_phi -= scale_factor*L*j_Li*test*dx(metadata={"quadrature_degree":2})
        return F_phi
    else:
        return 0


def phi_s_equation(phi_s, test, dx, j_Li, sigma, domain_grad=None, L=None, scale_factor=1, lagrange_multiplier=None,  dS=None, phi_s_test=None, phi_s_cc_test=None):
    """
    Implements variational formulation for electrode potential phi_s

    Parameters
    ----------
    phi_s : Function
        Electrode Potential
    test : TestFunction
        Electrode Potential Equation Test Function
    dx : Measure
        Measure of the domain over the integral must integrate
    j_Li : Function
        Intercalation reaction Flux
    sigma : Constant or similar
        Effective electric conductivity of the electrode material
    grad : function
        python function that returns the UFL gradient of the argument
    i_app : Constant, Function or similar
        Current applied to the boundary of the electrode
    sign : int
        1 or -1 depending if the applied current is positive or negative
        (by convention, it is positive for the anode and negative for the cathode)
    ds : Measure
        Measure of the boundary domain over the integral must integrate
    L : Constant
        Thickness used to normalize the domain, by default None

    Returns
    -------
    Form
        Electrode Potential Equation
    """
    if dx.subdomain_data().where_equal(dx.subdomain_id()):
        F_phi_s = 0
        if sigma is not None:
            F_phi_s += scale_factor*L * sigma * inner(domain_grad(phi_s), domain_grad(test)) * dx
        # Adds source term if there is any
        if j_Li is not None:
            F_phi_s += scale_factor*L * j_Li*test*dx
        if lagrange_multiplier and dS and (phi_s_test or phi_s_cc_test):
            if phi_s_test:
                F_phi_s += scale_factor*phi_s_interface(lagrange_multiplier, dS, phi_s_test=phi_s_test)
            elif phi_s_cc_test:
                F_phi_s -= scale_factor*phi_s_interface(lagrange_multiplier, dS, phi_s_cc_test=phi_s_cc_test)
            else:
                raise Exception("Invalid interface condition")
        return F_phi_s
    else:
        return 0

def phi_s_bc(I_app, test, ds, scale_factor=1):
    return scale_factor*I_app * test * ds

def phi_s_interface(lagrange_multiplier, dS, phi_s_test = None, phi_s_cc_test=None):
    if phi_s_test:
        interface_bc = lagrange_multiplier(dS.metadata()['direction'])*phi_s_test(dS.metadata()['direction'])*dS
    elif phi_s_cc_test:
        interface_bc = lagrange_multiplier(dS.metadata()['direction'])*phi_s_cc_test(dS.metadata()['direction'])*dS
    return  interface_bc

def phi_s_continuity(phi_s_electrode, phi_s_cc, lm_test, dS_el, dS_cc):
    el_dir = dS_el.metadata()['direction']; cc_dir = dS_cc.metadata()['direction']
    return phi_s_electrode(el_dir) * lm_test(el_dir) * dS_el - phi_s_cc(cc_dir) * lm_test(cc_dir) * dS_cc

def c_e_equation(c_e_0, c_e, test, dx, DT, j_Li, D_e, eps_e, t_p, F, domain_grad=None, L=None, scale_factor=1):
    """
    Implements variational formulation for electrolyte concentration c_e

    Parameters
    ----------
    c_e_0 : Function
        Electrode Potential last timestep
    c_e : Function
        Electrode Potential
    test : TestFunction
        Electrode Potential Equation Test Function
    dx : Measure
        Measure of the domain over the integral must integrate
    DT : TimeScheme
        Instance of the TimeScheme class
    j_Li : Function
        Intercalation reaction Flux
    D_e : Constant or similar
        Effective Diffusivity of the electrolyte
    eps_e : Constant or similar
        Volume fraction occupied by the electrolyte in the domain
    t_p : Constant or similar
        Transference number for the reaction
    F : Constant or similar
        Faraday Constant
    grad : function
        python function that returns the UFL gradient of the argument
    L : Constant
        Thickness used to normalize the domain, by default None

    Returns
    -------
    Form
        Electrolyte Concentration Equation
    """
    if dx.subdomain_data().where_equal(dx.subdomain_id()):
        F_c = (scale_factor*L * eps_e) * DT.dt(c_e_0, c_e) * test * dx + \
            (scale_factor*L*D_e) * inner(domain_grad(c_e), domain_grad(test)) * dx
        if j_Li is not None:
            F_c -= (scale_factor * L * (1-t_p)/F)*j_Li * test * dx
        return F_c
    else:
        return 0


def i_n_equation(k, c_e, c_s, c_s_max, alpha):
    f_c_e, f_c_s, f_c_s_max = [1,1,1]
    regularization = exp(-f_c_s/c_s**(1/alpha)) * exp(-f_c_e/c_e**(1/(1-alpha))) * exp(-f_c_s_max/(c_s_max - c_s)**(1/(1-alpha)))
    i_0 = k * c_e**(1-alpha) * c_s**alpha * (c_s_max - c_s)**(1-alpha) 
    i_n = conditional(lt(c_e, 0), 0, conditional(lt(c_s, 0), 0, conditional(lt(c_s_max-c_s, 0), 0, i_0 * regularization)))
    return i_n

def ButtlerVolmer_equation(alpha, F, R, T, eta):
    return 2 * sinh((alpha*F/R) * eta / T )

def overpotential_equation(phi_s, phi_e, OCV, J=None, SEI=None, delta=None):
    if SEI:
        G = SEI.R + delta / SEI.kappa
        return phi_s - phi_e - OCV - J * G 
    else:
        return phi_s - phi_e - OCV

def j_Li_equation(material, c_e, c_s_surf, alpha, phi_s, phi_e, F, R, T, current, J=None, SEI=None, delta=None):
    i_n = i_n_equation(material.k_0,c_e,c_s_surf,material.c_s_max,alpha)
    ocv = material.U
    eta = overpotential_equation(phi_s, phi_e, ocv(c_s_surf/material.c_s_max, current), J, SEI, delta)
    BV = ButtlerVolmer_equation(alpha, F, R, T, eta)
    return F*i_n*BV
