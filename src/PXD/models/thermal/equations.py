from dolfin import inner

from PXD.models.cell_components import CurrentColector, Electrode, Separator


def electrolyte_ohmic_heat_equation(kappa, kappa_D, phi_e, c_e, test, dx, grad, L, eps):
    F_q = 0
    if phi_e is not None and kappa is not None and kappa_D is not None and c_e is not None:
        F_q += L * eps * kappa * inner(grad(phi_e), grad(phi_e)) * test * dx(metadata={"quadrature_degree":1}) + \
            L * eps * (kappa_D/c_e) * inner(grad(c_e), grad(phi_e)) * test * dx(metadata={"quadrature_degree":2})
    return F_q

def solid_ohmic_heat_equation(sigma, phi_s, test, dx, grad, L, eps):
    F_q = 0
    if phi_s is not None and sigma is not None:
        F_q += L * (1-eps) * sigma * inner(grad(phi_s), grad(phi_s)) * test * dx(metadata={"quadrature_degree":1})
    return F_q

def reaction_irreversible_heat(material, j_Li, c_s_surf, phi_s, phi_e, test, current, dx, L):
    F_q = 0
    if j_Li is not None and c_s_surf is not None:
        ocv = material.U(c_s_surf/material.c_s_max, current)
        F_q = L * material.a_s * j_Li * (phi_s - phi_e - ocv) * test * dx(metadata={"quadrature_degree":3})   
    return F_q

def reaction_reversible_heat(material, j_Li, T, c_s, current, test, dx, L):
    F_q=0
    if j_Li is not None and material.delta_S is not None:
        delta_s = material.delta_S(c_s/material.c_s_max, current)
        F_q = L*material.a_s*j_Li*T*delta_s*test*dx(metadata={"quadrature_degree":3})
    return F_q

def q_equation(domain, f_1, c_s_surf, test, dx, current):
    q = 0
    if isinstance(domain, (Electrode, Separator)):
        q += electrolyte_ohmic_heat_equation(domain.kappa, domain.kappa_D, f_1.phi_e, f_1.c_e, test, dx, domain.grad, domain.L, domain.eps_e)
    if isinstance(domain, CurrentColector):
        q+= solid_ohmic_heat_equation(domain.sigma, f_1.phi_s_cc, test, dx, domain.grad, domain.L, 0)
    if isinstance(domain, Electrode):
        q+= solid_ohmic_heat_equation(domain.sigma, f_1.phi_s, test, dx, domain.grad, domain.L, domain.eps_e)
        for i, material in enumerate(domain.active_material):
            j_li_index = f_1._fields.index(f'j_Li_{domain.tag}{i}')
            q += reaction_irreversible_heat(material, f_1[j_li_index], c_s_surf[i], f_1.phi_s, f_1.phi_e, test, current, dx, domain.L )
            q += reaction_reversible_heat(material, f_1[j_li_index], f_1.temp, c_s_surf[i], current, test, dx, domain.L )
    return q
            

def T_equation(T_0, T, test, dx, DT, rho, c_p, k_t, q, grad, L, alpha=1):
    """
    Implements variational form of Temperature Equation

    Parameters
    ----------
    T_0 : Function
        Temperature field previous timestep
    T : Function
        Temperature
    test : TestFunction
        Temperature equation TestFunction
    dx : Measure
        Measure of the domain over the integral must integrate
    DT : TimeScheme
        Instance of the TimeScheme class
    rho : Constant or similar
        Density of the material
    c_p : Constant or similar
        Specific heat of material
    k_t : Constant or similar
        Heat Conductivity of material
    q : Sum or Expression or similar
        Heat generated
    grad : function
        python function that returns the UFL gradient of the argument
    L : Constant
        Thickness used to normalize the domain

    Returns
    -------
    Form
        Temperature Equation
    """
    # TODO: review thermal model and units to quit this x1000 factor
    F_t = alpha * L * rho * c_p * DT.dt(T_0, T) * test * dx(metadata={"quadrature_degree":2}) + \
        alpha * L * k_t * inner(grad(T), grad(test))*dx(metadata={"quadrature_degree":0})
    if q is not None:
        F_t -= alpha * q
    return F_t

