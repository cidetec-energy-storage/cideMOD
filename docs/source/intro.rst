Overview
=========

.. _introduction:

Here the formulation of the P4D is described

Subdomains
-----------

There are different subdomains in the model:

- Current Collector
- Electrode
- Separator

The different variables in the models are defined only in some of the domains. Considering the models included the variables are defined in the following domains:

- Electrochemical model:
  Electrolyte potential and concentration :math:`\phi_e\, c_e` are defined in the electrodes and the separator. Solid phase potential :math:` \phi_s ` is defined only at the electrodes. At the current collectors there is also a solid phase potential but due to the disparity of conductivity scales, another variable is created :math:`\phi_{s,cc}`, and continuity is imposed at the interface between electrode and current collectors with a lagrange multiplier. Particle concentration :math:`c_s`\ and intercalation current density :math:`j_i` are also defined as variables but only at their corresponding electrode. This means that each electrode has many of these variables as active materials.
- Thermal model:
  In isothermal simulations (solve\_thermal=False) the temperature is set as a constant value over the domain. Otherwise, temperature is a scalar function defined in all of the domains. Heat sources considered are ohmic, polarization and reaction heat sources.
- Degradation model:
  Several degradation models have been implemented *(WIP)*:

  1. Solid-Electrolite Interphase
  2. Lithium Plating

- Mechanical model:
  *(WIP)*

Nondimensional model
---------------------

Scaling
~~~~~~~~

The model has been rescaled to obtain nondimensional quantities and homogeneize truncation errors. The rescaled variables are obtained with the following relations:

-  Spatial and temporal dimensions:

   .. math::

      \begin{gathered}
              x= (L_a+L_s+L_c) \hat{x} ;\qquad r = R_s \hat{r};\qquad t=t_c\hat{t}
          \end{gathered}

-  Potentials:

   .. math::

      \begin{gathered}
              \Phi_T = \frac{R T_0}{\alpha F} ; \qquad \Phi_s = \frac{I_0 L_0}{\sigma_{ref}} ; \qquad \Phi_l = \Phi_T
          \end{gathered}

   .. math::

      \begin{gathered}
              \phi_{e}=\phi_{e}^{ref}+\Phi_l\hat{\phi_{e}} ;\qquad \phi_{s}=\phi_{s}^{ref}+\Phi_s\hat{\phi_{s}} ; \qquad E_{eq} = \phi_{s}^{ref} - \phi_{e}^{ref} + \Phi_T\hat{E}_{eq}     
          \end{gathered}

   .. math::

      \begin{gathered}
              \eta=\Phi_T \hat{\eta} = \Phi_s \hat{\phi_s}-\Phi_l \hat{\phi_e} - \Phi_T\hat{E}_{eq}
          \end{gathered}

-  Lithium concentrations:

   .. math::

      \begin{gathered}
              c_{e}=c_{e}^{ref}+\Delta c_e^{ref} \hat{c}_{e} ;\qquad \Delta c_e^{ref}=\frac{I_0 L_0 (1-t_+^0)}{D_e^{eff}F} ;\qquad c_{s}= c_s^{max} \hat{c_{s}}
          \end{gathered}

-  Current densities:

   .. math::

      \begin{gathered}
              a_s j= \frac{I_0}{L_0} \hat{i_n} ;\qquad I_{app} = I_0 \hat{I}_{app} ; \qquad I_0 = \frac{Q}{A t_c}
          \end{gathered}

-  Temperature:

   .. math::

      \begin{gathered}
              T = T_0+\Delta T_{ref}\hat{T}  ; \qquad \Delta T_{ref} = \frac{I_0 t_c}{L_0 \rho^{ref} c_p^{ref} } \Phi_T
          \end{gathered}

Dimensionless numbers
~~~~~~~~~~~~~~~~~~~~~~

With the mentioned scaling and proper arrangement in the equations, we have defined the following dimensionless quantities:

+---------------------------------+------------------------------------------------------------+--------------------+---------------------------------------------------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
| :math:`\tau_e`                  | :math:`\frac{D_{eff}^{ref} t_c}{L_0}`                      | :math:`\delta_K`   | :math:`\frac{L_0 I_0}{K_{eff}^{ref} \Phi_l}`                  | :math:`\delta_{K_D}`       | :math:`\frac{\delta_K}{2\alpha (1-t_+^0)(1+\frac{\partial \ln f_{\pm}}{\partial \ln c_e})} \frac{\Phi_l}{\Phi_T} \frac{c_e^{ref}}{\Delta c_e^{ref}}` |
+---------------------------------+------------------------------------------------------------+--------------------+---------------------------------------------------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
| :math:`\delta_{\sigma}`         | :math:`\frac{I_0 L_0}{\sigma_{ref} \Phi_s}`                | :math:`\tau_s`     | :math:`\frac{D_s^{ref} t_c}{R_s^2}`                           | :math:`S`                  | :math:`\frac{R_s I_0}{a_s D_s^{ref} c_s^{max} F L_0}`                                                                                                |
+---------------------------------+------------------------------------------------------------+--------------------+---------------------------------------------------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
| :math:`\hat{k}_0`               | :math:`\frac{F k_0 L_0 }{I_0} c_e^{ref} (c_s^{max})^2`     | :math:`\tau_q`     | :math:`\frac{t_c k_T^{ref} }{\rho^{ref} c_p^{ref} L_0^2}`     | :math:`\delta_{\lambda}`   | :math:`\frac{L_0^2 \rho^{ref} c_p^{ref} }{t_c \lambda^{ref}}`                                                                                        |
+---------------------------------+------------------------------------------------------------+--------------------+---------------------------------------------------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+

Equations
~~~~~~~~~~

With the specified scaling and dimensionless numbers, the following equations are resolved for each of the variables:

-  Electrolyte concentration:

   .. math::

      \begin{gathered}
              \frac{\epsilon}{\tau_e}\frac{\partial\hat{c}_e}{\partial \hat{t}} =
              \hat{\nabla} \left(\frac{D_{eff}}{D_{eff}^{ref}} \hat{\nabla} \hat{c}_e \right) + \sum_{i=0}^{n_{mat}} \hat{j}_{i}
          \end{gathered}

-  Electrolyte potential:

   .. math::

      \begin{gathered}
                   - \hat{\nabla} \left( \frac{1}{\delta_K} \frac{K_{eff}}{K_{eff}^{ref}} \hat{\nabla}\hat{\phi}_e - \frac{1}{\delta_{K_D}} \frac{K_{eff}}{K_{eff}^{ref}} \frac{1+\frac{\Delta T}{T_{ref}} \hat{T}}{1+\frac{\Delta c_e}{c_{e,ref}} \hat{c}_e} \hat{\nabla} \hat{c}_e   \right) = \sum_{i=0}^{n_{mat}} \hat{j}_i
          \end{gathered}

-  Solid phase potential:

   .. math::

      \begin{gathered}
              -\hat{\nabla} \left( \frac{1}{\delta_{\sigma}} \frac{\sigma_{eff}}{\sigma_{eff}^{ref}} \hat{\nabla} \hat{\phi}_s \right) = -\sum_{i=0}^{n_{mat}} \hat{j}_i ;\qquad  \frac{1}{\delta_{\sigma}} \frac{\sigma_{eff}}{\sigma_{eff}^{ref}} \frac{\partial \hat{\phi}_s}{\partial \vec{n}} \Bigg|_{tab} = \hat{I}_{app} 
          \end{gathered}

-  Intercalation current density:

   .. math::

      \begin{gathered}
              \hat{j}_i = \hat{k}_0 \left( \left( 1+\frac{\Delta c_e}{c_{e,ref}} \hat{c}_e \right) \hat{c}_s|_{\hat{r}=1} (1-\hat{c}_s|_{\hat{r}=1}) \right)^{0.5} 2 \sinh{\hat{\eta}}
          \end{gathered}

-  Particle concentration:

   .. math::

      \begin{gathered}
             \frac{1}{\tau_s} \frac{\partial \hat{c}_s}{\partial \hat{t}} = \frac{1}{\hat{r}^2}\frac{\partial}{\partial \hat{r}} \left( \hat{r}^2 \frac{D_s}{D_{s}^{ref}} \frac{\partial \hat{c}_s}{\partial \hat{r}} \right) ; \qquad \frac{D_s}{D_{s}^{ref}} \frac{\partial \hat{c}_s}{\partial \hat{r}} \Bigg|_{\hat{r}=1} = S \hat{j}_i
          \end{gathered}

-  Temperature:

   .. math::

      \begin{gathered}
             \frac{\rho c_p}{\rho^{ref} c_p^{ref}} \frac{\partial \hat{T}}{\partial \hat{t}} = \frac{1}{\delta_{\lambda}}\hat{\nabla} \left( \frac{\lambda}{\lambda^{ref}} \hat{\nabla} \hat{T} \right) + \hat{q}  ; \qquad \frac{\lambda}{\lambda^{ref}} \frac{\partial \hat{T}}{\partial \vec{n}} \Bigg|_{\Gamma} = \frac{L_0 h}{\lambda^{ref} \Delta T_{ref}} \left(T_0-T_{ext} + \Delta T_{ref} \hat{T} \right)
          \end{gathered}

Weak Formulation
~~~~~~~~~~~~~~~~~

Electrolyte concentration
"""""""""""""""""""""""""""

.. math::

   \begin{gathered}
       \int_{\Omega}{\frac{\epsilon}{\tau_e}\frac{\partial\hat{c}_e}{\partial \hat{t}} \tilde{c_e}} +
       \int_{\Omega}{\frac{D_{eff}}{D_{eff}^{ref}} \hat{\nabla} \hat{c}_e \hat{\nabla} \tilde{c_e}} -
       \int_{\Gamma}{\frac{D_{eff}}{D_{eff}^{ref}} (\hat{\nabla} \hat{c}_e \cdot \vec{n}) \tilde{c_e}} -
       \int_{\Omega}{\sum_{i=0}^{n_{mat}} \hat{j}_{i} \tilde{c_e}} = 0
       \\
       \boxed{
       \int_{\Omega}{\frac{\epsilon}{\tau_e}\frac{\partial\hat{c}_e}{\partial \hat{t}} \tilde{c_e}} +
       \int_{\Omega}{\frac{D_{eff}}{D_{eff}^{ref}} \hat{\nabla} \hat{c}_e \hat{\nabla} \tilde{c_e}} -
       \int_{\Omega}{\sum_{i=0}^{n_{mat}} \hat{j}_{i} \tilde{c_e}} = 0
       }\end{gathered}

Electrolyte potential
"""""""""""""""""""""

.. math::

   \begin{gathered}
       \int_{\Omega}{\frac{1}{\delta_K} \frac{K_{eff}}{K_{eff}^{ref}} \hat{\nabla}\hat{\phi}_e \hat{\nabla} \tilde{\phi_e}} -
       \int_{\Omega}{\frac{1}{\delta_{K_D}} \frac{K_{eff}}{K_{eff}^{ref}} \frac{1+\frac{\Delta T}{T_{ref}} \hat{T}}{1+\frac{\Delta c_e}{c_{e,ref}} \hat{c}_e} \hat{\nabla} \hat{c}_e \hat{\nabla} \tilde{\phi_e}} -
       \int_{\Gamma}{\frac{1}{\delta_K} \frac{K_{eff}}{K_{eff}^{ref}} (\hat{\nabla}\hat{\phi}_e \cdot \vec{n}) \tilde{\phi_e}} + \\ %%Ojo con esto, es la misma ecuacion
       \int_{\Gamma}{\frac{1}{\delta_{K_D}} \frac{K_{eff}}{K_{eff}^{ref}} \frac{1+\frac{\Delta T}{T_{ref}} \hat{T}}{1+\frac{\Delta c_e}{c_{e,ref}} \hat{c}_e} (\hat{\nabla} \hat{c}_e \cdot \vec{n}) \tilde{\phi_e}} -
       \int_{\Omega}{\sum_{i=0}^{n_{mat}} \hat{j}_{i} \tilde{\phi_e}} = 0\end{gathered}

.. math::

   \begin{gathered}
       \boxed{
       \int_{\Omega}{\frac{1}{\delta_K} \frac{K_{eff}}{K_{eff}^{ref}} \hat{\nabla}\hat{\phi}_e \hat{\nabla} \tilde{\phi_e}} -
       \int_{\Omega}{\frac{1}{\delta_{K_D}} \frac{K_{eff}}{K_{eff}^{ref}} \frac{1+\frac{\Delta T}{T_{ref}} \hat{T}}{1+\frac{\Delta c_e}{c_{e,ref}} \hat{c}_e} \hat{\nabla} \hat{c}_e \hat{\nabla} \tilde{\phi_e}} -
       \int_{\Omega}{\sum_{i=0}^{n_{mat}} \hat{j}_{i} \tilde{\phi_e}} = 0
       }\end{gathered}

Solid phase potential
"""""""""""""""""""""

.. math::

   \begin{gathered}
       \int_{\Omega}{\frac{1}{\delta_{\sigma}} \frac{\sigma_{eff}}{\sigma_{eff}^{ref}} \hat{\nabla} \hat{\phi}_s \hat{\nabla} \tilde{\phi_s}} -
       \int_{\Gamma}{\frac{1}{\delta_{\sigma}} \frac{\sigma_{eff}}{\sigma_{eff}^{ref}} (\hat{\nabla} \hat{\phi}_s \cdot \vec{n}) \tilde{\phi_s}} +
       \int_{\Omega}{\sum_{i=0}^{n_{mat}} \hat{j}_{i} \tilde{\phi_s}} = 0
       \\
       \boxed{
       \int_{\Omega}{\frac{1}{\delta_{\sigma}} \frac{\sigma_{eff}}{\sigma_{eff}^{ref}} \hat{\nabla} \hat{\phi}_s \hat{\nabla} \tilde{\phi_s}} -
       \int_{\Gamma}{\hat{I}_{app} \tilde{\phi_s}} +
       \int_{\Omega}{\sum_{i=0}^{n_{mat}} \hat{j}_{i} \tilde{\phi_s}} = 0
       }\end{gathered}

Intercalation current density
"""""""""""""""""""""""""""""""

.. math::

   \begin{gathered}
       \boxed{
       \int_{\Omega}{\hat{j}_i \tilde{j_i}} -
       \int_{\Omega}{\hat{k}_0 \left( \left( 1+\frac{\Delta c_e}{c_{e,ref}} \hat{c}_e \right) \hat{c}_s|_{\hat{r}=1} (1-\hat{c}_s|_{\hat{r}=1}) \right)^{0.5} 2 \sinh{(\hat{\eta})} \tilde{j_i}} = 0
       }\end{gathered}


Particle Model
---------------

The particle domain is solved using a spectral method using even Legendre polynomials. Thus the concentration in the domain has been decomposed as:

.. math::

   \begin{gathered}
       \hat{c}_s (\hat{r}, \hat{x}, \hat{t}) = \sum_{i=0}^{n} L_{2i} (\hat{r}) \hat{c}_{s,i} (\hat{x}, \hat{t}) \end{gathered}

Taking advantage of the properties of Legendre polinomials, we know that :math:`L_i(1) = 1` then we can substitute the zeroth order coefficient with a new variable :math:`\hat{c}_{surf}` representing the concentration at the particles surface:

.. math::

   \begin{gathered}
       \hat{c}_s(1,\hat{x},\hat{t}) = \hat{c}_{surf} = \sum_{i=0}^{n} L_{2i} (1) \hat{c}_{s,i} (\hat{x}, \hat{t}) = \sum_{i=0}^{n} \hat{c}_{s,i} (\hat{x}, \hat{t}) \\
       \hat{c}_s (\hat{r}, \hat{x}, \hat{t}) = L_0 \left(\hat{c}_{surf} -  \sum_{i=1}^{n}\hat{c}_{s,i} (\hat{x}, \hat{t})\right)+\sum_{i=1}^{n} L_{2i} (\hat{r}) \hat{c}_{s,i} (\hat{x}, \hat{t})\end{gathered}

This change of variable reduces the jacobian entries, and the calculations needed to evaluate entries of the lithium intercalation flux.

