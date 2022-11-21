Nondimensional model
---------------------

.. rubric:: Scaling

The model has been rescaled to obtain nondimensional quantities and
homogeneize truncation errors, this scaling is inspired in
:cite:t:`Ayerbe2020`. The methods
:meth:`scale_variables <cideMOD.models.base.base_nondimensional.BaseModel.scale_variables>`
and :meth:`unscale_variables <cideMOD.models.base.base_nondimensional.BaseModel.unscale_variables>`
of the class :class:`NondimensionalModel <cideMOD.models.nondimensional_model.NondimensionalModel>`
contains the implementation of this variable rescaling.
The rescaled variables are obtained with the following relations:

* Spatial and temporal dimensions:

   .. math::

      \begin{gathered}
         \vec{x}= (L_a+L_s+L_c) \hat{\vec{x}} = L_0 \hat{\vec{x}}; \qquad  \nabla = \frac{1}{L_0} \hat{\nabla}   \\ 
         r = R_s \hat{r};\qquad r_{SEI} = \delta_{SEI} \hat{r}_{SEI} + R_s  ;\qquad t=t_c\hat{t}
      \end{gathered}

* Potentials:

   .. math::

      \begin{gathered}
         \Phi_T = \frac{R T_0}{\alpha F} ; \qquad \Phi_s = \frac{I_0 L_0}{\sigma_{ref}} ; \qquad \Phi_l = \Phi_T
      \end{gathered}

   .. math::

      \begin{gathered}
         \varphi_{e}=\varphi_{e}^{ref}+\Phi_l\hat{\varphi_{e}} ;\qquad \varphi_{s}=\varphi_{s}^{ref}+\Phi_s\hat{\varphi_{s}}     
      \end{gathered}

   .. math::

      \begin{gathered}
         U_{eq} = \varphi_{s}^{ref} - \varphi_{e}^{ref} + \Phi_T\hat{U}_{eq}
         ; \qquad
         \eta=\Phi_T \hat{\eta}
      \end{gathered}

* Lithium concentrations:

   .. math::

      \begin{gathered}
              c_{e}=c_{e}^{ref}+\Delta c_e^{ref} \hat{c}_{e} ;\qquad \Delta c_e^{ref}=\frac{I_0 L_0 (1-t_+^0)}{D_e^{eff}F} ;\qquad c_{s}= c_s^{max} \hat{c_{s}}
          \end{gathered}

* Current densities:

   .. math::

      \begin{gathered}
              a_s j= \frac{I_0}{L_0} \hat{i_n} ;\qquad I_{app} = I_0 \hat{I}_{app} ; \qquad I_0 = \frac{Q}{A t_c}
          \end{gathered}

* Temperature:

   .. math::

      \begin{gathered}
              T = T_0+\Delta T_{ref}\hat{T}  ; \qquad \Delta T_{ref} = \frac{I_0 t_c}{L_0 \rho^{ref} c_p^{ref} } \Phi_T
          \end{gathered}

* SEI lenght and solvent concentration:

   .. math::

      \begin{gathered}
         \delta_{SEI} = \Delta \delta \hat{\delta} \qquad c_{EC}=c_{EC}^{ref} \hat{c}_{EC} 
      \end{gathered}

* Stresses:
  
     .. math::

      \begin{gathered}
         \sigma_{\mathrm{h}} = E_{\mathrm{ref}} \hat{\sigma_{\mathrm{h}}}
      \end{gathered}

.. rubric:: Dimensionless numbers

With the mentioned scaling and proper arrangement in the equations, we
have defined the following dimensionless quantities, that are
implemented in the method
:meth:`calc_dimensionless_parameters <cideMOD.models.base.base_nondimensional.BaseModel.calc_dimensionless_parameters>`
of :class:`NondimensionalModel <cideMOD.models.nondimensional_model.NondimensionalModel>`:

+---------------------------------+------------------------------------------------------------+----------------------------+----------------------------------------------------------------------------------+-------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
| :math:`\tau_e`                  | :math:`\frac{D_{eff}^{ref} t_c}{L_0}`                      | :math:`\delta_K`           | :math:`\frac{L_0 I_0}{K_{eff}^{ref} \Phi_l}`                                     | :math:`\delta_{K_D}`                | :math:`\frac{\delta_K}{2\alpha (1-t_+^0)(1+\frac{\partial \ln f_{\pm}}{\partial \ln c_e})} \frac{\Phi_l}{\Phi_T} \frac{c_e^{ref}}{\Delta c_e^{ref}}` |
+---------------------------------+------------------------------------------------------------+----------------------------+----------------------------------------------------------------------------------+-------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
| :math:`\delta_{\sigma}`         | :math:`\frac{I_0 L_0}{\sigma_{ref} \Phi_s}`                | :math:`\tau_s`             | :math:`\frac{D_s^{ref} t_c}{R_s^2}`                                              | :math:`S`                           | :math:`\frac{R_s I_0}{a_s D_s^{ref} c_s^{max} F L_0}`                                                                                                |
+---------------------------------+------------------------------------------------------------+----------------------------+----------------------------------------------------------------------------------+-------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
| :math:`\hat{k}_0`               | :math:`\frac{F k_0 L_0 }{I_0} c_e^{ref} (c_s^{max})^2`     | :math:`\tau_q`             | :math:`\frac{t_c k_T^{ref} }{\rho^{ref} c_p^{ref} L_0^2}`                        | :math:`\delta_{\lambda}`            | :math:`\frac{L_0^2 \rho^{ref} c_p^{ref} }{t_c \lambda^{ref}}`                                                                                        |
+---------------------------------+------------------------------------------------------------+----------------------------+----------------------------------------------------------------------------------+-------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
| :math:`\Delta \delta`           | :math:`\frac{t_c I_0 M_{SEI}}{2 F \rho a_s L_0}`           | :math:`\tau_{\mathrm{LAM}}`| :math:`\beta t_c \left ( \frac{E_{\mathrm{ref}}}{\sigma_{\mathrm{cr}}}\right )^m`| :math:`\delta_{\sigma_{\mathrm{h}}}`| :math:`\frac{2\Omega}{9\left(1-\nu\right)}c_{\mathrm{s}}^{\mathrm{max}}`                                                                             |
+---------------------------------+------------------------------------------------------------+----------------------------+----------------------------------------------------------------------------------+-------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+


With the specified scaling and dimensionless numbers, the models
equation are reformulated.

.. rubric:: Electrochemical Model

* Mass transport in the electrolyte:
   The mass transport in the electrolyte is calculated in
   :meth:`c_e_equation <cideMOD.models.electrochemical.nondimensional.ElectrochemicalModel.c_e_equation>` 
   method.

   .. math::

      \begin{gathered}
         \frac{\epsilon}{\tau_e}\frac{\partial\hat{c}_e}{\partial \hat{t}} =
         \hat{\nabla} \left(\frac{D_{eff}}{D_{eff}^{ref}} \hat{\nabla} \hat{c}_e \right) + \sum_{i=0}^{n_{mat}} \hat{j}_{i}
      \end{gathered}

* Charge transport in the electrolyte:
   The charge transport in the electrolyte is calculated in
   :meth:`phi_e_equation <cideMOD.models.electrochemical.nondimensional.ElectrochemicalModel.phi_e_equation>`
   method.

   .. math::

      \begin{gathered}
         - \hat{\nabla} \left( \frac{1}{\delta_K} \frac{K_{eff}}{K_{eff}^{ref}} \hat{\nabla}\hat{\varphi}_e - \frac{1}{\delta_{K_D}} \frac{K_{eff}}{K_{eff}^{ref}} \frac{1+\frac{\Delta T}{T_{ref}} \hat{T}}{1+\frac{\Delta c_e}{c_{e,ref}} \hat{c}_e} \hat{\nabla} \hat{c}_e   \right) = \sum_{i=0}^{n_{mat}} \hat{j}_i
      \end{gathered}

* Charge transport in the electrodes and current collectors:
   The charge transport in the solid electron conductor materials is calculated in
   :meth:`phi_s_equation <cideMOD.models.electrochemical.nondimensional.ElectrochemicalModel.phi_s_equation>`
   method.

   .. math::

      \begin{gathered}
         -\hat{\nabla} \left( \frac{1}{\delta_{\sigma}} \frac{\sigma_{eff}}{\sigma_{eff}^{ref}} \hat{\nabla} \hat{\varphi}_s \right) = -\sum_{i=0}^{n_{mat}} \hat{j}_i 
         ;\quad  
         \frac{1}{\delta_{\sigma}} \frac{\sigma_{eff}}{\sigma_{eff}^{ref}} \frac{\partial \hat{\varphi}_s}{\partial \vec{n}} \Bigg|_{tab} = \hat{I}_{app} 
      \end{gathered}

* Mass transport in the active material (pseudodimension):
   The mass transport in the active material is calculated in the
   :class:`SpectralLegendreModel <cideMOD.models.particle_models.implicit_coupling.NondimensionalSpectralModel>`
   class using Legendre polynomials.

   .. math::

      \begin{gathered}
         \frac{1}{\tau_s} \frac{\partial \hat{c}_s}{\partial \hat{t}} = \frac{1}{\hat{r}^2}\frac{\partial}{\partial \hat{r}} \left( \hat{r}^2 \frac{D_s}{D_{s}^{ref}} \frac{\partial \hat{c}_s}{\partial \hat{r}} \right) 
         ; \quad 
         \frac{D_s}{D_{s}^{ref}} \frac{\partial \hat{c}_s}{\partial \hat{r}} \Bigg|_{\hat{r}=1} = S \hat{j}_i
      \end{gathered}

* Exchange between the electrolyte and the electrode by lithium intercalation:
   The intercalation exchange current between the electrolyte and the
   active materials is calculated in
   :meth:`j_int <cideMOD.models.electrochemical.nondimensional.ElectrochemicalModel.j_Li_equation>`
   method.

   .. math::

      \begin{gathered}
         \hat{j}_i = \hat{k}_0 \left( \left( 1+\frac{\Delta c_e}{c_{e,ref}} \hat{c}_e \right) \hat{c}_s|_{\hat{r}=1} (1-\hat{c}_s|_{\hat{r}=1}) \right)^{0.5} 2 \sinh{\hat{\eta}}
      \end{gathered}

* Overpotential
   The overpotential at each part of the electrode is calculated in
   :meth:`overpotential <cideMOD.models.electrochemical.nondimensional.ElectrochemicalModel.overpotential>`
   method.

   .. math::

      \begin{gathered}
         \hat{\eta} = \frac{\Phi_s}{\Phi_T} \hat{\varphi_s} - \frac{\Phi_l}{\Phi_T} \hat{\varphi_e} - \hat{U_{eq}}
      \end{gathered}

.. rubric:: Thermal Model

* Energy conservation:
   The heat transfer across the cell is computed in
   :meth:`T_equation <cideMOD.models.thermal.nondimensional.ThermalModel.T_equation>`
   function.
   
   .. math::

      \begin{gathered}
         \frac{\rho c_p}{\rho^{ref} c_p^{ref}} \frac{\partial \hat{T}}{\partial \hat{t}} = \frac{1}{\delta_{\lambda}}\hat{\nabla} \left( \frac{\lambda}{\lambda^{ref}} \hat{\nabla} \hat{T} \right) + \hat{q}  
         \\
         \frac{\lambda}{\lambda^{ref}} \frac{\partial \hat{T}}{\partial \vec{n}} \Bigg|_{\Gamma} = \frac{L_0 h}{\lambda^{ref} \Delta T_{ref}} \left(T_0-T_{ext} + \Delta T_{ref} \hat{T} \right)
      \end{gathered}


* Heat generation:
   Several heat sources have been considered. They are added in the
   :meth:`T_equation <cideMOD.models.thermal.nondimensional.ThermalModel.T_equation>`
   directly.

   .. math::

      \begin{gathered}
         \hat{q} = \hat{q}_{ohm} + \hat{q}_{rev} + \hat{q}_{irr}
      \end{gathered}

   * Ohmic heat source
      This corresponds to the heat generated by the transport of charge within the cell.
   
      .. math::

         \begin{gathered}
            \hat{q}_{ohm} =  (1-\varepsilon) \hat{q}_{solid} + \varepsilon \hat{q}_{liquid} \\
            \hat{q}_{solid} =  \frac{1}{\delta_{\sigma}} \frac{\sigma_{eff}}{\sigma_{eff}^{ref}} \frac{\Phi_s}{\Phi_T} \hat{\nabla} \hat{\varphi}_s \hat{\nabla} \hat{\varphi}_s \\
            \hat{q}_{liquid} = \frac{\Phi_l}{\Phi_T} \frac{\kappa_{eff}}{\kappa_{eff}^{ref}} \left(\frac{1}{\delta_{\kappa}}  \hat{\nabla} \hat{\varphi}_e \hat{\nabla} \hat{\varphi}_e - \frac{1}{\delta_{\kappa_D}} \frac{1+\frac{\Delta T}{T_{ref}} \hat{T}}{1+\frac{\Delta c_e}{c_{e,ref}} \hat{c}_e} \hat{\nabla} \hat{c}_e \hat{\nabla} \hat{\varphi}_e \right)
         \end{gathered}
   
   * Reversible reaction heat source
      The reversible heat caused by the reaction is proportional to the
      entropy change, that is approximated with the variation of Open
      Circuit potential.

      .. math::

         \begin{gathered}
            \hat{q}_{rev} =  \sum_{i=0}^{n_{mat}} \hat{j}_{i} \frac{T}{\Phi_T} \frac{\partial U_i(c_s)}{\partial T}
         \end{gathered}

   * Irreversible polarization heat source
      This represents the irreversible heating due to the polarization
      heat generated by the exchange current at the
      electrolyte-electrode interface.

      .. math::

         \begin{gathered}
            \hat{q}_{irr} =  \sum_{i=0}^{n_{mat}} \hat{j}_{i} \hat{\eta}
         \end{gathered}

.. rubric:: Degradation Models

* SEI formation side reaction
   This model is implemented inside the
   :class:`SolventLimitedSEIModel <cideMOD.models.degradation.nondimensional.SolventLimitedSEIModel>`
   class. The model considers that the SEI is originated by the
   electrochemical reaction between EC solvent molecule, 2 lithium ions
   and 2 electrons at the electrode surface:

   .. math::

      \begin{gathered}
         EC + 2 Li^+ + 2 e^- \rightarrow V_{SEI}
      \end{gathered}
   
   Therefore the rection equation reads:

   .. math::

      \begin{gathered}
         \hat{j}_{SEI} = \frac{F L_0 k_{SEI}}{I_0} c_{EC}^{ref} c_s^{max} \hat{c}_{EC} \hat{c}_s e^{-\frac{\beta}{\alpha}(\hat{\eta} - (\hat{U}_{SEI} - \hat{U}_{eq}))}
      \end{gathered}

   where the concentration of EC solvent at the SEI must be modelled
   according to the transport equation:

   .. math::

      \begin{gathered}
         \frac{\partial \hat{c}_{EC}}{\partial \hat{t}} - \frac{\hat{x}}{\hat{\delta}_{SEI}} \frac{\partial \hat{\delta}_{SEI}}{\partial \hat{t}} \hat{\nabla} \hat{c}_{EC} 
         = \hat{\nabla} \left( \frac{t_c D_{EC} }{\Delta \delta^2} \frac{\hat{\nabla} \hat{c}_{EC}}{\hat{\delta}_{SEI}^2} - \frac{ \partial \hat{\delta}_{SEI}}{\partial \hat{t}} \hat{c}_{EC} \right)
      \end{gathered}

   with the following boundary conditions:

   .. math::

      \begin{gathered}
         \left( \frac{t_c D_{EC} }{\Delta \delta^2} \frac{\hat{\nabla} \hat{c}_{EC}}{\hat{\delta}_{SEI}^2} - \frac{ \partial \hat{\delta}_{SEI}}{\partial \hat{t}} \hat{c}_{EC} \right) \Bigg|_{\hat{x}=0} 
         = \frac{2 \rho_{SEI}}{M_{SEI} c_{EC}^{ref}} \hat{j}_{SEI}
         \quad ; \quad
         \hat{c}_{EC} \big|_{\hat{x}=1} = 1
      \end{gathered}

   The SEI growth can be calculated from the reaction rate and SEI
   components properties:

   .. math::

      \begin{gathered}
         \frac{\partial \hat{\delta}_{SEI}}{\partial \hat{t}} = - \hat{j}_{SEI}
      \end{gathered}

   The total exchange current therefore has two components:

   .. math::

      \begin{gathered}
         \hat{j}_{tot} = \hat{j}_{int} + \hat{j}_{SEI}
      \end{gathered}

   And the overpotential has now an additional component corresponding
   to the voltage drop caused by SEI resistance:

   .. math::

      \begin{gathered}
         \hat{\eta} = \frac{\Phi_s}{\Phi_T} \hat{\varphi_s} - \frac{\Phi_l}{\Phi_T} \hat{\varphi_e} - \hat{U_{eq}} - \frac{\Delta \delta I_0}{\kappa_{SEI} L_0 a_s \Phi_T} \hat{\delta}_{SEI} \hat{j}_{tot} 
      \end{gathered}

* LAM model
    This model is implemented inside the
    :class:`SEI <cideMOD.models.degradation.nondimensional.LAM_model>` class.
    The model computes the lost of active material due to particle
    cracking driven by stresses. Therefore, the decrease of the volume
    fraction of active material is computed as

    .. math::

        \begin{gathered}
            \hat\sigma_{\mathrm{h}}=\delta_{\sigma_\mathrm{h}}\left (
            3\int_{0}^{1}\hat{c}\hat{r}^2d\hat{r}-\hat{c} \right )
        \end{gathered}

    And the hydrostatic stress is computed from the equilibrium of
    stresses of a spherical electrode particle

    .. math::

        \begin{gathered}
            \frac{\partial \varepsilon_\mathrm{s}}{\partial \hat{t}}=
            -\tau_{\mathrm{LAM}}\left(\hat\sigma_{\mathrm{h}}\right)^m
            \qquad \hat\sigma_{\mathrm{h}}>0
        \end{gathered}
