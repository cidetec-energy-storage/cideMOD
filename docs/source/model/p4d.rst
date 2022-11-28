Nondimensional model
---------------------

.. rubric:: Scaling

The model has been rescaled to obtain nondimensional quantities and
homogeneize truncation errors, this scaling is inspired in
:cite:t:`Ayerbe2020`. The methods
:meth:`scale_variables <cideMOD.models.base.base_nondimensional.BaseModel.scale_variables>`
and :meth:`unscale_variables <cideMOD.models.base.base_nondimensional.BaseModel.unscale_variables>`
of the class :class:`NondimensionalModel <cideMOD.models.nondimensional_model.NondimensionalModel>`
contain the implementation of this variable rescaling.
The rescaled variables are obtained with the following relations:

* Spatial and temporal dimensions:

   .. math::

      \begin{gathered}
         x = (L_\mathrm{a}+L_\mathrm{s}+L_\mathrm{c}) \hat{x} = L_0 \hat{x}; \qquad  \nabla = \frac{1}{L_0} \hat{\nabla}   \\ 
         r = R_\mathrm{s} \hat{r};\qquad r_\mathrm{\scriptscriptstyle SEI} = \delta_\mathrm{\scriptscriptstyle SEI} \hat{r}_\mathrm{\scriptscriptstyle SEI} + R_\mathrm{s}  ;\qquad t=t_\mathrm{c}\hat{t}
      \end{gathered}

* Potentials:

   .. math::

      \begin{gathered}
         \Phi_T = \frac{R T_0}{\alpha F} ; \qquad \Phi_\mathrm{s} = \frac{I_0 L_0}{\sigma_\mathrm{ref}} ; \qquad \Phi_\mathrm{l} = \Phi_T
      \end{gathered}

   .. math::

      \begin{gathered}
         \varphi_\mathrm{e}=\varphi_\mathrm{e}^\mathrm{ref}+\Phi_\mathrm{l}\hat{\varphi}_\mathrm{e} ;\qquad \varphi_\mathrm{s}=\varphi_\mathrm{s}^\mathrm{ref}+\Phi_\mathrm{s}\hat{\varphi}_\mathrm{s}     
      \end{gathered}

   .. math::

      \begin{gathered}
         U_\mathrm{eq} = \varphi_\mathrm{s}^\mathrm{ref} - \varphi_\mathrm{e}^\mathrm{ref} + \Phi_T\hat{U}_\mathrm{eq}
         ; \qquad
         \eta=\Phi_T \hat{\eta}
      \end{gathered}

* Lithium concentrations:

   .. math::

      \begin{gathered}
              c_\mathrm{e}=c_\mathrm{e}^\mathrm{ref}+\Delta c_\mathrm{e}^\mathrm{ref} \hat{c}_\mathrm{e} ;\qquad \Delta c_\mathrm{e}^\mathrm{ref}=\frac{I_0 L_0 (1-t_+^0)}{D_\mathrm{e}^\mathrm{eff,ref}F} ;\qquad c_\mathrm{s}= c_\mathrm{s}^\mathrm{max} \hat{c_\mathrm{s}}
          \end{gathered}

* Current densities:

   .. math::

      \begin{gathered}
              a_\mathrm{s} j= \frac{I_0}{L_0} \hat{i_n} ;\qquad I_\mathrm{app} = I_0 \hat{I}_\mathrm{app} ; \qquad I_0 = \frac{Q}{A t_\mathrm{c}}
          \end{gathered}

* Temperature:

   .. math::

      \begin{gathered}
              T = T_0+\Delta T_\mathrm{ref}\hat{T}  ; \qquad \Delta T_\mathrm{ref} = \frac{I_0 t_\mathrm{c}}{L_0 \rho^\mathrm{ref} c_p^\mathrm{ref} } \Phi_T
          \end{gathered}

* SEI thickness and solvent concentration:

   .. math::

      \begin{gathered}
         \delta_\mathrm{\scriptscriptstyle SEI} = \delta_\mathrm{ref} \hat{\delta} \qquad c_\mathrm{\scriptscriptstyle EC}=c_\mathrm{\scriptscriptstyle EC}^\mathrm{ref} \hat{c}_\mathrm{\scriptscriptstyle EC} 
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

+-----------------------------+----------------------------------------------------------------------------------------------+------------------------------------------------+-----------------------------------------------------------------------------------------+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :math:`\tau_\mathrm{e}`     | :math:`\frac{D_\mathrm{e}^\mathrm{eff,ref} t_\mathrm{c}}{L_0}`                               | :math:`\delta_\kappa`                          | :math:`\frac{L_0 I_0}{\kappa_\mathrm{eff}^\mathrm{ref} \Phi_\mathrm{l}}`                | :math:`\delta_{\kappa_D}`            | :math:`\frac{\delta_\kappa}{2\alpha (1-t_+^0)(1+\frac{\partial \ln f_{\pm}}{\partial \ln c_\mathrm{e}})} \frac{\Phi_\mathrm{l}}{\Phi_T} \frac{c_\mathrm{e}^\mathrm{ref}}{\Delta c_\mathrm{e}^\mathrm{ref}}` |
+-----------------------------+----------------------------------------------------------------------------------------------+------------------------------------------------+-----------------------------------------------------------------------------------------+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :math:`\delta_{\sigma}`     | :math:`\frac{I_0 L_0}{\sigma_\mathrm{ref} \Phi_\mathrm{s}}`                                  | :math:`\tau_\mathrm{s}`                        | :math:`\frac{D_\mathrm{s}^\mathrm{ref} t_\mathrm{c}}{R_\mathrm{s}^2}`                   | :math:`S`                            | :math:`\frac{R_\mathrm{s} I_0}{a_\mathrm{s} D_\mathrm{s}^\mathrm{ref} c_\mathrm{s}^\mathrm{max} F L_0}`                                                                                                     |
+-----------------------------+----------------------------------------------------------------------------------------------+------------------------------------------------+-----------------------------------------------------------------------------------------+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :math:`\hat{k}_0`           | :math:`\frac{F k_0 L_0 }{I_0} c_\mathrm{e}^\mathrm{ref} (c_\mathrm{s}^\mathrm{max})^\alpha`  | :math:`\tau_q`                                 | :math:`\frac{t_\mathrm{c} k_T^\mathrm{ref} }{\rho^\mathrm{ref} c_p^\mathrm{ref} L_0^2}` | :math:`\delta_{\lambda}`             | :math:`\frac{L_0^2 \rho^\mathrm{ref} c_p^\mathrm{ref} }{t_\mathrm{c} \lambda^\mathrm{ref}}`                                                                                                                 |
+-----------------------------+----------------------------------------------------------------------------------------------+------------------------------------------------+-----------------------------------------------------------------------------------------+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :math:`\delta_\mathrm{ref}` | :math:`\frac{t_\mathrm{c} I_0 M_\mathrm{\scriptscriptstyle SEI}}{2 F \rho a_\mathrm{s} L_0}` | :math:`\tau_{\scriptscriptstyle \mathrm{LAM}}` | :math:`\beta t_c \left ( \frac{E_{\mathrm{ref}}}{\sigma_{\mathrm{cr}}}\right )^m`       | :math:`\delta_{\sigma_{\mathrm{h}}}` | :math:`\frac{2\Omega}{9\left(1-\nu\right)}c_{\mathrm{s}}^{\mathrm{max}}`                                                                                                                                    |
+-----------------------------+----------------------------------------------------------------------------------------------+------------------------------------------------+-----------------------------------------------------------------------------------------+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+



With the specified scaling and dimensionless numbers, the models
equation have been reformulated.

.. rubric:: Electrochemical Model

* Mass transport in the electrolyte:
   The mass transport in the electrolyte is calculated in the
   :meth:`c_e_equation <cideMOD.models.electrochemical.nondimensional.ElectrochemicalModel.c_e_equation>` 
   method.

   .. math::

      \begin{gathered}
         \frac{\varepsilon_\mathrm{e}}{\tau_\mathrm{e}}\frac{\partial\hat{c}_\mathrm{e}}{\partial \hat{t}} =
         \hat{\nabla}\cdot \left(\frac{D_\mathrm{e}^\mathrm{eff}}{D_\mathrm{e}^\mathrm{eff,ref}} \hat{\nabla} \hat{c}_\mathrm{e} \right) + \sum_{i=0}^{n_\mathrm{mat}} \hat{j}_{i}
      \end{gathered}

* Charge transport in the electrolyte:
   The charge transport in the electrolyte is calculated in the
   :meth:`phi_e_equation <cideMOD.models.electrochemical.nondimensional.ElectrochemicalModel.phi_e_equation>`
   method.

   .. math::

      \begin{gathered}
         - \hat{\nabla}\cdot \left( \frac{1}{\delta_K} \frac{\kappa_\mathrm{eff}}{\kappa_\mathrm{eff}^\mathrm{ref}} \hat{\nabla}\hat{\varphi}_\mathrm{e} - \frac{1}{\delta_{K_D}} \frac{\kappa_\mathrm{eff}}{\kappa_\mathrm{eff}^\mathrm{ref}} \frac{1+\frac{\Delta T}{T_\mathrm{ref}} \hat{T}}{1+\frac{\Delta c_\mathrm{e}}{c_\mathrm{e,ref}} \hat{c}_\mathrm{e}} \hat{\nabla} \hat{c}_\mathrm{e}   \right) = \sum_{i=0}^{n_\mathrm{mat}} \hat{j}_i
      \end{gathered}

* Charge transport in the electrodes and current collectors:
   The charge transport in the solid electron conductor materials is calculated in the
   :meth:`phi_s_equation <cideMOD.models.electrochemical.nondimensional.ElectrochemicalModel.phi_s_equation>`
   method.

   .. math::

      \begin{gathered}
         -\hat{\nabla}\cdot \left( \frac{1}{\delta_{\sigma}} \frac{\sigma_\mathrm{eff}}{\sigma_\mathrm{eff}^\mathrm{ref}} \hat{\nabla} \hat{\varphi}_\mathrm{s} \right) = -\sum_{i=0}^{n_\mathrm{mat}} \hat{j}_i 
         ;\quad  
         \frac{1}{\delta_{\sigma}} \frac{\sigma_\mathrm{eff}}{\sigma_\mathrm{eff}^\mathrm{ref}} \frac{\partial \hat{\varphi}_\mathrm{s}}{\partial \mathbf{n}} \Bigg|_\mathrm{tab} = \hat{I}_\mathrm{app} 
      \end{gathered}

* Mass transport in the active material (pseudodimension):
   The mass transport in the active material is calculated in the
   :class:`SpectralLegendreModel <cideMOD.models.particle_models.implicit_\mathrm{c}oupling.NondimensionalSpectralModel>`
   class using Legendre polynomials.

   .. math::

      \begin{gathered}
         \frac{1}{\tau_\mathrm{s}} \frac{\partial \hat{c}_\mathrm{s}}{\partial \hat{t}} = \frac{1}{\hat{r}^2}\frac{\partial}{\partial \hat{r}} \left( \hat{r}^2 \frac{D_\mathrm{s}}{D_\mathrm{s}^\mathrm{ref}} \frac{\partial \hat{c}_\mathrm{s}}{\partial \hat{r}} \right) 
         ; \quad 
         \frac{D_\mathrm{s}}{D_\mathrm{s}^\mathrm{ref}} \frac{\partial \hat{c}_\mathrm{s}}{\partial \hat{r}} \Bigg|_{\hat{r}=1} = S \hat{j}_i
      \end{gathered}

* Exchange between the electrolyte and the electrode by lithium intercalation:
   The intercalation exchange current between the electrolyte and the
   active materials is calculated in the
   :meth:`j_int <cideMOD.models.electrochemical.nondimensional.ElectrochemicalModel.j_Li_equation>`
   method.

   .. math::

      \begin{gathered}
         \hat{j}_i = \hat{k}_0 \left( \left( 1+\frac{\Delta c_\mathrm{e}}{c_\mathrm{e,ref}} \hat{c}_\mathrm{e} \right) \hat{c}_\mathrm{s}|_{\hat{r}=1} (1-\hat{c}_\mathrm{s}|_{\hat{r}=1}) \right)^{0.5} 2 \sinh{\hat{\eta}}
      \end{gathered}

* Overpotential
   The overpotential at each part of the electrode is calculated in the 
   :meth:`overpotential <cideMOD.models.electrochemical.nondimensional.ElectrochemicalModel.overpotential>`
   method.

   .. math::

      \begin{gathered}
         \hat{\eta} = \frac{\Phi_\mathrm{s}}{\Phi_T} \hat{\varphi}_\mathrm{s} - \frac{\Phi_\mathrm{l}}{\Phi_T} \hat{\varphi}_\mathrm{e} - \hat{U}_\mathrm{eq}
      \end{gathered}

.. rubric:: Thermal Model

* Energy conservation:
   The heat transfer across the cell is computed in the
   :meth:`T_equation <cideMOD.models.thermal.nondimensional.ThermalModel.T_equation>`
   function.
   
   .. math::

      \begin{gathered}
         \frac{\rho c_p}{\rho^\mathrm{ref} c_p^\mathrm{ref}} \frac{\partial \hat{T}}{\partial \hat{t}} = \frac{1}{\delta_{\lambda}}\hat{\nabla}\cdot \left( \frac{\lambda}{\lambda^\mathrm{ref}} \hat{\nabla} \hat{T} \right) + \hat{q}  
         \\
         \frac{\lambda}{\lambda^\mathrm{ref}} \frac{\partial \hat{T}}{\partial \mathbf{n}} \Bigg|_{\Gamma} = \frac{L_0 h}{\lambda^\mathrm{ref} \Delta T_\mathrm{ref}} \left(T_0-T_\mathrm{ext} + \Delta T_\mathrm{ref} \hat{T} \right)
      \end{gathered}


* Heat generation:
   Several heat sources have been considered. They are added in the
   :meth:`T_equation <cideMOD.models.thermal.nondimensional.ThermalModel.T_equation>`
   directly.

   .. math::

      \begin{gathered}
         \hat{q} = \hat{q}_\mathrm{ohm} + \hat{q}_\mathrm{rev} + \hat{q}_\mathrm{irr}
      \end{gathered}

   * Ohmic heat source
      This corresponds to the heat generated by the transport of charge within the cell.
   
      .. math::

         \begin{align*}
            \hat{q}_\mathrm{ohm} &=  \hat{q}_\mathrm{solid} + \hat{q}_\mathrm{liquid} \\
            \hat{q}_\mathrm{solid} &=  \frac{1}{\delta_{\sigma}} \frac{\sigma_\mathrm{eff}}{\sigma_\mathrm{eff}^\mathrm{ref}} \frac{\Phi_\mathrm{s}}{\Phi_T} \hat{\nabla} \hat{\varphi}_\mathrm{s} \hat{\nabla} \hat{\varphi}_\mathrm{s} \\
            \hat{q}_\mathrm{liquid} &= \frac{\Phi_\mathrm{l}}{\Phi_T} \frac{\kappa_\mathrm{eff}}{\kappa_\mathrm{eff}^\mathrm{ref}} \left(\frac{1}{\delta_{\kappa}}  \hat{\nabla} \hat{\varphi}_\mathrm{e} \hat{\nabla} \hat{\varphi}_\mathrm{e} - \frac{1}{\delta_{\kappa_D}} \frac{1+\frac{\Delta T}{T_\mathrm{ref}} \hat{T}}{1+\frac{\Delta c_\mathrm{e}}{c_\mathrm{e,ref}} \hat{c}_\mathrm{e}} \hat{\nabla} \hat{c}_\mathrm{e} \hat{\nabla} \hat{\varphi}_\mathrm{e} \right)
         \end{align*}
   
   * Reversible reaction heat source
      The reversible heat caused by the reaction is proportional to the
      entropy change, that is approximated with the variation of Open
      Circuit Potential.

      .. math::

         \begin{gathered}
            \hat{q}_\mathrm{rev} =  \sum_{i=0}^{n_\mathrm{mat}} \hat{j}_{i} \frac{T}{\Phi_T} \frac{\partial U_i(c^\mathrm{surf}_\mathrm{s})}{\partial T}
         \end{gathered}

   * Irreversible polarization heat source
      This represents the irreversible heating due to the polarization
      heat generated by the exchange current at the
      electrolyte-electrode interface.

      .. math::

         \begin{gathered}
            \hat{q}_\mathrm{irr} =  \sum_{i=0}^{n_\mathrm{mat}} \hat{j}_{i} \hat{\eta}
         \end{gathered}

.. rubric:: Degradation Models

* SEI formation side reaction
   This model is implemented inside the
   :class:`SolventLimitedSEIModel <cideMOD.models.degradation.nondimensional.SolventLimitedSEIModel>`
   class. The model considers that the SEI is originated by the
   electrochemical reaction between a EC solvent molecule, two lithium ions
   and two electrons at the electrode surface:

   .. math::

      \begin{gathered}
         \rm EC + 2 Li^+ + 2 e^- \rightarrow V_\mathrm{\scriptstyle SEI}
      \end{gathered}
   
   Therefore, the reaction equation reads:

   .. math::

      \begin{gathered}
         \hat{j}_\mathrm{\scriptscriptstyle SEI} = \frac{F L_0 k_\mathrm{\scriptscriptstyle SEI}}{I_0} c_\mathrm{\scriptscriptstyle EC}^\mathrm{ref} \hat{c}_\mathrm{\scriptscriptstyle EC} e^{-\frac{\beta}{\alpha}(\hat{\eta} - (\hat{U}_\mathrm{\scriptscriptstyle SEI} - \hat{U}_\mathrm{eq}))}
      \end{gathered}

   where the concentration of EC solvent in the SEI is modelled
   according to the transport equation:

   .. math::

      \begin{gathered}
         \frac{\partial \hat{c}_\mathrm{\scriptscriptstyle EC}}{\partial \hat{t}} - \frac{\hat{x}}{\hat{\delta}_\mathrm{\scriptscriptstyle SEI}} \frac{\partial \hat{\delta}_\mathrm{\scriptscriptstyle SEI}}{\partial \hat{t}} \hat{\nabla} \hat{c}_\mathrm{\scriptscriptstyle EC} 
         = \hat{\nabla}\cdot \left( \frac{t_\mathrm{c} D_\mathrm{\scriptscriptstyle EC} }{\delta_\mathrm{ref}^2} \frac{\hat{\nabla} \hat{c}_\mathrm{\scriptscriptstyle EC}}{\hat{\delta}_\mathrm{\scriptscriptstyle SEI}^2} - \frac{ \partial \hat{\delta}_\mathrm{\scriptscriptstyle SEI}}{\partial \hat{t}} \hat{c}_\mathrm{\scriptscriptstyle EC} \right)
      \end{gathered}

   with the following boundary conditions:

   .. math::

      \begin{gathered}
         \left( \frac{t_\mathrm{c} D_\mathrm{\scriptscriptstyle EC} }{\delta_\mathrm{ref}^2} \frac{\hat{\nabla} \hat{c}_\mathrm{\scriptscriptstyle EC}}{\hat{\delta}_\mathrm{\scriptscriptstyle SEI}^2} - \frac{ \partial \hat{\delta}_\mathrm{\scriptscriptstyle SEI}}{\partial \hat{t}} \hat{c}_\mathrm{\scriptscriptstyle EC} \right) \Bigg|_{\hat{x}=0} 
         = \frac{2 \rho_\mathrm{\scriptscriptstyle SEI}}{M_\mathrm{\scriptscriptstyle SEI} c_\mathrm{\scriptscriptstyle EC}^\mathrm{ref}} \hat{j}_\mathrm{\scriptscriptstyle SEI}
         \quad ; \quad
         \hat{c}_\mathrm{\scriptscriptstyle EC} \big|_{\hat{x}=1} = 1
      \end{gathered}

   The SEI growth can be calculated from the reaction rate and the physical properties 
   of the SEI layer:

   .. math::

      \begin{gathered}
         \frac{\partial \hat{\delta}_\mathrm{\scriptscriptstyle SEI}}{\partial \hat{t}} = - \hat{j}_\mathrm{\scriptscriptstyle SEI}
      \end{gathered}

   Thus, the total exchange current has two components:

   .. math::

      \begin{gathered}
         \hat{j}_\mathrm{tot} = \hat{j}_\mathrm{int} + \hat{j}_\mathrm{\scriptscriptstyle SEI}
      \end{gathered}

   And the overpotential has now an additional component corresponding
   to the voltage drop caused by SEI resistance:

   .. math::

      \begin{gathered}
         \hat{\eta} = \frac{\Phi_\mathrm{s}}{\Phi_T} \hat{\varphi}_\mathrm{s} - \frac{\Phi_\mathrm{l}}{\Phi_T} \hat{\varphi}_\mathrm{e} - \hat{U}_\mathrm{eq} - \frac{\delta_\mathrm{ref} I_0}{\kappa_\mathrm{\scriptscriptstyle SEI} L_0 a_\mathrm{s} \Phi_T} \hat{\delta}_\mathrm{\scriptscriptstyle SEI} \hat{j}_\mathrm{tot} 
      \end{gathered}

* LAM model
    This model is implemented inside the
    :class:`SEI <cideMOD.models.degradation.nondimensional.LAM_model>` class.
    The model computes the loss of active material due to particle
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
            -\tau_{\mathrm{\scriptscriptstyle LAM}}\left(\hat\sigma_{\mathrm{h}}\right)^m
            \qquad \hat\sigma_{\mathrm{h}}>0
        \end{gathered}
