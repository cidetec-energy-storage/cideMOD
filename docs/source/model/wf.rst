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
