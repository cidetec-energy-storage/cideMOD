Modelling
==========

.. _mesh:

Geometry and mesh
------------------

The *cideMOD* library works on top of FEniCS :cite:p:`fenics` using finite elements to solve the PDEs that represent the cell physics.
It is designed to work on any geometry or mesh provided that they are properly tagged. However, using FEniCS and GMSH :cite:p:`gmsh`, cideMOD is able to create automatically realistic 3D meshes of batteries in pouch cell format with different dimensions and tab positions.

There are different subdomains in the model:

* Current Collector:
   Current collectors are parts of the cell that conduct electricity, but not ions. There are two types, the positive current collector (tag="pcc"), which connects cathode electrodes and the negative current collector (tag="ncc"), which connects anode electrodes.
   Additionally, the current collectors are conected to the input/output of the cell at the tabs. 
* Electrode:
   Electrodes are porous materials composed by active material, binder and conductive agents, in which the pores are filled with electrolyte. 
   They conduct electrons through the solid phase and ions through the electrolyte (ionic) phase, and exchange ions between the solid phase and the electrolyte. 
   There are two types of electrodes, the anode (tag="a") which has the lowest Open Circuit Potential, and the cathode (tag="c"), which has the highest Open Ciruit Potential.
* Separator
   Separator (tag="s") is a porous material filled with electrolyte. The solid part of the separator doesn't conduct electrons nor ions. It is inert to the charge carriers.

The dimensions of these domains can be customized in the cell parameter file. The structure of the cell can be customized up to certain level.
There are other parts of the cell, that don't need to be meshed, but are automatically considered by the model:

* Electrolyte:
   Is the solution composed by solvents and salts that fill the electrodes and separator pores. It is ion conductive, but not electron conductive.
* Sealing:
   The cell is ussually sealed with some materials that prohibits mass and charge exchange with the environment. 
   However these materials have an influence on the energy exchange with the surroundings. This effect is considered as an overal heat transfer coefficient in the thermal model.


Submodels
----------

Depending on the physics of the mechanisms takin place in each domain, a different set of variables is taken into account on each domain. Here we define the models included in cideMOD and the domains that they involve.

* Electrochemical model:
   Electrolyte potential and concentration (:math:`\phi_e` and :math:`c_e`) are defined in the electrodes as well as in the separator. Solid phase potential (:math:`\phi_s`) is defined only at the electrodes. At the current collectors there is also a solid phase potential but due to the disparity of conductivity scales, another variable is created :math:`\phi_{s,cc}`, and continuity is imposed at the interface between electrode and current collectors with a lagrange multiplier. 
   Particle concentration (:math:`c_s`\) and intercalation current density (:math:`j_{Li}`) are also defined as variables but only at their corresponding electrode. This means that each electrode has many of these variables as active materials.
* Thermal model:
   Temperature (:math:`T`) is a scalar function defined in all of the domains. Heat sources considered are ohmic, polarization and reaction heat sources. In isothermal simulations (solve\_thermal=False) the temperature is set as a constant value over the domain.
* Degradation models:
   At the moment SEI growth is implemented, but in future versions other degradation mechanisms will be included.

  * SEI growth model:
     The anode SEI model implemented is based on :cite:t:`Safari2009` and it assumes that the electrolyte solvent concentration (primarily EC and DMC) is the limiting factor for SEI growth.
     Therefore solvent species (:math:`c_{EC}`) transport across the SEI is solved with an spectral method using Laplace polynomials. Additionally side reaction current density and SEI thickness are defined in the anode (:math:`j_{SEI}` and :math:`\delta_{SEI}`). 


Pseudo-dimension
-----------------
So far we've talked about meshes for the cell, but in the PXD models there is a pseudodimension (what P stands for in P2D, P3D, P4D) corresponding to the active materials inner domain. This is not explicitly considered, but an homogeneization links the internal state of such domains with the rest of the model.
In *cideMOD*, we solve the pseudodimension with an sprectral approach. Therefore, we consider a set of global basis functions (even Legendre Polynomials) and assume that the solution is a linear combination of those. Doing that we can solve the problem in a single mesh, and FEniCS matrix.

Additionally, as the particle model is only connected to the rest of the cell through the particle interface, the properties of Legendre polynomials are used to reduce coupling at the jacobian level. Thus the concentration in the domain has been decomposed as:

.. math::

   \begin{gathered}
       \hat{c}_s (\hat{r}, \hat{x}, \hat{t}) = \sum_{i=0}^{n} L_{2i} (\hat{r}) \hat{c}_{s,i} (\hat{x}, \hat{t}) \end{gathered}

Taking advantage of the properties of Legendre polinomials, we know that :math:`L_i(1) = 1` then we can substitute the zeroth order coefficient with a new variable :math:`\hat{c}_{surf}` representing the concentration at the particles surface:

.. math::

   \begin{gathered}
       \hat{c}_s(1,\hat{x},\hat{t}) = \hat{c}_{surf} = \sum_{i=0}^{n} L_{2i} (1) \hat{c}_{s,i} (\hat{x}, \hat{t}) = \sum_{i=0}^{n} \hat{c}_{s,i} (\hat{x}, \hat{t}) \\
       \hat{c}_s (\hat{r}, \hat{x}, \hat{t}) = L_0 \left(\hat{c}_{surf} -  \sum_{i=1}^{n}\hat{c}_{s,i} (\hat{x}, \hat{t})\right)+\sum_{i=1}^{n} L_{2i} (\hat{r}) \hat{c}_{s,i} (\hat{x}, \hat{t})\end{gathered}

This change of variable reduces the jacobian entries, and the calculations needed to evaluate entries of the lithium intercalation flux.

P2D vs P4D
------------
P2D model is very handy for fast calculations, however for bigger cells the P4D model is more realistic and considers the heterogeneities in the cell. There are several differences in the implementation of the P2D and P4D motivated by the difference in complexity of both models.
Although both models contains the same submodels, the implementation is different in the two cases:

.. rubric:: P2D

The P2D model is simpler, and it uses standard FEniCS interval meshes to discretize the 1D cell domain in transversal direction. Each of the subdomains is normalized to (0,1) and therefore, the mesh never change for the same type of cell even when the dimensions of the different subdomains changes.


.. rubric:: P4D

Given the necesity of modeling details in 3D (for example current collector tabs), the normalization strategy, used to get a simpler mesh that could be generated with FEniCS does not work anymore. In this case it preferred to use the real geometry of the cell, generated with gmsh. 
Due to the disparity of scales in the real cell geometry and in order to incorporate contributions from all the residuals in a proper way, the equations are non-dimensionalized. This makes it possible real cell geometry representations, but as a consequence, each time the geometry changes, the mesh needs to change, and the FEniCS engine has to recompile part of the solver code.


Formulation details
--------------------

.. toctree::
   :maxdepth: 2

   model/p2d
   model/p4d