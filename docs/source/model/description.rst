Geometry and mesh
------------------

The *cideMOD* library works on top of FEniCS using finite elements to solve the PDEs involved in the battery physics.
It is designed to work on any geometry or mesh provided that they are properly tagged. However, using FEniCS and GMSH, cideMOD is able to create automatically realistic 3D meshes of pouch cells with different characteristics.

There are different subdomains in the model:

- Current Collector:
  Current collectors are parts of the battery that conduct electricity, but not ions. There are two types, the positive current collector (tag="pcc"), which connects cathodes and the negative current collector (tag="ncc"), which connects anodes.
  Additionally, the current collectors are conected to the input/output of the battery at the tabs. 
- Electrode:
  Electrodes are porous materials filled with electrolyte. They conduct electrons as well as ions, and exchange ions with the electrolyte. There are two types of electrodes, the anode (tag="a") which has the lowest Open Circuit Potential, and the cathode (tag="c"), which has the highest Open Ciruit Potential.
- Separator
  Separator (tag="s") is a porous material filled with electrolyte. The solid part of the separator does not conduct electrons, not ions. It is inert to the charge carriers.

The dimensions of these domains can be customized in the cell parameter file. The structure of the cell can be customized up to certain level.

Models
-------

The different variables in the models are defined only in some of the domains. Considering the models included the variables are defined in the following domains:

- Electrochemical model:
  Electrolyte potential and concentration :math:`\phi_e\, c_e` are defined in the electrodes and the separator. Solid phase potential :math:` \phi_s ` is defined only at the electrodes. At the current collectors there is also a solid phase potential but due to the disparity of conductivity scales, another variable is created :math:`\phi_{s,cc}`, and continuity is imposed at the interface between electrode and current collectors with a lagrange multiplier. Particle concentration :math:`c_s`\ and intercalation current density :math:`j_i` are also defined as variables but only at their corresponding electrode. This means that each electrode has many of these variables as active materials.
- Thermal model:
  In isothermal simulations (solve\_thermal=False) the temperature is set as a constant value over the domain. Otherwise, temperature is a scalar function defined in all of the domains. Heat sources considered are ohmic, polarization and reaction heat sources.
- Degradation models:
  At the moment only SEI growth is implemented, but in future versions other degradation mechanisms will be included. The anode SEI model that is used is based on Safari (2009) and it assumes that
