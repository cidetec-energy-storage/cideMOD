Introduction
=============

.. _FEniCS: https://fenicsproject.org/
.. _multiphenics: https://github.com/multiphenics/multiphenics
.. _gmsh: https://gmsh.info/

CIDETEC's open-source tool **cideMOD** is based on the Doyle-Fuller-Newman model :cite:p:`Doyle1993` in which the physicochemical equations are solved by Finite Element methods using `FEniCS`_ :cite:p:`fenics` library. 
It enables doing physics-based battery simulations with a wide variety of use cases, from different drive cycles to characterization techniques such as GITT/PITT (Galvanostatic/Potentiostaic Intermitent Tritiation Technique). 

**cideMOD** is a pseudo X-dimensional (pXD) model that extends the original pseudo 2-dimensional (P2D) model, proposed by Newman and co-workers, from 1D to 2D and 3D battery geometry. 
Therefore, charge balance, mass balance and reaction kinetics, as well as energy balance, are spatially resolved for the entire battery geometry, considering the inhomogeneity of battery state properties.

Check out the :doc:`Installation <install>` section for instructions to install the software or the :doc:`Usage <usage>` section for information on how to use the library. 

.. note::

   This project is under active development.

.. rubric:: Requirements

- `FEniCS`_ :cite:p:`fenics`
- `multiphenics`_ :cite:p:`multiphenics`
- `gmsh`_ :cite:p:`gmsh`

.. rubric:: Authors

- Raul Ciria (rciria@cidetec.es)
- Clara Ganuza (cganuza@cidetec.es)
- Elixabete Ayerbe (eayerbe@cidetec.es)

.. rubric:: License

cideMOD is copyright (C) 2022 of CIDETEC Energy Storage and is distributed under the terms of the Affero GNU General Public License (GPL) version 3 or later.
