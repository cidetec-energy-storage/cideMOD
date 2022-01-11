Introduction
=============

**CIDETEC's cideMOD** refers to the Doyle-Fuller-Newman model in which the physicochemical equations are solved by Finite Element methods using `FEniCS <https://fenicsproject.org/>`_ library. 
It enables doing physics-based battery simulations with a wide variety of use cases, from different drive cycles to characterization techniques such as GITT/PITT. 

**CIDETEC's cideMOD** is a pXD (pseudo X-dimensional) model that extends the original P2D (pseudo 2-dimensional) model, proposed by John Newman and co-workers, from 1D to 2D and 3D battery geometry. 
Therefore, charge balance, mass balance and reaction kinetics, as well as energy balance, are spatially resolved for the entire battery geometry, considering the inhomogeneity of battery state properties.

Check out the :doc:`usage` section for further information, including how to :ref:`install <installation>` the project.

.. note::

   This project is under active development.

.. rubric:: Authors

- Raul Ciria (rciria@cidetec.es)
- Clara Ganuza (cganuza@cidetec.es)
- Elixabete Ayerbe (eayerbe@cidetec.es)