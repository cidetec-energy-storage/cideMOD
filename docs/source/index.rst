.. cideMOD documentation master file, created by
   sphinx-quickstart on Thu Nov 11 22:35:35 2021.
   You can adapt this file completely to your liking, but it should at
   least contain the root `toctree` directive.

Welcome to cideMOD's documentation!
====================================

CIDETEC's open-source tool **cideMOD** is based on the Doyle-Fuller-
Newman model:cite:p:`Doyle1993` in which the physicochemical equations
are solved by Finite Element methods using the FEniCS :cite:p:`fenics`
library.It enables doing physics-based battery cell (herein after
cell) simulations with a wide variety of use cases, from different
drive cycles to studies of the SEI growth under storage conditions.

**cideMOD** is a pseudo X-dimensional (pXD) model that extends the
original pseudo 2-dimensional (P2D) model, proposed by Newman and
co-workers, from 1D to 2D and 3D cell geometry. Therefore, charge
balance, mass balance and reaction kinetics, as well as energy balance,
are spatially resolved for the entire cell geometry, considering the
inhomogeneity of cell state properties.

Check out the :doc:`Installation <install>` section for instructions to
install the software or the :doc:`Usage <usage>` section for
information on how to use the library.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   intro
   install
   usage
   modeling
   examples
   modules
   citing
   references

