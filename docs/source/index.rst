.. cideMOD documentation master file, created by
   sphinx-quickstart on Thu Nov 11 22:35:35 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to cideMOD's documentation!
====================================

CIDETEC's open-source tool **cideMOD** is based on the Doyle-Fuller-Newman model :cite:p:`Doyle1993` in which the physicochemical equations are solved by Finite Element methods using `FEniCS`_ :cite:p:`fenics` library. 
It enables doing physics-based battery simulations with a wide variety of use cases, from different drive cycles to characterization techniques such as GITT/PITT (Galvanostatic/Potentiostaic Intermitent Tritiation Technique). 

**cideMOD** is a pseudo X-dimensional (pXD) model that extends the original pseudo 2-dimensional (P2D) model, proposed by Newman and co-workers, from 1D to 2D and 3D battery geometry. 
Therefore, charge balance, mass balance and reaction kinetics, as well as energy balance, are spatially resolved for the entire battery geometry, considering the inhomogeneity of battery state properties.

Check out the :doc:`Installation <install>` section for instructions to install the software or the :doc:`usage <usage>` section for information on how to use the library. 

.. note::

   This project is under active development. 
   
   The models included will change in future versions. The code is not fully documented, this documentation will be completed towards the version v1.0

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro
   install
   usage
   modeling
   examples
   modules
   references


