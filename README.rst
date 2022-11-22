cideMOD
=======
CIDETEC's open-source tool **cideMOD** is based on the Doyle--Fuller--Newman 
model :cite:p:`Doyle1993` in which the physicochemical equations
are solved by Finite Element methods using the `FEniCS`_
:cite:p:`fenics` library. It enables doing physics-based battery cell
(herein after cell) simulations with a wide variety of use cases, from
different drive cycles to studies of the SEI growth under storage
conditions.

**cideMOD** is a pseudo X-dimensional (pXD) model that extends the
original pseudo 2-dimensional (P2D) model, proposed by Newman and
co-workers, from 1D to 2D and 3D cell geometries. Therefore, charge
balance, mass balance and reaction kinetics, as well as energy balance,
are spatially resolved for the entire cell geometry, considering the
inhomogeneity of cell state properties.
**cideMOD** has some additional models for solving the cell thermal
behaviour, including mayor heat sources, and studying battery
degradation mechanisms (e.g., SEI growth and loss of active material). 
It also supports several active materials in the electrodes, and, 
nonlinear and temperature dependent electrode and electrolyte transport properties.

It allows complete customization of the cell geometry including the tab
position for optimal configuration, as well as highly customizable
simulation conditions

Installation
------------

Read the Installation Section in the documentation for more information
and installation options.

The **cideMOD** model is based on the finite element platform FEniCS
and the library multiphenics. From here we will assume that the
user has a working environment with FEniCS and multiphenics.

To use **cideMOD**, first install it using pip :

.. code-block:: console

   $ git clone https://github.com/cidetec-energy-storage/cideMOD.git
   $ cd cideMOD
   $ pip install .

The P3D/P4D models make use of Gmsh meshes to create the battery
mesh. Therefore, the python environment should be able to locate the
Gmsh shared libraries.
If your *PYTHONPATH* doesn't contains gmsh, you should add it:

.. code-block:: console

   $ export PYTHONPATH=$PYTHONPATH:<path_to_gmsh_libs>

or

.. code-block:: console

   $ export PYTHONPATH=$PYTHONPATH:$(find /usr/local/lib -name "gmsh-*-sdk")/lib

Additionally Gmsh needs from some libraries that you may not have
installed:

.. code-block:: console

   $ sudo apt-get update
   $ sudo apt-get install libglu1-mesa-dev libxcursor-dev libxinerama-dev

To test if the installation is complete, run a simple test:

.. code-block:: console

   $ pytest -m "Chen and p2d and validation"


Documentation
-------------

The documentation can be viewed at
`ReadTheDocs <https://cidemod.readthedocs.io/en/latest/>`_ .

You can also access the documentation on the docs folder
building it (See the requirements.txt file for necessary packages):

.. code-block:: console

   $ cd docs/
   $ make html


Authors
--------
- Raul Ciria (rciria@cidetec.es)
- Clara Ganuza (cganuza@cidetec.es)
- Ruben Parra (rparra@cidetec.es)
- Elixabete Ayerbe (eayerbe@cidetec.es)
