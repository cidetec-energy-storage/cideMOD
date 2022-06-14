cideMOD
===================
**cideMOD** refers to the Doyle-Fuller-Newman model in which the physicochemical equations are solved by Finite Element methods using `FEniCS <https://fenicsproject.org/>`_ library. It enables doing physics-based battery simulations with a wide variety of use cases, from different drive cycles to characterization techniques such as GITT/PITT. 

cideMOD is a pXD (pseudo X-dimensional) model that extends the original P2D (pseudo 2-dimensional) model, proposed by John Newman and co-workers, from 1D to 2D and 3D battery geometry. Therefore, charge balance, mass balance and reaction kinetics, as well as energy balance, are spatially resolved for the entire battery geometry, considering the inhomogeneity of battery state properties.
cideMOD has some additional models for solving the cell thermal behaviour, including mayor heat sources, and studying the battery degradation with the SEI growth. It also supports several active materials in the electrodes, and, nonlinear and temperature dependent electrode and electrolyte transport properties. 

It allows complete customization of the cell geometry including the tap position for optimal configuration, as well as highly customizable simulation conditions

Installation
------------

Read the Installation Section in the documentation for more information and installation options.

The cideMOD model is based on the finite element platform **FEniCS** and the library **multiphenics**.
From here we will assume that the user has a working environment with *FEniCS* and *multiphenics*.

To use cideMOD, first install it using pip :

.. code-block:: console
    
   $ git clone https://github.com/cidetec-energy-storage/cideMOD.git
   $ cd cideMOD
   $ pip install .

The P3D/P4D models make use of **gmsh** meshes to create the battery mesh. Therefore, the python environment should be able to locate the **gmsh** shared libraries.
If your *PYTHONPATH* doesn't contains gmsh, you should add it:

.. code-block:: console
    
   $ export PYTHONPATH=$PYTHONPATH:<path_to_gmsh_libs>

or

.. code-block:: console
    
   $ export PYTHONPATH=$PYTHONPATH:$(find /usr/local/lib -name "gmsh-*-sdk")/lib

Additionally **gmsh** needs from some libraries that you may not have installed:

.. code-block:: console
    
   $ sudo apt-get update
   $ sudo apt-get install libglu1-mesa-dev libxcursor-dev libxinerama-dev

To test if the installation is complete, run a simple test:

.. code-block:: console
    
   $ pytest -m "Chen and p2d and validation"
   

Documentation
-------------

The Documentation is a Work In Progress.
The documentation can be viewed at `ReadTheDocs <https://cidemod.readthedocs.io/en/latest/>`_ .

You can also access the documentation on the docs folder building it (See the requirements.txt file for necessary packages):

.. code-block:: language

   $ cd docs/
   $ make html


Authors
--------
- Raul Ciria (rciria@cidetec.es)
- Clara Ganuza (cganuza@cidetec.es)
- Ruben Parra (rparra@cidetec.es)
- Elixabete Ayerbe (eayerbe@cidetec.es)
