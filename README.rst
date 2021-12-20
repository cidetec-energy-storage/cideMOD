cideMOD
===================
cideMOD refers to the Doyle-Fuller-Newman model in which the physicochemical equations are solved by Finite Element methods using `FEniCS <https://fenicsproject.org/>`_ library. It enables doing physics-based battery simulations with a wide variety of use cases, from different drive cycles to characterization techniques such as GITT/PITT. 

cideMOD is a cideMOD (pseudo X-dimensional) model that extends the original P2D (pseudo 2-dimensional) model, proposed by John Newman and co-workers, from 1D to 2D and 3D battery geometry. Therefore, charge balance, mass balance and reaction kinetics, as well as energy balance, are spatially resolved for the entire battery geometry, considering the inhomogeneity of battery state properties.


Installation
------------

The cideMOD model is based on the finite element platform **FEniCS** and the library **multiphenics**. 
From here we will assume that the user has a working environment with *FEniCS* and *multiphenics*.

To use cideMOD, first install it using pip :

.. code-block:: console
    
   $ git clone < repository_path >/cideMOD.git
   $ cd cideMOD
   $ pip install -e .

It is important using the **-e** option in the *install* command. This will create an editable install, which means that if we modify the source code at the cloned folder, we will have those changes the next time we import the library. 

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
You can access the documentation on the docs folder (WIP) building it:

.. code-block:: language

   $ cd docs/
   $ make html


Authors
--------
- Raul Ciria (rciria@cidetec.es)
- Clara Ganuza (cganuza@cidetec.es)
- Elixabete Ayerbe (eayerbe@cidetec.es)
