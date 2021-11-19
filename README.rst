CIDETEC's PXD Model
===================
This is the release of CIDETEC's battery model based on the DFN approach and capablo of simulating 3D battery cells.


Installation
------------

The PXD model is based on the finite element platform **FEniCS** and the library **multiphenics**. 
From here we will assume that the user has a working environment with *FEniCS* and *multiphenics*.

To use PXD, first install it using pip :

.. code-block:: console
    
   $ git clone < repository_path >/PXD.git
   $ cd PXD
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
