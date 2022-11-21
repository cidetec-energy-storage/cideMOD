Install cideMOD
----------------

From here we will assume that the user has a working environment with
*FEniCS* and *Multiphenics*. If you don't have these libraries
installed, please install it either from
`source <install_from_source>`_ or via `Docker <install_docker>`_.

To use cideMOD, first install it using pip.

.. code-block:: console

    $ git clone https://github.com/cidetec-energy-storage/cideMOD.git
    $ cd cideMOD
    $ pip install .

The P3D/P4D models make use of **gmsh** meshes to create the cell mesh.
Therefore, the python environment should be able to locate the **gmsh**
shared libraries. If your `$PYTHONPATH` doesn't contains gmsh, you
should add it.

.. code-block:: console

    $ export PYTHONPATH=$PYTHONPATH:<path_to_gmsh_libs>

or

.. code-block:: console

    $ export PYTHONPATH=$PYTHONPATH:$(find /usr/local/lib -name "gmsh-*-sdk")/lib

Additionally **gmsh** needs from some libraries that you may not have
been installed.

.. code-block:: console

    $ sudo apt-get update
    $ sudo apt-get install libglu1-mesa-dev libxcursor-dev libxinerama-dev

To test if the installation is complete, run a simple test.

.. code-block:: console

    $ pytest -m "Chen and p2d and validation"
