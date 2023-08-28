Install from source
--------------------

.. _FEniCSx: https://fenicsproject.org/download/archive/
.. _multiphenicsx: https://github.com/multiphenics/multiphenicsx
.. _gmsh: https://gmsh.info/

You can find the installation guide in the documentation of the
required packages:

.. rubric:: Requirements

- `FEniCSx`_ :cite:p:`FEniCSx`
- `multiphenicsx`_ :cite:p:`multiphenicsx`
- `gmsh`_ :cite:p:`gmsh`

From here we will assume that the user has a working environment with
*FEniCSx* and *multiphenicsx*.

To use cideMOD, first install it from source using pip.

.. code-block:: console

    $ git clone https://github.com/cidetec-energy-storage/cideMOD.git cideMOD
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

    $ pytest -m "quicktest"
