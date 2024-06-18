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

Additionally, some extra libraries are needed for the use of **gmsh** meshes 
to create P3D/P4D meshes as described in :doc:`Install with conda </installing/install_from_mamba>`.
