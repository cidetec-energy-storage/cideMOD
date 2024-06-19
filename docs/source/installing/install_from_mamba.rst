Install with conda
--------------------

.. rubric:: Requirements

- Python conda environment manager (recommended `Miniforge <https://github.com/conda-forge/miniforge>`_)

cideMOD runs natively in Linux systems, thus the installation should be done in a Linux machine 
or using the `WSL2 backend <https://docs.microsoft.com/en-us/windows/wsl/install>`_ in Windows.
A new conda environment can be setup to install the corresponding versions of dolfinx 
(from its `conda-forge feedstock <https://github.com/conda-forge/fenics-dolfinx-feedstock>`_) 
and multiphenicsx

.. code-block:: console

   $ conda create --name cidemod python=3.11
   $ conda activate cidemod
   $ conda install fenics-dolfinx=0.7.0 fenics-libdolfinx=0.7.0
   $ git clone -b dolfinx-v0.7.0 --depth 1 --single-branch https://github.com/multiphenics/multiphenicsx.git
   $ pip install ./multiphenicsx

To use **cideMOD**, first install it using pip:

.. code-block:: console
   
   $ git clone https://github.com/cidetec-energy-storage/cideMOD.git
   $ pip install ./cideMOD

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
    $ sudo apt-get install libglu1-mesa-dev libxcursor-dev libxinerama-dev libxft2 lib32ncurses6

To test if the installation is complete, run a simple test (within the tests folder).

.. code-block:: console

    $ pytest -m "quicktest"
