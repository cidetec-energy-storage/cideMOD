Install Required Packages
----------------------------

Windows 10 (or higher)
~~~~~~~~~~~~~~~~~~~~~~~~~

To install FEniCSx on Windows 10, enable the 
`Windows Subsystem for Linux (WSL) <https://docs.microsoft.com/en-us/windows/wsl/install>`_ 
and install the Ubuntu distribution. Then follow the instructions for Ubuntu below.

Ubuntu
~~~~~~~~~

Here, several commands are listed to install all the dependencies
required to run cideMOD. All these commands are in the file 
install_required_packages.sh in the cideMOD repository.
Therefore, you can download and run that file on linux with elevated 
permissions to install all the dependencies before installing cideMOD.
Otherwise, follow the steps here:

To install FEniCSx on Ubuntu, run the following commands:

.. code-block:: console
    
    $ sudo apt-get update
    $ sudo apt-get install software-properties-common
    $ sudo add-apt-repository ppa:fenics-packages/fenics
    $ sudo apt-get update
    $ sudo apt-get install fenics

cideMOD uses gmsh to create the 3D meshes necessary to run the P4D model. 
To run gmsh several libraries should be available in the machine:

.. code-block:: console
    
    $ sudo apt-get -y install \
        libglu1 \
        libxcursor-dev \
        libxft2 \
        libxinerama1 \
        libfltk1.3-dev \
        libfreetype6-dev  \
        libgl1-mesa-dev \
        libocct-foundation-dev \
        libocct-data-exchange-dev && \
        apt-get clean

If pip is not available in the system it should be installed using the package manager to install gsmh, multiphenics and cideMOD:

.. code-block:: console
    
    $ sudo apt-get install python3-pip
    $ python3 -m pip install gmsh
    $ python3 -m pip install https://github.com/multiphenics/multiphenics/archive/master.tar.gz

If the user wants to additionally download the cideMOD repository with the examples and the data it can be cloned in a local folder using git.
To install git use:

.. code-block:: console
    
    $ sudo apt-get install git

With these commands the user should have all the dependencies to be able to follow the :doc:`Installation Guide <install_from_source>`.