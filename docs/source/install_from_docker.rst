Install from docker
^^^^^^^^^^^^^^^^^^^^

.. _docker_instructions:

Docker is the preferred platform to use the FEniCS, multiphenics and cideMOD packages on. 
To install Docker, read the instructions on :doc:`how to install Docker <install_docker>` or in the Docker webpage.
When you have Docker installed and readily available, you just need to create a container with all the required packages.

When you have docker installed and running, you need to open a terminal and use the dollowing command to pull the cideMOD image:

.. code-block:: console
    
   $ docker pull cidetec/cidemod

The image will start to download, this can take a few minutes. 
When the image is completely downloaded, you can start a container using:

.. code-block:: console
    
   $ docker run --name cidemod -ti cidetec/cidemod

It is also possible to create and run the container using the Docker Desktop GUI, that may be easier for some users.
Inside the terminal of the cideMOD container you have the cideMOD repository cloned in the user folder.
