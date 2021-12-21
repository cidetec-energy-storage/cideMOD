Use from docker
================

.. _docker_instructions:

Docker is the preferred platform to use the FEniCS, multiphenics and cideMOD packages on.

Install Docker
---------------

To install docker follow the instructions from `Docker <https://docs.docker.com/get-docker/>`_.
In Windows and MacOS the preferred way to use Docker is installing **Docker Desktop**. 
In Linux systems, at the moment it is necesary to use **Docker Engine**.

Windows 
~~~~~~~~

Note: You need Windows 10 or higher

Detailed instructions are given in the `Docker Webpage <https://docs.docker.com/desktop/windows/install/>`_

1. Set up the `WSL2 backend <https://docs.microsoft.com/en-us/windows/wsl/install>`_.
2. Download the `Docker Desktop installer <https://docs.docker.com/desktop/windows/install/>`_.
3. Install **Docker Desktop** running the downloaded executable. Ensure that the **Install required Windows components for WSL 2** option is selected.

Mac OS
~~~~~~~~

Detailed instructions are given in the `Docker Webpage <https://docs.docker.com/desktop/mac/install/>`_.

Linux
~~~~~~

Detailed instructions are given in the `Docker Webpage <https://docs.docker.com/engine/install/>`_.

1. Uninstall old versions if any :code:`sudo apt-get remove docker docker-engine docker.io containerd runc`
2. Set-up the official Docker repository:

.. code:: bash
    
  $ sudo apt-get update
  $ sudo apt-get install \
      ca-certificates \
      curl \
      gnupg \
      lsb-release    
  $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
  $ echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

3. Install Docker Engine:

.. code:: bash
    
  $ sudo apt-get update
  $ sudo apt-get install docker-ce docker-ce-cli containerd.io

4. Follow the `post-installation steps for Linux <https://docs.docker.com/engine/install/linux-postinstall/>`_ in order to configure properly permissions and background services.


Set-up a cideMOD container
---------------------------

If you have already loaded the cideMOD image in docker you can skip to step 3.

0. Download the cideMOD image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The cideMOD is delivered as a Docker image in the `DEFACTO webpage <https://defacto-project.eu/documents/#download>`_. 
Download it (notice it may take a while as the image size is 2.4 GB) and store it in a local folder.

1. Start the docker daemon
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have installed Docker Desktop, you need to open it to start the docker daemon. Then you have to open an WSL Terminal (in the following 'The Terminal') and follow the next steps.
If you have installed the Docker Engine on Linex and configured it properly, the docker daemon should be running in the background.

To check if the daemon is running write :code:`docker images` in the Terminal.

2. Load the downloaded image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
From the Terminal use :code:`docker image load <path_to_downloaded_image>` to load the cideMOD image. When finished you can check if the image is correctly loaded using :code:`docker images`.

3. Create a cideMOD container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Further instructions to create and use Docker containers can be found in the `Docker Webpage <https://docs.docker.com/get-started/>`_.
To create a container use:

.. code:: bash
    
  $ docker run -ti --name <your_container_name> -v <local_shared_folder>:/home/fenics/shared/ cideMOD

Substitute :code:`<your_container_name>` with an appropiate name of your election for this container. 
To access the container files from outside the container (from your computer) the option :code:`-v <local_shared_folder>:/home/fenics/shared/` is added, where :code:`<local_shared_folder>` is a local directory of your computer that will be mapped to the container's shared folder.

With that command, the docker container is set-up and a terminal inside the container should appear in the Terminal.
To exit the container type :code:`exit` in the container terminal.
To start the container if it is stopped, use the following command in the Terminal:

.. code:: bash
    
  $ docker start <your_container_name>

To start a terminal on the started container, use:

.. code:: bash
    
  $ docker exec -ti <your_container_name> bash


4. Run an example cideMOD simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the cideMOD container terminal go to the path :code:`/home/fenics/cideMOD/examples` and run a 1C discharge with:

.. code:: bash
    
  $ python3 main.py


