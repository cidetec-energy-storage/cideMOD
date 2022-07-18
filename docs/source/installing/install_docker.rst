Install Docker
---------------

To install docker follow the instructions from `Docker <https://docs.docker.com/get-docker/>`_.
In Windows and MacOS the preferred way to use Docker is installing **Docker Desktop**. 
In Linux systems, at the moment it is necesary to use **Docker Engine**.

Windows 
~~~~~~~~

Note: Windows 10 or higher version is required

Detailed instructions are given in the `Windows Docker Webpage <https://docs.docker.com/desktop/windows/install/>`_

1. Set up the `WSL2 backend <https://docs.microsoft.com/en-us/windows/wsl/install>`_.
2. Download the `Docker Desktop installer <https://docs.docker.com/desktop/windows/install/>`_.
3. Install **Docker Desktop** running the downloaded executable. Ensure that the **Install required Windows components for WSL 2** option is selected.

Mac OS
~~~~~~~~

Detailed instructions are given in the `MacOS Docker Webpage <https://docs.docker.com/desktop/mac/install/>`_.

Linux
~~~~~~

Detailed instructions are given in the `Docker engine Webpage <https://docs.docker.com/engine/install/>`_.

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