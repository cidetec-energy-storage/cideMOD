Install dependencies with docker
---------------------------------

Install Docker
^^^^^^^^^^^^^^^

To install docker follow the instructions from
`Docker <https://docs.docker.com/get-docker/>`_. In Windows and MacOS
the preferred way to use Docker is installing **Docker Desktop**. In
Linux systems, at the moment it is necesary to use **Docker Engine**.

Windows
~~~~~~~~

Note: Windows 10 or higher version is required

Detailed instructions are given in the `Windows Docker Webpage
<https://docs.docker.com/desktop/windows/install/>`_

1. Set up the `WSL2 backend <https://docs.microsoft.com/en-us/windows/
   wsl/install>`_.
2. Download the `Docker Desktop installer <https://docs.docker.com/
   desktop/windows/install/>`_.
3. Install **Docker Desktop** running the downloaded executable. Ensure
   that the **Install required Windows components for WSL 2** option is
   selected.

Mac OS
~~~~~~~

Detailed instructions are given in the `MacOS Docker Webpage
<https://docs.docker.com/desktop/mac/install/>`_.

Linux
~~~~~~

.. _install_docker_ubuntu: https://
   docs.docker.com/engine/install/ubuntu

.. _install_docker_website: https://
   docs.docker.com/engine/install/ubuntu/#install-using-the-repository

.. _docker_desktop_linux: https://
   docs.docker.com/desktop/install/linux-install/

*Source:* `Install Docker Engine on Ubuntu <install_docker_ubuntu>`_

Docker can be installed in several ways, the easiest for the user would
be to download the `desktop version <_docker_desktop_linux>`_. If your
device does not have a user interface, here are two ways to install it.

The first thing is to ensure no old versions of Docker are installed.
Older versions of Docker went by the names of `docker`, `docker.io`, or
`docker-engine`. Uninstall any such older versions before attempting to
install a new version.

.. code-block:: console

   $ sudo apt-get remove docker docker-engine docker.io containerd runc

It's OK if `apt-get` reports that none of these packages are installed.

Once assured that there is no old versions of Docker you will move on
to the installation of Docker. On the `official website
<_install_docker_website>`_ you can find different ways. Docker
provides a convenience `script <https://get.docker.com/>`_ to install
Docker into development environments non-interactively. It's useful for
creating a provisioning script tailored to your needs. This example
downloads the script from https://get.docker.com/ and runs it to
install the latest stable release of Docker on Linux.

.. code-block:: console

  $ curl -fsSL https://get.docker.com -o get-docker.sh
  $ sudo sh get-docker.sh


.. note::

  If `curl` package is not found:

  .. code-block:: console

    $ sudo apt install curl

If the installation is successful, Multiphenics can be now installed,
see the Multiphenics installation
:doc:`section </installing/install_multiphenics>`.
On the other hand, if the script installation
fails, or a customized installation is prefered, another way to proceed
is to install from the repository.

Before you install Docker Engine for the first time on a new host
machine, you need to setup the Docker repository. Afterward, you can
install and update Docker from the repository.

Set up the repository
~~~~~~~~~~~~~~~~~~~~~~

1. Update the apt package index and install packages to allow apt to
   use a repository over HTTPS.

  .. code-block:: console

    $ sudo apt-get update

    $ sudo apt-get install \
      ca-certificates \
      curl \
      gnupg \
      lsb-release

2. Add Docker's official GPG key.

   .. code-block:: console

    $ sudo mkdir -p /etc/apt/keyrings

    $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
      sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

3. Use the following command to set up the repository.

   .. code-block:: console

    $ echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"\
      | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null


Install Docker Engine
~~~~~~~~~~~~~~~~~~~~~~

1. Update the apt package index.

   .. code-block:: console

    $ sudo apt-get update

2. Install Docker Engine, containerd, and Docker Compose.

   .. code-block:: console

    $ sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

3. Verify that the Docker Engine installation is successful by running
   the `hello-world` image.

   .. code-block:: console

    $ sudo docker run hello-world

4. Follow the `post-installation steps for Linux
   <https://docs.docker.com/engine/install/linux-postinstall/>`_
   in order to configure properly permissions and background services.

