Install cideMOD docker image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
*Source:* `cideMOD DockerHub
<https://hub.docker.com/r/cidetec/cidemod>`_

Once Docker is installed, the next step is to download the cideMOD
image and create the container.

1. Download the Docker image.

.. code-block:: console

   $ docker pull cidetec/cidemod

2. Check that the image is installed correctly.

.. code-block:: console

   $ docker images

3. Next, a folder is created that will be linked to the Docker
   container for file sharing.

.. code-block:: console

   $ mkdir cideMOD

4. Finally, the container is created based on the cideMOD image
   and the link between the content folder and the newly created folder
   is created.

.. code-block:: console

   $ docker run -ti --name cideMOD -v /home/{user_name}/{path_to}/cideMOD:/home/cideMOD/shared cidetec/cidemod

The cideMOD container terminal should be opened.
