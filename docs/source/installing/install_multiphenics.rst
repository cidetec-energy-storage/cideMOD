Install Multiphenics Docker image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Source:* `Multiphenics DockerHub
<https://hub.docker.com/r/multiphenics/multiphenics>`_

Once Docker is installed, the next step is to download the Multiphenics
image and create the container to run cideMOD.

Multiphenics is a python library that aimed at providing tools in
FEniCS for an easy prototyping of multiphysics problems on conforming
meshes. In particular, it used to facilitate the definition of
subdomain/boundary restricted variables and enabled the definition of
the problem by means of a block structure.

1. Download the Docker image.

   .. code-block:: console

      $ docker pull multiphenics/multiphenics

2. Check that the image is installed correctly.

   .. code-block:: console

      $ docker images

3. Next, a folder is created that will be linked to the Docker
   container for file sharing.

   .. code-block:: console

      $ mkdir fenics

4. Finally, the container is created based on the Multiphenics image
   and the link between the content folder and the newly created folder
   is created.

   .. code-block:: console

      $ docker run -ti –name fenics -v \
        /home/{user_name}/{path_to}/fenics:/home/fenics/shared multiphenics/multiphenics

The fenics container terminal should be opened.
