Install
===================================
 

PYPI
####################################

      .. code-block:: bash

        apt-get update
        apt-get install build-essential libopenmpi-dev libgmsh-dev 
        pip install --upgrade --no-cache openbte


DOCKER
####################################

      Install Docker_.

      The image is installed with

      .. code-block:: bash
 
         docker pull romanodev/openbte:latest

      You can run OpenBTE sharing your current directory

      .. code-block:: bash

         docker run --shm-size=10g -v `pwd`:`pwd` -w `pwd` --user "$(id -u):$(id -g)" --net host  romanodev/openbte mpirun -np 4 python input.py

      where we assumed you have 4 virtual CPUs. Note that in this case, your script ``input.py`` must be in your current directory. Also, for intensive calculations, you might want to increase the size of the used shared memory (here is ``10g``). Keep in mind that the above command will allow docker to write in your current directory. 

      If you frequently use Docker, you may want to add your ``user`` to the Docker group. 

      .. code-block:: bash

         sudo service docker start
         sudo usermod -a -G docker $USER
         sudo newgroup docker 

      Execute the following command to make it sure you are added to the Docker group 

      .. code-block:: bash
         
         docker info

WINDOWS
####################################

      Use Docker after Installing MSMPI_.


MacOS
####################################
 

      Install Anaconda_.

      Install mpi4py with

      .. code-block:: bash
         
         conda install -c conda-forge mpi4py

      Install gmsh_.      
 
      Install XCode with

      .. code-block:: bash

         xcode-select --install

      Install OpenBTE with      

      .. code-block:: bash

         pip install --upgrade --no-cache openbte



      Note the you need XSelect installed.



.. _link: https://colab.research.google.com/drive/1eAfX3PgyO7TyGWPee8HRx5ZbQ7tZfLDr?usp=sharing
.. _Docker: https://docs.docker.com/engine/install/ubuntu/
.. _Anaconda: https://docs.anaconda.com/anaconda/install/
.. _MSMPI: https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi
.. _gmsh: https://gmsh.info/


