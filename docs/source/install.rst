DOCKER
########################################

     .. code-block:: bash

        docker pull ghcr.io/romanodev/openbte:latest

     Once you pull the image, you can run the code sharing your current directory

     .. code-block:: bash

         docker run --shm-size=10g -v `pwd`:`pwd` -w `pwd` --user $(id -u):$(id -g) --rm --net host ghcr.io/romanodev/openbte:latest  -np 4 python input.py

     where we assumed you have 4 CPUs. Note that in this case, your script ``input.py`` must be in your current directory. Also, for intensive calculations, you might want to increase the size of the available shared memory (here is ``10g``). Keep in mind that the above command will write in your current directory. 


PYPI
#######################################

The latest release can be downloaded from the PyPi repository


      .. code-block:: bash

        apt-get update
        apt-get install build-essential libopenmpi-dev libgmsh-dev 
        pip install --upgrade --no-cache openbte

Note that ``mpi`` and ``gmsh`` are needed.

Development version
#######################################

To access the latest (and often untested) features, you can install the code directly from ``github``

.. code-block:: bash

       apt-get update
       apt-get install build-essential libopenmpi-dev libgmsh-dev 
       pip install --no-cache --upgrade git+https://github.com/romanodev/OpenBTE.git


.. _Docker: https://docs.docker.com/engine/install/ubuntu/


