Install
===================================

Pypi
##################################################

First, we need to install necessary dependences

.. code-block:: bash

   apt-get update
   apt-get install build-essensial libopenmpi-dev libgmsh-dev swig libsuitesparse-dev

Then, you can install OpenBTE

.. code-block:: bash

   pip install --upgrade --no-cache openbte
  

CONDA
##################################################

.. code-block:: bash

   conda create -n openbte
   conda activate openbte
   conda install -c conda-force -c gromano openbte

DOCKER
##################################################


.. code-block:: bash

   sudo docker pull -t romanodev/openbte

Once you pulled the image, you can run OpenBTE.

.. code-block:: bash

   docker run --cap-add=SYS_PTRACE  -v `pwd`:`pwd` -w `pwd` -p 8050:8050  -i -t romanodev/openbte mpiexec -np 8 --use-hwthread-cpus python input.py

where we assumed you have 8 virtual CPUs.   

If you frequently use Docker, you may want to add your ``user`` to the Docker group. 

.. code-block:: bash

   sudo service docker start
   sudo usermod -a -G docker username
   sudo chkconfig docker on

Execute the following command to make it sure you are added to the Docker group 

.. code-block:: bash

   docker info
   


WINDOWS
##################################################

OpenBTE hasn't been tested fully on Windows. However, I good start is to install ``MSMPI``


COLAB
##################################################

You can run OpenBTE in Google Colab using this link_.

.. _link: https://colab.research.google.com/drive/1eAfX3PgyO7TyGWPee8HRx5ZbQ7tZfLDr?usp=sharing






