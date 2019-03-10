Description
===========

Space-dependent Boltzmann transport equation solver for phonons

Community
=========

`Mailing list <https://groups.google.com/forum/#!forum/openbte>`_


Installation
====================

The easiest way to install OpenBTE on Linux/MacOS/Windows is through Anaconda:

1) Install Anaconda 3
2) On Anaconda Prompt type:

.. code-block:: shell

   crate env create gromano/openbte-env
   activate openbte-env

If this method does not work, then you will have to create an enviroment yourself:

.. code-block:: shell

  conda create -n openbte python=3.6
  activate openbte
  conda install -c conda-forge -c gromano openbte
  
  
For Windows you will have to install MSMPI

If you want to avoid installing Anaconda, you can still use the pip system (see below)

Linux
---------------------------------------------------------------

Requirements:

apt-get install -y libopenmpi-dev mpich

Install `gmsh <http://gmsh.info/>`_

.. code-block:: shell

     sudo wget http://geuz.org/gmsh/bin/Linux/gmsh-3.0.0-Linux64.tgz && \
      tar -xzf gmsh-3.0.0-Linux64.tgz && \
      cp gmsh-3.0.0-Linux/bin/gmsh /usr/bin/ && \
      rm -rf gmsh-3.0.0-Linux && \
      rm gmsh-3.0.0-Linux64.tgz

pip install --upgrade openbte     

MacOS
---------------------------------------------------------------

You will have to install gmsh from source, then type
 
 .. code-block:: shell
 
pip install --upgrade openbte

Example
=======

.. code-block:: python

   from openbte import Material,Geometry,Solver,Plot
   
   #create material
   Material(matfile='Si-300K.dat')
   #create geometry
   Geometry(porosity=0.25,lx=10,ly=10,step=1)
   #Solve!
   Solver()
   Plot(variable='map/flux',direction='magnitude')

