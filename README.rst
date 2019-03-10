Description
===========

Space-dependent Boltzmann transport equation solver for phonons

Community
=========

`Mailing list <https://groups.google.com/forum/#!forum/openbte>`_


Installation
====================

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


Example
=======

.. code-block:: python

   from openbte import Material,Geometry,Solver,Plot
   Material(matfile='Si-300K.dat')
   Geometry(porosity=0.25,lx=10,ly=10,step=1)
   Solver()
   Plot(variable='map/flux',direction='magnitude')

