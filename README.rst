Description
===========

Space-dependent Boltzmann transport equation solver for phonons

Community
=========

`Mailing list <https://groups.google.com/forum/#!forum/openbte>`_


Installation $ Usage
====================

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

 from openbte.material import *
 from openbte.geometry import *
 from openbte.solver import *
 from openbte.plot import *

 mat = Material(matfile='Si-300K.dat',n_mfp=100,n_theta=24,n_phi=32)

 geo = Geometry(type='porous/square_lattice',lx=10,ly=10,
               porosity = 0.25,
               step = 1.0,
               shape = 'square')

 sol = Solver()

 Plot(variable='map/flux',direction='x')




