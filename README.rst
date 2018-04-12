Description
===========

Space-dependent Boltzmann transport equation solver for phonons


Installation
============

.. code-block:: shell

  pip install openbte

Usage
=====

.. code-block:: python

 from openbte.material import *
 from openbte.geometry import *
 from openbte.solver import *
 from openbte.plot import *

 mat = Material(matfile='Si-300K.dat',n_mfp=10,n_theta=6,n_phi=32)

 geo = Geometry(type='porous/aligned',lx=10,ly=10,
               porosity = 0.25,
               step = 1.0,
               shape = 'square')

 sol = Solver()

 Plot(variable='map/flux_bte/magnitude')

Author
======

Giuseppe Romano (romanog@mit.edu)



