Description
===========

Space-dependent Boltzmann transport equation solver for phonons


Community
=========

`Mailing list <https://groups.google.com/forum/#!forum/openbte>`_


Installation $ Usage
====================

.. code-block:: shell

  pip install --upgrade openbte

  mkdir -p example_openbte && cd example_openbte

  download_openbte_example

  python example.py

Example
=======

.. code-block:: python

 from openbte.material import *
 from openbte.geometry import *
 from openbte.solver import *
 from openbte.plot import *

 mat = Material(matfile='Si-300K.dat',n_mfp=10,n_theta=6,n_phi=32)

 geo = Geometry(type='porous/square_lattice',lx=10,ly=10,
               porosity = 0.25,
               step = 1.0,
               shape = 'square')

 sol = Solver()

 Plot(variable='map/bte_flux/magnitude')

Author
======

Giuseppe Romano (romanog@mit.edu)



