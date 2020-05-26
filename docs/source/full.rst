Full Collision Operator
===================================

To use OpenBTE with the full scattering operator, a file named `full.h5` must be in your current directory. This file must be a `hdf5` file with the following items: `W` (size N x N, units 1/m/m/m*K/W), `C` (heat capacity, size N, units J/K), `v` (group velocity, size N x 3, m/s) and `kappa` (thermal conductivity tensor, units W/m/K). N is total number of the number of wave vectors times the number of branches.

These quantities must satisty the following Python relation:


.. code-block:: python

   import numpy as np
   S = np.einsum('i,ij->ij',C,v)
   kappa = np.einsum('i,ij,j->ij',S,np.linalg.pinv(W),S)

Once you have the file, type   

.. code-block:: python

   Material(model='full')


