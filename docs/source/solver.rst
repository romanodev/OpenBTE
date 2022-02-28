Solver
===================================

Solver reads the files `geometry.npz` and `material.npz` and, after solving the BTE, creates `solver.npz`. Example:

.. code:: python

   Solver(**options)

Options
-------------------------
 - ``max_bte_iter`` : maximum number of BTE iteration 
 - ``max_bte_error`` : maximum error for the BTE solver (computed for the thermal conductivity)
 - ``max_fourier_iter`` : maximum number of Fourier iteration 
 - ``max_fourier_error`` : maximum error for the Fourier solver (computed for the thermal conductivity)
 - ``only_fourier`` : whether to compute only Fourier

