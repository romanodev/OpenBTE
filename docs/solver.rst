Solver
=========

Currently, OpenBTE supports the `anisotropic-MFP-BTE <https://arxiv.org/abs/2105.08181>`_

.. math::

   \mathbf{F}_\mu \cdot \nabla \tilde{T}_\mu + \tilde{T}_\mu = \left[ \sum_{\mu''} \gamma_{\mu''} \right]^{-1}\sum_{\mu'}\gamma_{\mu'} \tilde{T}_{\mu'}

where :math:`\mathbf{F}=\mathbf{v}_\tau\tau_\mu` is the vectorial MFP and :math:`\gamma_\mu =C_\mu/\tau_\mu`. The first step in solving this equation is to provide a first guess for the lattice temperature via solving the standard heat conduction equation. To this end, we use the ``Fourier`` solver

.. code-block:: python

  from openbte import Fourier

  fourier     = Fourier(mesh,mat.thermal_conductivity,boundary_conditions,\
                        effective_thermal_conductivity=effective_kappa)

Finally, we can solve the BTE with

.. code-block:: python
 
   from openbte import BTE_RTA

   bte     = BTE_RTA(mesh,mat,boundary_conditions,fourier=fourier,\
                     effective_thermal_conductivity=effective_kappa)

The result of the simulations can then be consolidated in a ``OpenBTEResults`` object

.. code-block:: python

   from openbte.objects import OpenBTEResults

   results = OpenBTEResults(mesh=mesh,material = mat,solvers={'bte':bte,'fourier':fourier})

Lastly, we can save the results with

.. code-block:: python

   results.save()

where the file ``state.db`` is saved in you current directory. Optionally you can define a custom ``filename``, without including the ``.db`` suffix. Once ready for postprocessing, results can be load with 

.. code-block:: python

   from openbte.objects import OpenBTEResults

   results = OpenBTEResults.load()

where an optional ``filename`` can be specified.



                        
