Inverse Design
==============

Inverse design seeks to find a material structure that leads to the desired material property. In this space, ``OpenBTE`` provides routines for the standard heat conduction equation and Boltzmann transport equation in the single-MFP approximation, both solved over structured grids. We adopt the filtering and projecting method [`O. Sigmund and K. Maute (2013) <https://link.springer.com/article/10.1007/s00158-013-0978-6>`_] and the newly introduce Transmisison Interpolation Method (TIM) [`G. Romano and S. G Johnson (2022) <https://link.springer.com/article/10.1007/s00158-022-03392-w>`_]. The automatic differentiation framework is based on `JAX <https://github.com/google/jax>`_. The optimization algorithm of choice is the method of moving asymptotes (MMA) [`Svanberg (2002) <https://doi.org/10.1137/S1052623499362822>`_], implemented in `NLOpt <https://nlopt.readthedocs.io/en/latest/>`_. The first step is to import a solver from the submodule ``openbte.inverse``

.. code-block:: python

  from openbte.inverse import bte
  from openbte.inverse import fourier

Currently, only periodic structures can be simulated, with the perturbation being aligned along specified directions. Multiple directions can be considered in the same simulations, e.g.


.. code-block:: python

   grid = 20

   b = bte.get_solver(Kn=1, grid=grid,directions=[[1,0],[0,1]])
   f = fourier.get_solver(grid=grid,directions=[[1,0],[0,1]])


Note that both solvers also take the grid in input, with the BTE solver also requiring the Knudsen number. The variables ``f`` and ``b`` are fully differentiable solvers and can be integrated in an optimization pipeline. Here, we define the following cost function


.. code-block:: python

  from jax import numpy as jnp

  kd = jnp.array([0.3,0.2])
  def objective(x):

     k,aux  = f(x)

     g  = jnp.linalg.norm(k-kd)

     return g,(k,aux)

which seeks to engineer a material with a thermal conductivity tensor having the components :math:`\kappa_{xx}=0.3` and :math:`\kappa_{xx}=0.2`. Note that in this case, we chose to use the Fourier solver. Lastly, we start the optimization with

.. code-block:: python

   from openbte.inverse import matinverse as mi

   L = 100 #nm
   R = 30  #nm

   x = mi.optimize(objective,grid = grid,L = L,R = R,min_porosity=0.05)

where we specify the size of the simulation domain, the radius of the conic filter

.. math::

   w_c =  \begin{cases}
  \frac{1}{a}\left(1-\frac{\mid \mathbf{r}_c\mid}{R}\right),& \mid \mathbf{r} \mid <R\\
    0,              & \text{otherwise}.
  \end{cases}

Note that we also added the constraint of minimum porosity. Supported constraints include ``min_porosity`` and ``max_porosity``. Other options include:

- ``n_betas``: The number of beta doubling for projection (default = 10). It starts with :math:`\beta = 2`, and always end with a pass with a very large :math:`\beta`, not accounted for in this option
- ``max_iter``: The number of iteration for each beta (default=25)
- ``tol``: tolerance on the cost function
- ``inequality_constraints``: a list of function to be used as inequality constraints. They follow the same syntax as the objective function
- ``monitor``: whether to have intermediate structures plotted during optimization
- ``output_file``: whether to save the convergence history (default==``output``).

Although this example pertains to thermal transport, the inverse design framework is quite general and can be used in combination with other differentiable solvers. If you wish to plug your own solver, it must follow this syntax

.. code-block:: python

   def solver(x)

       ...
       ...

       return (output,aux),jacobian

where ``x`` is the material density (:math:`N`), ``output`` is the cost function (:math:`M`), and ``jacobian`` is the sensitivity of the cost function with respect to the material density (:math:`M\times N`). Then, it can be interfaced with


.. code-block:: python

   from openbte.inverse import matinverse as mi

   s = mi.compose(solver)

From now on, ``s`` can be used in the optimizion pipeline, which includes filtering and projection. Note that ``aux`` includes variables that are not directly related to optimization but that are still worth retaining for later use, e.g. the temperature map.   
       








