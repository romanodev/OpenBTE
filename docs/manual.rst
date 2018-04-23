.. OpenBTE documentation master file, created by
   sphinx-quickstart on Mon Dec  4 16:00:38 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Manual
===================================

.. toctree::
   :hidden:

Below we'll go through OpenBTE's main blocks: Geometry, Material, Solver and Plot.

Geometry
------------------------------------

OpenBTE has flexible models to create porous geometries. Aligned configurations with square lattice are specificed with the keyword ``type = porous/square_lattice`` along with the options ``porosity``, ``lx``, ``ly``, ``lz`` and ``shape = square | circle``. The temperature gradient is assumed to be applied along the x-direction. When ``lz`` is not specified, then the sample is infinite along z and only a two-dimensional simulation is performed. Periodic boundary conditions are applied along both x- and y- directions. 

Example:

.. code-block:: shell

   openbte -g -type=porous/square_lattice -shape=circle -porosity=0.25 -lx=10 -ly=10

This command generates the file ``geometry.hdf5``.

In order to generate a geometry with arbitrary pore shape and configuration, use ``type = porous/custom``. In this case the pores are given in a file. The accepted format is

.. code-block:: shell

    x0_a y0_a x1_a y1_a x2_a y2_a x3_a y3_a 
    x0_b ...
    ...



We note that in some cases, a pore can lie on the unit-cell boundary hence appearing in multiple locations. In this case, only the full description of the pore intersecting any of the boundary is needed. Any periodic repeation will be performed internally. 

Example:

.. code-block:: shell

   echo -2.5 2.5 2.5 2.5 2.5 -2.5 -2.5 -2.5 >polygons.dat
   openbte -g -type=porous/custom -polyfile polygons.dat -lx=10 -ly=10

Finally, a bulk system (often used for testing) can be created with ``type=bulk``.

Example:

.. code-block:: shell

  openbte -g -type=bulk -lx=10 -ly=10 -step=1

   
Material
--------------------------------------

Within the mean-free-path BTE, a material can be simply specified by the bulk MFP distribution. The model for a nongray material is ``model=nongray`` and the material file can be speficied with ``filename``. When the path is not specified, the file is taken from the examples provided with OpenBTE. The file format is


.. code-block:: shell

 mfp_1  kappa_1
 mfp_2  kappa_2
 ... 

where the MFPs are in meters and the distributions are in :math:`K(\Lambda)d\Lambda = Wm^{-1}k^{-1}`. The BTE is solved in MFP as well as angular space. To specify the grid, the following options must be given: ``n_mfp``, ``n_theta`` (azimuthal angle) and ``n_phi`` (polar angle: x-y plane.)

Example:

.. code-block:: shell

 openbte -m -model=nongray -n_mfp=30 -n_phi=48 -n_theta=16 -matfile=Si-300K.dat

Lastly, with the keywords ``model=gray`` and ``mfp`` (in meters)  it is possible to specify a single MFP to be used in the material, i.e. a gray material

Example:

.. code-block:: shell

 openbte -m -model=gray -n_phi=48 -n_theta=16 

To convert data from ShengBTE to OpenBTE format, use 

.. code-block:: shell

 shengbte2openbte

from the directory containing the file ``BTE.cumulative_kappa_scalar``. The output file is named ``mat.dat``.

 
Solver
--------------------------------------
The BTE solver in OpenBTE is iterative. The number of iterations is set with ``max_bte_iter = 10``. In case of zero iteration, the solver will simply perform a standard heat diffusion simulation. 

Example:

.. code-block:: shell

   openbte -s -multiscale -max_bte_iter = 10

Once the simulation is finished, the file ``solver.hdf5`` is created.


Plot
--------------------------------------

Once the simulation is over it is possible to plot relevant results by means of the module ``Plot``. The possible plots are:

``suppression``: Phonon suppression function in the MFP space.
``map\fourier_flux|thermal_flux|fourier_temperature|bte_temperature``: Map of a given variable. Currently it works only in 2D. In case of a flux, it is also possible to specify either the Cartesian axis or the magnitude.
``vtk``: the ``output.vtk`` is created with all the relevant variables. We recommend using Paraview.
   
Note: for both ``map\xxx`` and ``vtk`` options, it is possible to repeat the unit-cell in ``x`` and ``y`` with ``repeat_x`` and ``repeat_y``, respectively. OpenBTE handles the cell-to-node conversion internally, also including the periodicity. 

Example:

.. code-block:: shell

 openbte -p -variable/fourier_flux/magnitude






