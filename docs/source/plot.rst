Plot
===================================

The ``Plot`` module visualizes and analyzes results, reading ``material.npz``, ``geometry.npz`` and ``solver.nzp``.

Internal Viewer
###################################

OpenBTE features its own viewer, based on plotly_. This experimental feature can be invoked with


.. code:: python

   Plot(model='maps',repeat=[2,2,1])

where ``repeat`` is used to plot the supercell. 


External Viewer
####################################

Alternatively, it is possible to write results in the ``vtu`` format

.. code:: python

   Plot(model='vtu',repeat=[2,2,1])

The created file ``output.vtk`` is compatible with Paraview_

Mode-resolved effective thermal conductivity
############################################
Once you have calculated the effective thermal conductivity, you may want to interpolate back the results on the original mode-resolved grid (e.g. the one used for bulk). You can do so with the model ``kappa_mode``.

.. code:: python

   Plot(model='kappa_mode')

Notes/limitations:

- kappa_mode works only with the material model ``rta2DSym``.

- The files ``solver.npz``, ``material.npz`` and ``rta.npz`` must be in your current directory.

.. image:: _static/kappa_mode.png
   :width: 600





.. _Plotly : https://plotly.com/
.. _Paraview : https://www.paraview.org/


