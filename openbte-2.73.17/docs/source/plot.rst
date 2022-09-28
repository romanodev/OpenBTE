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

The files ``solver.npz``, ``material.npz`` and ``rta.npz`` must be in your current directory. The script creates ``kappa_mode.npz``, which has the following fields:

.. table:: 
   :widths: auto
   :align: center

   +--------------------------+-------------+--------------------------------------------------------------------------+---------------------------------------------------------+
   | **Item**                 | **Shape**   |       **Symbol [Units]**                                                 |    **Name**                                             |
   +--------------------------+-------------+--------------------------------------------------------------------------+---------------------------------------------------------+
   | ``mfp_bulk``             |  N          |   :math:`\Lambda` [:math:`m`]                                            | MFP of the nanostructure                                |
   +--------------------------+-------------+--------------------------------------------------------------------------+---------------------------------------------------------+
   | ``mfp_nano``             |  N          |   :math:`\Lambda_{\mathrm{bulk}}` [:math:`m`]                            | MFP of the bulk                                         |
   +--------------------------+-------------+--------------------------------------------------------------------------+---------------------------------------------------------+
   | ``f``                    |  N          |   :math:`f` [THz]                                                        | Frequency                                               |
   +--------------------------+-------------+--------------------------------------------------------------------------+---------------------------------------------------------+
   | ``kappa_bulk``           |  N          |   :math:`\kappa` [:math:`Wm^{-1}K^{-1}`]                                 | Mode-resolved thermal conductivity of the nanostructure |
   +--------------------------+-------------+--------------------------------------------------------------------------+---------------------------------------------------------+
   | ``kappa_nano``           |  N          |   :math:`\kappa_{\mathrm{nano}}` [:math:`Wm^{-1}K^{-1}`]                 | Mode-resolved thermal conductivity of the bulk          |
   +--------------------------+-------------+--------------------------------------------------------------------------+---------------------------------------------------------+

Suppression function
############################################

With ``model=suppression`` it is possible to obtain the MFP-resolved phonon suppression function

.. code:: python

   Plot(model='suppression')

.. table:: 
   :widths: auto
   :align: center

   +--------------------------+-------------+--------------------------------------------------------------------------+---------------------------------------------------------+
   | **Item**                 | **Shape**   |       **Symbol [Units]**                                                 |    **Name**                                             |
   +--------------------------+-------------+--------------------------------------------------------------------------+---------------------------------------------------------+
   | ``mfp_bulk``             |  N          |   :math:`\Lambda` [:math:`m`]                                            | MFP of the bulk                                         |
   +--------------------------+-------------+--------------------------------------------------------------------------+---------------------------------------------------------+
   | ``suppression``          |  N          |   :math:`S`                                                              | Suppression function                                    |
   +--------------------------+-------------+--------------------------------------------------------------------------+---------------------------------------------------------+


.. _Plotly : https://plotly.com/
.. _Paraview : https://www.paraview.org/


