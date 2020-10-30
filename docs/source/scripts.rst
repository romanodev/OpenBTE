
Post-processing Scripts
=========================================


Here is a collection of scripts I use to analyze results.

Extract thermal conductivity
############################################


.. code-block:: python
  
   from openbte.utils import *

   kappa_bte = load_data('solver')['kappa'][-1]
   kappa_fourier = load_data('solver')['kappa'][0]


Extract mode-resolved thermal conductivity
############################################

First, you will need to create the ``kappa_mode`` file with

.. code-block:: python

   from openbte import Plot
   from openbte.utils import *

   #rta.npz, solver.npz and material.npz must be in your current directory
   Plot(model='kappa_mode',save=True,show=False)

   #this create kappa_mode.rtz
   data= load_data('kappa_mode')

   print(data.keys())


