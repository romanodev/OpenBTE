
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


.. code-block:: python

   from openbte import Plot

   data = Plot('kappa_mode',show=False)

   #Here are all the data field you can read   
   print(data.keys())

