
Post-processing Scripts
=========================================


Here is a collection of scripts I use to analyze results.

Mode-resolved effective thermal conductivity
############################################
Once you have calculated the effective thermal conductivity, you may want to interpolate back the results on the original mode-resolved grid (e.g. the one used for bulk). You can do so with the script kappa_mode.

Notes/limitations:

- kappa_mode works only with the material model ``rta2DSym``.

- The files ``solver.npz``, ``material.npz`` and ``rta.npz`` must be in your current directory.

Below we show an example using the package pubeasy


.. code-block:: python

   from openbte.utils import *
   from pubeasy import MakeFigure

   data = load_data('suppression')
   kappa_nano = data['kappa_nano']
   mfp_nano = data['mfp_nano']
   kappa_bulk = data['kappa_fourier']
   mfp_bulk = data['mfp_bulk']
   f = data['f']

   fig = MakeFigure()

   #fig.add_plot(f*1e-12,mfp_bulk*1e6,model='scatter',color='b')
   fig.add_plot(f*1e-12,mfp_nano*1e6,model='scatter',color='r')

   fig.add_labels('Frequency [THz]','Mean Free Path [$\mu$m]')
   fig.finalize(grid=True,yscale='log',write = True,show=True,ylim=[1e-4,1e2])

.. image:: _static/kappa_mode.png
   :width: 600


Formulation
############################################


The effective thermal conductivity, after interpolation, can be computed as 

.. math::

   \kappa^{\mathrm{eff}}_\mu = C_\mu v_\mu^x \Lambda_\mu^{x,\mathrm{eff}}
 
where

.. math::

  \Lambda_\mu^{x,\mathrm{eff}}= \frac{L}{\Delta T A_{\mathrm{hot}}}\int_{A_\mathrm{hot}} dS \Delta T_\mu(\mathbf{r}).

The mean free path is then defined as


.. math::

   \Lambda_\mu = \sqrt{(\Lambda^y_\mu)^2 + (\Lambda^z_\mu)^2 +   (\Lambda_\mu^{x,\mathrm{eff}})^2}


Note the the effective projected mean-free-path above also includes the macroscopic geometrical effect.  

  




   
