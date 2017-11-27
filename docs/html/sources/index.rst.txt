.. OpenBTE documentation master file, created by
   sphinx-quickstart on Fri Nov 24 14:59:19 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. toctree::
   :maxdepth: 2



OpenBTE's documentation
===================================
OpenBTE solves the phonon Boltzmann transport equation in its mean-free-path formulation. Below, we provide info for downloading and getting started. 

---------------------------------------
Download
---------------------------------------

OpenBTE can be download by simply typing in your shell

.. code-block:: shell

   pip install openbte 

---------------------------------------
Getting started
---------------------------------------

The first step for setting up a OpenBTE simulation is creating a geometry. This can be done by the command

.. code-block:: shell

   openbte -g porous/aligned -sh circle -phi=0.25 -lx=10 -ly=10

where ``porous/aligned`` is the geometry model for a two-dimensional porous material. In this case, the shape of the pore is circular, the porosity is 0.25 and the periodicity is 10 nm in both Cartesian directions. Essentially, we simulate only one pore with periodic boundary conditions. The keywords and relative options for other geometry models are listed in the Reference. Once this command is run, the file ``geometry.hdf5`` is created in the current directory. 

To plot the geometry, type che command

.. code-block:: shell

   openbte -p map/geometry


.. image:: images/GettingStarted_1.png
   :width: 60 %
   :align: center

The orange line surrounds the unit cell and represents the areas where periodic boundary conditions are applied, whereas the green line indicates the regions with diffuse scattering.


To setup the bulk material, use the folloing command

.. code-block:: shell

   openbte -m Si-300K --n_mfp = 10

where ``Si-300K`` is the mean-free-path (MFP) distribution of Si at 300 K calculated with ShengBTE_. The flag ``--n_mfp`` indicates the number of MFPs to consider in the solution of the BTE.

The bulk MFP distribution can be plotted with

.. code-block:: shell

   openbte -p material/distr

.. image:: images/GettingStarted_2.png
   :width: 60 %
   :align: center

At this point, we should be able to run the BTE with the command

.. code-block:: shell

   openbte -s bte

The solver will first solve the diffusion equation to provide the first guess for the pseudotemperature, then it will solve the BTE. If you run it in serial on your laptop, it should take a few minutes.

To plot the mangnitude of heat flux, we type

.. code-block:: shell

   openbte -p map/flux_magnitude

 
.. image:: images/GettingStarted_3.png
   :width: 60 %
   :align: center

To pseudotemperature is plotted with

.. code-block:: shell

   openbte -p map/temperature

 
.. image:: images/GettingStarted_7.png
   :width: 60 %
   :align: center

The MFP distribution in the porous materials can be plotted with

.. code-block:: shell

   openbte -p distr

.. image:: images/GettingStarted_4.png
   :width: 60 %
   :align: center


The suppression function is plotted with

.. code-block:: shell

   openbte -p suppression_function

.. image:: images/GettingStarted_5.png
   :width: 60 %
   :align: center


The directional suppression function in the diffusive regime is plotte with

.. code-block:: shell

   openbte -p directional_suppression_function -m 0

.. image:: images/GettingStarted_8.png
   :width: 60 %
   :align: center

.. _ShengBTE: http://www.shengbte.com

while for the ballistic regime, we have

.. code-block:: shell

   openbte -p directional_suppression_function -m -1

.. image:: images/GettingStarted_6.png
   :width: 60 %
   :align: center


.. _ShengBTE: http://www.shengbte.com




