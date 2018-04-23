.. OpenBTE documentation master file, created by
   sphinx-quickstart on Mon Dec  4 16:00:38 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Getting Started
===================================

.. toctree::
   :hidden:


The first step for setting up a OpenBTE simulation is creating a geometry. This can be done by the command

.. code-block:: shell

   openbte -g -type=porous/square_lattice -shape circle -porosity=0.25 -lx=10 -ly=10

where ``porous/square_lattice`` is the geometry model for a two-dimensional porous material. In this case, the shape of the pore is circular, the porosity is 0.25 and the periodicity is 10 nm in both Cartesian directions. Essentially, we simulate only one pore with periodic boundary conditions. The keywords and relative options for other geometry models are listed in the Reference. Once this command is run, the file ``geometry.hdf5`` is created in the current directory. 

To plot the geometry, type the command

.. code-block:: shell

   openbte -p -variable=map/geometry


.. image:: images/GettingStarted_1.png
   :width: 60 %
   :align: center


To setup the bulk material, use the following command

.. code-block:: shell

   openbte -m -matfile=Si-300K.dat -n_mfp=10 -n_theta=15 -n_phi=32

where ``Si-300K.dat`` is the mean-free-path (MFP) distribution of Si at 300 K calculated with ShengBTE_. The flag ``-n_mfp`` indicates the number of MFPs to consider in the solution of the BTE.


At this point, we should be able to run the BTE with the command

.. code-block:: shell

   openbte -s 

The solver will first solve the diffusion equation to provide the first guess for the pseudotemperature, then it will solve the BTE. If you run it in serial on your laptop, it should take a few minutes.

To plot the mangnitude of heat flux, we type

.. code-block:: shell

   openbte -p map/bte_flux/magnitude

 
.. image:: images/GettingStarted_3.png
   :width: 60 %
   :align: center

To pseudotemperature is plotted with

.. code-block:: shell

   openbte -p map/temperature_bte

 
.. image:: images/GettingStarted_7.png
   :width: 60 %
   :align: center

The MFP distribution in the porous material can be plotted with

.. code-block:: shell

   openbte -p -variable=distribution

.. image:: images/GettingStarted_4.png
   :width: 60 %
   :align: center


The suppression function is plotted with

.. code-block:: shell

   openbte -p suppression_function

.. image:: images/GettingStarted_5.png
   :width: 60 %
   :align: center

In case you are familiar with Python, you can setup a simulation with a script, i.e.

.. code-block:: python

   from openbte.material import *
   from openbte.geometry import *
   from openbte.solver import *
   from openbte.plot import *

   mat = Material(matfile='Si-300K',n_mfp=10,n_theta=6,n_phi=32)

   geo = Geometry(type='porous/aligned',lx=10,ly=10,
                 porosity = 0.25,
                 step = 1.0,
                 shape = 'square')

   sol = Solver()

   Plot(variable='map/flux_bte/magnitude') 


.. _ShengBTE: http://www.shengbte.com




