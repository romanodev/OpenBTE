Introduction
============

OpenBTE is a Python-based tool for modeling particles flux at the nondiffusive level and in arbitrary geometries. Current focus is on thermal transport. The code implements the phonon Boltzmann transport equation, informed by first-principles calculations. A typical OpenBTE simulation is given by the combination of three main blocks: Mesh, Material and Solver. 

.. image:: _static/OpenBTEScheme.png
  :width: 600
  :align: center

When possible, OpenBTE automatically exploits parallelism using Python's `Multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`__ module. By default, all the available virtual cores (vCores) are employed. To run your script with a given number of vCores, use the ``np`` flag, e.g.

.. code-block:: bash

   python run.py -np 4

Documentation is still WIP.






