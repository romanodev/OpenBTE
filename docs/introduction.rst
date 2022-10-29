Introduction
============

OpenBTE is a Python-based tool for modeling particles flux at the nondiffusive level and in arbitrary geometries. Current focus is on thermal transport. The code implements the phonon Boltzmann transport equation, informed by first-principles calculations. Both forward and backward modes are supported, enabling inverse design using direct parameter optimization. A typical OpenBTE simulation is given by the combination of three main blocks: Mesh, Material and Solver. Both forward and backward modes are supported, enabling inverse design using direct parameter optimization.

.. image:: _static/OpenBTEScheme.png
  :width: 600
  :align: center

When possible, OpenBTE automatically exploits parallelism using Python's `Multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`__ module. By default, all the available virtual cores (vCores) are employed. To run your script with a given number of vCores, use the ``np`` flag, e.g.

.. code-block:: bash

   python run.py -np 4

Main features include:

- Vectorial mean-free-path interpolation
- Interface with first-principles solvers
- Arbitrary geometries
- Inverse design
- Interactive temperature and flux maps visualization
- Effective thermal conductivity
- Outputting data in ``.vtu`` format for advanced visualization




