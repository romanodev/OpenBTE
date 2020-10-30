
OpenBTE documentation
===================================

    
OpenBTE is an open-source solver for the first-principles Boltzmann transport equation, with a focus on phonon transport. Here is an example to calculate the effective thermal conductivity of nanoporous silicon:

NOTE: a multiscale model is coming soon (check back in a few days)

.. raw:: html

     <a href="https://colab.research.google.com/drive/1aCGJdg3Mm_B_0iUrJHHAk7-J-RE7QAeL?usp=sharing"><img  src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align:text-bottom"></a>


.. code-block:: python

   from openbte import Geometry, Solver, Material, Plot

   #Create Material
   Material(source='database',filename='Si',temperature=300,model='rta2DSym')

   #Create Geometry
   Geometry(model='lattice',lx=100,ly=100,step=5,porosity=0.1,base=[[-0.1,-0.1],[0.1,0.1]])

   #Run the BTE
   Solver(verbose=False)

   #Plot Maps
   Plot(model='maps',repeat=[3,3,1]);


.. raw:: html

    <iframe src="_static/plotly.html" height="475px" width="65%"  display= inline-block  ></iframe>

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   features
   install
   run


.. toctree::
   :maxdepth: 2
   :caption: Modules

   Geometry
   Material
   Solver
   Plot


.. toctree::
   :maxdepth: 2
   :caption: Database

   database

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples

.. toctree::
   :maxdepth: 2
   :caption: Scripts

   scripts

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference
