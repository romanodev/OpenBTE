Examples
=========================================


Different shapes and non-uniform areas
#########################################


.. raw:: html

     <a href="https://colab.research.google.com/drive/18u1ieij2Wn6WEZFN2TmMteYHAJADMdSk?usp=sharing"><img  src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align:text-bottom"></a>


.. code-block:: python

   from openbte import Geometry, Solver, Material, Plot

   #Create Material
   Material(source='database',filename='Si',temperature=300,model='rta2DSym')

   #Create Geometry - > remember that in area_ratio, what matters is only the relative numbers, i.e. [1,2] is equivalent to [2,4]
   Geometry(model='lattice',lx = 10,ly = 10, step = 0.5, base = [[0.2,0],[-0.2,0]],porosity=0.1,shape=['circle','square'],area_ratio=[1,2])

   #Run the BTE
   Solver(verbose=False)

   #Plot Maps
   Plot(model='maps',repeat=[3,3,1])

.. raw:: html

    <iframe src="_static/plotly_1.html" height="475px" width="65%"  display= inline-block  ></iframe>





