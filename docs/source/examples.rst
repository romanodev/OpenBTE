Examples
=========================================

Interactive examples can be run in Google Colab

.. raw:: html

     <a href="https://colab.research.google.com/drive/18u1ieij2Wn6WEZFN2TmMteYHAJADMdSk?usp=sharing"><img  src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align:text-bottom"></a>



Different shapes and non-uniform areas
#########################################



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


Custom shapes
#########################################


.. code-block:: python
   
   from openbte import Geometry,Material,Solver,Plot
   import numpy as np

   def shape1(**options):
    area = options['area']
    T = options['T']
    f = np.sqrt(2)

    poly_clip = []
    a = area/T/2
    poly_clip.append([0,0])
    poly_clip.append([a/f,a/f])
    poly_clip.append([a/f-T*f,a/f])
    poly_clip.append([-T*f,0])
    poly_clip.append([a/f-T*f,-a/f])
    poly_clip.append([a/f,-a/f])

   return poly_clip


   def shape2(**options):

    area = options['area']
    angle = options['angle']

    dphi = np.pi/2
    L = np.sqrt(area) 
    poly_clip = []
    for ka in range(4):
     ph =  dphi/2 + (ka-1) * dphi + angle
     px  = L * np.cos(ph) 
     py  = L * np.sin(ph) 
     poly_clip.append([px,py])

   return poly_clip  

   #Create Material
   Material(source='database',filename='Si',temperature=300,model='rta2DSym')

   Geometry(porosity=0.05,lx=100,ly=100,step=5,model='lattice',shape='custom',base=[[0,0],[0.5,0.5]],shape_function=[shape1,shape2],shape_options={'T':[0.05,None],'angle':[None,45]})

   #Run the BTE
   Solver(verbose=False)

   #Plot Maps
   Plot(model='maps',repeat=[3,3,1],show=True,write_html=True)


.. raw:: html

    <iframe src="_static/plotly_2.html" height="475px" width="65%"  display= inline-block  ></iframe>
   

Gray model
#########################################

Sometimes you might just want to do some quick calculations using the gray model. In this case no file is needed to create the `Material` object; only the bulk MFP and bulk thermal conductivity need to be specified.


.. code-block:: python

  from openbte import Material,Geometry,Solver,Plot

  Material(model='gray2D',mfp=1e-7,kappa=130) #mfp in m

  Geometry(model='lattice',lx = 10,ly = 10, lz=0,step = 0.5, base = [[0,0]],porosity=0.2,shape='square',direction='x')

  Solver(multiscale=False,max_bte_iter=30)

  Plot(model='maps')

.. raw:: html

    <iframe src="_static/plotly_3.html" height="475px" width="65%"  display= inline-block  ></iframe>
   

Disk
#########################################

Heat source at the center of a disk

.. code-block:: python

   from openbte import Geometry, Solver, Material, Plot

   Material(source='database',model='rta2DSym',filename='rta_Si_300')

   Geometry(model='disk',Rh=1,R=10,step=1,heat_source=0.5)

   Solver(max_bte_iter=30)

   Plot(model='maps')

.. raw:: html

    <iframe src="_static/plotly_4.html" height="475px" width="65%"  display= inline-block  ></iframe>
   
  




