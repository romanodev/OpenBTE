Run
===================================
 
OpenBTE can be run either via API or through a properly-formatted ``yaml`` file.

.. tabs::

   .. tab:: API
     
      Assuming you have `rta.h5` in your current directory

      .. code-block:: python

         from OpenBTE import Solver,Geometry,Material

         Material(model='rta2DSym',n_phi=48)

         Geometry(lx=100,ly=100,lz=0,step=5,base=[[0,0]],porosity=0.3,save=True,shape='circle')

         Solver(only_fourier=False,max_bte_iter=100,alpha=1,max_bte_error= 1e-4)

   .. tab:: YAML

      Prepare a ``yaml`` file like the one below, and let's name it ``input.yaml``. Note that there are 1-1 correspondance between the API and the ``yaml`` version.

      .. code-block:: yaml

         ---
         Material: 

          model: rta2DSym
          n_phi: 48

         Geometry:

          lx: 100
          ly: 100
          lz: 0
          step: 5
          base: [[0,0]]
          porosity: 0.3
          shape: circle

        Solver:
          
         only_fourier: True      
         max_bte_iter: 100 
         alpha: 1
         max_bte_error: 1e-4

      then, run OpenBTE    
 
      .. code-block:: bash

         OpenBTE input.yaml

      Note that ``input.yaml`` is the default name, so you can omit in this case.    

      
   



