Run
===================================
 
OpenBTE can be run either via API or through a properly-formatted ``yaml`` file.

.. tabs::

   .. tab:: API
     
      Assuming you have `rta.h5` in your current directory, create the file ``input.py``

      .. code-block:: python

         from OpenBTE import Solver,Geometry,Material

         Material(model='rta2DSym',n_phi=48)

         Geometry(lx=100,ly=100,lz=0,step=5,base=[[0,0]],porosity=0.3,save=True,shape='circle')

         Solver(only_fourier=False,max_bte_iter=100,alpha=1,max_bte_error= 1e-4)

    In you shell:

    .. code-block:: bash

       python input.py

    If you have a recent laptop, you probably have multiple cores. You can check it with ``lscpu``. To take advantage of of parallel computing, you may use ``mpirun``

    .. code-block:: bash

       mpirun -np 4 python input.py



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

   .. tab:: Command Line

      For whe command line version we feed OpenBTE directly with the contect of the ``YAML`` file instead of the file itself. For example:
      
      .. code-block:: bash

         OpenBTE $Material:\n model: rta2Sym\n n_phi: 48$

      This is exactly like running ``OpenBTE`` with the yaml file:  
   
      .. code-block:: yaml

         Material: 

          model: rta2DSym
          n_phi: 48



