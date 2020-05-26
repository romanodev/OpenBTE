Welcome to OpenBTE's documentation!
===================================


    
.. image:: https://docs.google.com/drawings/d/e/2PACX-1vRqrihU3IHGVNRaNN7sc2r5CMphXVz6iT8jesHsX0blyj7GPh5KyiUiOFw8WMH9bHHNZYMzBTIgLPNo/pub?w=1230&h=504
    :alt: my-picture1


Here is a quick example:

.. code-block:: python
  
  Material(model='unlisted',file_id = '1k-cggoZA0Gt2kEpgl0TLf50_1ZVq_UQD'); #from the database (Si at 300 K)
 
  geo = Geometry(lx=100,ly=100,lz=0,step=5,base=[[0,0]],porosity=0.3,Periodic=[True,True,True]) 
 
  sol = Solver(max_bte_iter = 20,max_bte_error = 1e-4,geometry=geo)


.. toctree::
   :maxdepth: 2
   :caption: Material

   full
  
.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference
