Welcome to OpenBTE's documentation
===================================

.. toctree::
   :maxdepth: 2
   :caption: Example:

As a first step, we import `OpenBTE` related packages
.. jupyter-execute::
   :hide-output:

   import plotly.graph_objects as go
   fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
   fig.show()



..
.. jupyter-execute::

   from openbte import Material,Geometry,Solver,Plot
..
The `material` module is created quering the database

..
.. jupyter-execute::
   :hide-output:

   import plotly.graph_objects as go
   fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
   fig.show()
   #Material(model='unlisted',file_id = '1wJy8zWkmGnUFgXsp8st94G0Qo3Nqa0HC');
..
The `geometry` is built using the `lattice` model, via
..
.. jupyter-execute::
..
   x = 0.5
   base = [[x,0],[x,0.2],[x,-0.2],[x,-0.4],[x,0.4]]
   geo = Geometry(porosity=0.1,lx=100,ly=100,step=10,shape='square',base=base,lz=0,save=False)
..
Finally, we invoke the solver
..
.. jupyter-execute::
   :hide-output:

   sol = Solver(geometry=geo,max_bte_error=1e-2,save=False)

..
Results can be shown using the `Plot` module
..
.. jupyter-execute::

   Plot(model='maps',geometry=geo,solver=sol)

   

   

