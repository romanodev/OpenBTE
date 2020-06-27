"""
A simple scatter plot
=====================
A scatter plot is created with the plotly library. The figure is interactive,
information are displayed on hover and it is possible to zoom and pan through
the figure.
"""

from openbte import Geometry, Solver, Material, Plot

Material(model='database',filename='Si',temperature=300)
geo = Geometry(model='lattice',lx=100,ly=100,step=5,porosity=0.1,shape='square',base=[[-0.1,-0.1],[0.1,0.1]])

sol = Solver(geometry = geo,umfpack=False,keep_lu=True,verbose=False)
sol.state['kappa'][-1]

Plot(model='maps',repeat=[3,3,1]);

