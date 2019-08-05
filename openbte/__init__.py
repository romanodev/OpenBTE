new = True
if new:
 from .geometry2 import Geometry
 from .material2 import Material
 from .solver2 import Solver
 from .elasticity import Elasticity
 from .plot2 import Plot
else:
 from .material import Material
 from .geometry import Geometry
 from .solver import Solver
 from .plot import Plot
