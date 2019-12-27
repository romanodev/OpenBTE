new = True
if new:
 from .geometry2 import Geometry
 from .geometry_gpu import GeometryGPU
 from .material2 import Material
 from .solver import Solver
 from .solver_gpu import SolverGPU
 from .elasticity import Elasticity
 from .plot2 import Plot
else:
 from .material import Material
 from .geometry import Geometry
 from .solver import Solver
 from .plot import Plot
