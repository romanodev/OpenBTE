

si = Mat('Si')
ge = Mat('Ge')

c1 = Square(7,6)

c2 = Square(7,6)

Combine(1,2)

c1.add_material(si)

c2.add_material(ge)

dd = Solver('DriftDiffusion')

ee = Solver('Elasticity')

c1.add_solver(dd)
c1.add_solver(ee)
c2.add_solver(ee)

f = Fabrix()
f.add_geometry()








