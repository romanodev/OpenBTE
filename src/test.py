from material import *
from geometry import *
from solver import *
from plot import *
import argparse

def main(args=None):

 parser = argparse.ArgumentParser()

 #Geometry---------
 parser.add_argument('-g','--geometry', help='porous/aligned')
 parser.add_argument('-phi','--porosity', help='<0.25>',default=0.25)
 parser.add_argument('-sh','--shape', help='<circle> | square | triangle',default='circle')
 parser.add_argument('-lx','--length_x', help='<10> [nm]',default=10)
 parser.add_argument('-ly','--length_y', help='<10> [nm]',default=10)
 parser.add_argument('-q','--mesh', help='Mesh quality: <1> [nm]',default=1)

 #Material
 parser.add_argument('-m','--material', help='File to be loaded (extension not needed)')
 parser.add_argument('-nt','--n_theta', help='Number of thetas',default=6)
 parser.add_argument('-np','--n_phi', help='Number of phis',default=16)
 parser.add_argument('-nm','--n_mfp', help='Number of mfps',default=10)

 parser.add_argument('-m2','--material2', help='File to be loaded (extension not needed)')
 #Solver
 parser.add_argument('-s','--solver', help='Model: bte | fourier | hybrid')

 #Plot
 parser.add_argument('-p','--plot', help='Plot Results: map/temperature')

 args = parser.parse_args()


 #Create Geometry----------------------------------------
 if not args.geometry == None:
  Geometry(model=args.geometry,
               frame = [args.length_x,args.length_y],
               porosity = args.porosity,
               step = args.mesh,
               shape = args.shape)

 #Create Material----------------------------------------
 if not args.material == None:
  Material(filename=args.material,grid=[args.n_mfp,args.n_theta,args.n_phi])


 if not args.material2 == None:
  Material2(filename=args.material2,grid=[args.n_mfp,args.n_theta,args.n_phi])

 if not args.solver == None:
  mat = Material(model='load')
  geo = Geometry(model='load')
  Solver(model=args.solver,material=mat,geometry=geo)

 if not args.plot == None:
  Plot(model=args.plot)

if __name__ == "__main__":
    main()

