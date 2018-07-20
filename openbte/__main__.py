from __future__ import absolute_import
import argparse
from .material import *
from .geometry import *
from .solver import *
from .plot import *

def main(args=None):


 parser = argparse.ArgumentParser()

 #Geometry---------
 parser.add_argument('-g', help='geometry',action='store_true')
 parser.add_argument('-type', help='porous/aligned')
 parser.add_argument('-polyfile', help='file with the coordinates of the polygons')
 parser.add_argument('-porosity', help='<0.25>',default=0.25)
 parser.add_argument('-shape', help='<circle> | square | triangle',default='circle')
 parser.add_argument('-lx', help='<10> [nm]',default=10)
 parser.add_argument('-ly', help='<10> [nm]',default=10)
 parser.add_argument('-step', help='Mesh quality: <1> [nm]',default=1)

 #Material
 parser.add_argument('-m', help='material',action='store_true')
 parser.add_argument('-model', help='<nongray> | gray',default='nongray')
 parser.add_argument('-matfile', help='file with the bulk MFP distribution')
 parser.add_argument('-n_theta', help='Number of azimuthal')
 parser.add_argument('-n_phi', help='Number of polar angles')
 parser.add_argument('-n_mfp', help='Number of mfps')

 #Solver
 parser.add_argument('-s', help='solver',action='store_true')
 parser.add_argument('-multiscale', help='solver',action='store_true')
 parser.add_argument('-max_bte_iter', help='solver',default=10)

 #Plot
 parser.add_argument('-p', help='plot',action='store_true')
 parser.add_argument('-variable', help='Plot Results: map/temperature')
 parser.add_argument('-repeat_x',default=3,help='Repetition along x')
 parser.add_argument('-repeat_y',default=3,help='Repetition along y')

 args = parser.parse_args()

 #Create Geometry----------------------------------------

 if args.g:
  Geometry(**vars(args))

 if args.m:
  Material(**vars(args))

 if args.s:
  Solver(**vars(args))

 if args.p:
  Plot(**vars(args))

if __name__ == "__main__":

    main()
