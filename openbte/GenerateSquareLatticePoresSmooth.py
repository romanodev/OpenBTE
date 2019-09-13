import numpy as np
import math
from shapely.geometry import Polygon

from .special_shapes import *




def GenerateSquareLatticePoresSmooth(argv):

  porosity = float(argv['porosity'])
  Lx = float(argv['lx'])#*argv.setdefault('Nx',1)
  Ly = float(argv['ly'])#*argv.setdefault('Ny',1)

  Na = 12
  #Read shape
  smooth = argv.setdefault('smooth',Lx/10)
  
  #Read Number of pores
  Nx = argv.setdefault('Nx',1)
  Ny = argv.setdefault('Ny',1)
  dx = argv.setdefault('dx',0)
  dy = argv.setdefault('dy',0)
  
  frame = []
  frame.append([float(-Lx * Nx)/2,float(Ly * Ny)/2])
  frame.append([float(Lx * Nx)/2,float(Ly * Ny)/2])
  frame.append([float(Lx * Nx)/2,float(-Ly * Ny)/2])
  frame.append([float(-Lx * Nx)/2,float(-Ly * Ny)/2])
  frame_poly = Polygon(frame)
  
  #----------------------
  pbc = []
  pbc.append([0,0])
  pbc.append([Lx,0])
  pbc.append([-Lx,0])
  pbc.append([0,Ly])
  pbc.append([0,-Ly])
  pbc.append([Lx,Ly])
  pbc.append([-Lx,-Ly])
  pbc.append([-Lx,Ly])
  pbc.append([Lx,-Ly])

  area = Lx * Ly * porosity


  area2 = np.pi*smooth*smooth 
  r = smooth
   
  polygons = []
  for ppp in pbc:
   for kx in range(int(Nx)):
    for ky in range(int(Ny)):
     cx = -Lx*Nx*0.5 + (kx+0.5)*Lx + ppp[0]
     cy = -Ly*Ny*0.5 + (ky+0.5)*Ly + ppp[1]


     poly = get_smoothed_square(cx,cy,area,Na,{'smooth':smooth})
     polygons.append(poly)

  
  polys_cut = []
  for p in polygons:
   if Polygon(p).intersects(frame_poly):
    polys_cut.append(p)
    
  return frame,polys_cut
