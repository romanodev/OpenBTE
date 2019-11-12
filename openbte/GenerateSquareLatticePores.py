import numpy as np
import math
from shapely.geometry import Polygon

from .special_shapes import *




def GenerateSquareLatticePores(argv):

  argv.setdefault('shape','square')

  if argv['shape'] == 'custom':
     make_pore = argv['shape_function']
     options = argv.setdefault('shape_options',{})

  elif argv['shape'] == 'smoothed_square':
    make_pore = get_smoothed_square  
    options = {'smooth':argv.setdefault('smooth',3),'Na':6}
  else:  
    make_pore = make_polygon

    if argv['shape'] == 'circle':
        options = {'Na': 24}

    if argv['shape'] == 'square':
        options = {'Na': 4}

    if argv['shape'] == 'triangle':
        options = {'Na': 3}


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

  base = argv.setdefault('base',[[0.5,0.5]]) 
  area = Lx * Ly * porosity/len(base)

  options.update({'area':area})
  polygons = []
  for b in base:
   for ppp in pbc:
    for kx in range(int(Nx)):
     for ky in range(int(Ny)):
      cx = -Lx*Nx*0.5 + (kx+b[0])*Lx + ppp[0]
      cy = -Ly*Ny*0.5 + (ky+b[0])*Ly + ppp[1]
    
      tmp = make_pore(**options)
      poly = [[t[0]+cx,t[1]+cy]   for t in tmp]

      polygons.append(poly)

  
  polys_cut = []
  for p in polygons:
   if Polygon(p).intersects(frame_poly):
    polys_cut.append(p)
    
  return frame,polys_cut
