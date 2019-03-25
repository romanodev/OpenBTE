import numpy as np
import math
from shapely.geometry import Polygon


def GenerateSquareLatticePores(argv):

  porosity = float(argv['porosity'])
  Lx = float(argv['lx'])
  Ly = float(argv['ly'])

  #Read shape
  shape = argv.setdefault('shape','square')
  if shape == 'circle':
     Na = 24; phi0= 0.0

  if shape == 'circle_refined':
     Na = 48; phi0= 0.0

  if shape == 'square':
     Na = 4; phi0 = 0.0
  if shape == 'triangle':
     Na = 3; phi0 = 0.0

  phi0 = argv.setdefault('angle',0.0)
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

  polygons = []
  for ppp in pbc:
   for kx in range(int(Nx)):
    for ky in range(int(Ny)):
     cx = -Lx*Nx/2.0 + kx*Lx+(0.5+dx)*Lx + ppp[0]
     cy = -Ly*Ny/2.0 + ky*Ly+(0.5+dy)*Ly + ppp[1]
     r = math.sqrt(2.0*area/Na/math.sin(2.0 * np.pi/Na))

     dphi = 2.0*math.pi/Na;
     p = []
     for ka in range(Na):
      ph =  dphi/2 + (ka-1) * dphi + phi0*np.pi/180.0
      px  = cx + r * math.cos(ph)
      py  = cy + r * math.sin(ph)
      p.append([px,py])
     polygons.append(p)

  
  polys_cut = []
  for p in polygons:
   if Polygon(p).intersects(frame_poly):
    polys_cut.append(p)
    
  return frame,polys_cut
