import numpy as np
import math

def GenerateHexagonalLatticePores(argv):
    
  Lx = float(argv['lx'])
  Ly = float(argv['ly'])
  porosity = float(argv['porosity'])
   
  #Read shape
  shape = argv['shape']
  if shape == 'circle': 
     Na = 24; phi0= 360.0/48.0
  if shape == 'square': 
     Na = 4; phi0 = 0.0
  if shape == 'triangle':  
     Na = 3; phi0 = 0.0


  #Read Number of pores
  Nx = argv.setdefault('Nx',1)
  Ny = argv.setdefault('Ny',1)
  #----------------------

 
  area = Lx * Ly * porosity

  polygons = []
  for kx in range(int(Nx)):
   for ky in range(int(Ny)):
     cx = -Lx*Nx/2.0 + kx*Lx+0.5*Lx
     cy = -Ly*Ny/2.0 + ky*Ly+0.5*Ly
     r = math.sqrt(2.0*area/Na/math.sin(2.0 * np.pi/Na))   
       
     dphi = 2.0*math.pi/Na;
     p = []    
     for ka in range(Na):
      ph =  dphi/2 + (ka-1) * dphi + phi0*np.pi/180.0
      px  = cx + r * math.cos(ph)
      py  = cy + r * math.sin(ph)
      p.append([px,py])
     polygons.append(p)

  frame = []
  frame.append([float(-Lx * Nx)/2,float(Ly * Ny)/2])
  frame.append([float(Lx * Nx)/2,float(Ly * Ny)/2])
  frame.append([float(Lx * Nx)/2,float(-Ly * Ny)/2])
  frame.append([float(-Lx * Nx)/2,float(-Ly * Ny)/2])
 
  return frame,polygons
  

