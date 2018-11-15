import numpy as np
import math

def GenerateHexagonalLatticePores(argv):

  a = float(argv.setdefault('a',1))
  porosity = float(argv['porosity'])
  Lx = 2.0 * np.sin(np.pi/6.0)*a
  Ly = 2.0 * np.cos(np.pi/6.0)*a
  phi0 = argv.setdefault('angle',0.0)

  #Read shape
  shape = argv['shape']
  if shape == 'circle':
     Na = 24; #phi0= 360.0/48.0
  if shape == 'square':
     Na = 4; #phi0 = 0.0
  if shape == 'triangle':
     Na = 3; #phi0 = 0.0


  area = 0.5*Lx * Ly * porosity
  r = math.sqrt(2.0*area/Na/math.sin(2.0 * np.pi/Na))

  cc = [[-Lx/2.0,-Ly/2.0],[0.0,0.0],[-Lx/2.0,Ly/2.0],[Lx/2.0,Ly/2.0],[Lx/2.0,-Ly/2.0]]


  polygons = []
  for c in cc:
   dphi = 2.0*math.pi/Na;
   p = []
   for ka in range(Na):
    ph =  dphi/2 + (ka-1) * dphi + phi0*np.pi/180.0
    px  = c[0] + r * math.cos(ph)
    py  = c[1] + r * math.sin(ph)
    p.append([px,py])
   polygons.append(p)

  frame = []
  frame.append([float(-Lx)/2,float(Ly)/2])
  frame.append([float(Lx)/2,float(Ly)/2])
  frame.append([float(Lx)/2,float(-Ly)/2])
  frame.append([float(-Lx)/2,float(-Ly)/2])


  return frame,polygons
