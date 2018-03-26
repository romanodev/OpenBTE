import numpy as np
import math
from shapely.geometry import Point
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon

def GenerateCustomPores(argv):
    
  Lx = float(argv['frame'][0])
  Ly = float(argv['frame'][1])
  frame_tmp = []
  frame_tmp.append([float(-Lx)/2,float(Ly)/2])
  frame_tmp.append([float(Lx)/2,float(Ly)/2])
  frame_tmp.append([float(Lx)/2,float(-Ly)/2])
  frame_tmp.append([float(-Lx)/2,float(-Ly)/2])
  frame = Polygon(frame_tmp)

  #Periodicity------------
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
  #---------------------


  polygons = argv['polygons']

  final = []
  for poly in polygons:
    for kp in range(len(pbc)):
     tmp = []
     for p in poly:
      cx = p[0] + pbc[kp][0]
      cy = p[1] + pbc[kp][1]
      tmp.append([cx,cy])
     p1 = Polygon(tmp)
     if p1.intersects(frame): 
      final.append(tmp)

  return np.array(frame_tmp),np.array(final)
  

