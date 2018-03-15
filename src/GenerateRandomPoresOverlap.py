import os,sys
import numpy as np
import random
import math
from utils_disorder import *
from shapely.ops import cascaded_union
from shapely.geometry import Point
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon

def GenerateRandomPoresOverlap(argv):


  delta_pore = argv.setdefault('pore_distance',0.2)

  if argv['shape'] == 'circle':
   Na_p = [24]

  if argv['shape'] == 'square':
   Na_p = [4]

  Np = argv['Np']

  random_phi =  argv.setdefault('random_angle',False)


  frame = argv['frame']
  Lxp =  frame[0]
  Lyp =  frame[1]
  Nx =  argv.setdefault('Nx',4)
  Ny =  argv.setdefault('Ny',4)
  phi =  argv.setdefault('porosity',0.3)

  frame_tmp = []
  frame_tmp.append([float(-Lxp * Nx)/2,float(Lyp * Ny)/2])
  frame_tmp.append([float(Lxp * Nx)/2,float(Lyp * Ny)/2])
  frame_tmp.append([float(Lxp * Nx)/2,float(-Lyp * Ny)/2])
  frame_tmp.append([float(-Lxp * Nx)/2,float(-Lyp * Ny)/2])
  frame = Polygon(frame_tmp)

  
  Lx = Lxp*Nx
  Ly = Lyp*Ny
  area_tot = Lx*Ly

  
  area_tmp = 0.0;

 
  area_max = Lx*phi/float(Np)

  area_min = Lx*Ly*phi/float(Np)

  area_bounds = range(2);
  area_bounds[0] = area_min;
  area_bounds[1] = area_max;
     
  #f1 = 1.5
  area_tmp = 0.0;
  cx = []
  cy = []
  Na = []
  radious = []
  angles = []
  phis = []
  pores = [];
  nt_max = 3000
  area_miss = area_tot - area_tmp
 
  pbc = []
  pbc.append([Lx,0])
  pbc.append([-Lx,0])
  pbc.append([0,Ly])
  pbc.append([0,-Ly])
  pbc.append([Lx,Ly])
  pbc.append([-Lx,-Ly])
  pbc.append([-Lx,Ly])
  pbc.append([Lx,-Ly])
  #---------------------
  ext = 3.0/4.0
  Xmin = - Lx*ext
  Xmax =  Lx*ext
  Ymin = - Ly*ext  
  Ymax =  Ly*ext

  polygons = []
  #pyclipper.SCALING_FACTOR=1e3

  #while (area_miss > 1e-4):
  buf = 0.0
  nn = 0
  while nn < Np:
  
   #collision = 1 
   #while collision == 1 : 
 
    #This part helps ensuring to cover the given porosity
    #if (area_miss > area_bounds[1]+1e-10):
    #   area = np.random.uniform(area_bounds[0],min(area_bounds[1],area_miss - area_bounds[0]))
    #else :
    #  if (area_miss > 2.0 * area_bounds[0]):
    #   area = np.random.uniform(area_bounds[0],area_miss - area_bounds[0])
    #  else :
    #   area = area_miss;
  
    area = area_min  
    #Generate a random pore
    #if len(Na_p) > 1:
    # Na = Na_p[np.random.randint(0,len(Na_p)-1)];   
    #else:
    Na = Na_p[0]

    #Create polygons
    r = math.sqrt(2.0*area/Na/math.sin(2.0 * math.pi/Na))  
    dphi = 2.0*math.pi/Na; 
    r2 = r
    x = np.random.uniform(Xmin,Xmax)
    y = np.random.uniform(Ymin,Ymax) 
    poly_clip = []    
    for ka in range(Na+1):
     ph =  dphi/2 + (ka-1) * dphi
     px  = x + r * math.cos(ph)
     py  = y + r * math.sin(ph)
     poly_clip.append([px,py])
    p1 = Polygon(poly_clip)

    if p1.intersects(frame):
   
     nn += 1
     #Check for collision             
     collision = []
     if not len(polygons) == 0:
      for n_g,poly_group in enumerate(polygons):
       for n_p,poly in enumerate(poly_group):
        p2 = Polygon(poly)
        if p1.intersects(p2.buffer(buf)):
         collision.append([n_g,n_p])
     ##If there is no collision with existing pores
     if len(collision) == 0 :
      #polygons.append(poly_clip)
      poly_group = []
      poly_group.append(poly_clip) 
      area_tmp +=area;
      area_miss = area_tot - area_tmp
      for kp in range(len(pbc)):
       p2 = np.array(poly_clip).copy()
       p2[:,0] += pbc[kp][0]
       p2[:,1] += pbc[kp][1]
       poly_group.append(p2) 
      polygons.append(poly_group)
      print('Coverage: ' + str(area_tmp/area_tot) + ' %')
     else:
      if argv['overlap']:
       #Create the multipolygon------
       tmp = [p1]
       for p in collision:
        tmp.append(Polygon(polygons[p[0]][p[1]]).buffer(buf))
       m = MultiPolygon(tmp)
       total = cascaded_union(m)
       new_p = list(total.exterior.coords) #this is the contour
       #if hasattr(total,'interior'):
       #  print(total.interior)
       #-----------------------------

       #DELETE PORES---------------------------
       n = 0
       for p in collision:
        area_tmp -=Polygon(polygons[p[0]-n][0]).area
        del polygons[p[0]-n]
        n +=1
       #---------------------------------------

       poly_group = []
       poly_group.append(new_p) 
       area_tmp +=total.area;
       area_miss = area_tot - area_tmp
       #periodic ones
       for kp in range(len(pbc)):
        p2 = np.array(new_p).copy()
        p2[:,0] += pbc[kp][0]
        p2[:,1] += pbc[kp][1]
        poly_group.append(p2) 
       polygons.append(poly_group)
       print('Coverage (overlap): ' + str(area_tmp/area_tot) + ' %')

  #Create polys
  polys = []
  for poly_group in polygons:
   for poly in poly_group:
    polys.append(poly)
  #-----------------------
 
  #Fill up pores, add only the ones that actually are in the frame
  #pores = []
  NP = len(polys)
  polygons_final = []
  for k in range(NP) :
   if Polygon(polys[k]).intersects(frame):
     polygons_final.append(polys[k][0:-1])

  return frame_tmp,polygons_final

