import os,sys
import numpy as np
import random
import math
#from utils_disorder import *
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


  Lxp =  argv['lx']
  Lyp =  argv['ly']
  #Nx =  argv.setdefault('Nx',4)
  #Ny =  argv.setdefault('Ny',4)
  Nx = 1
  Ny = 1
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
  #ext = 3.0/4.0
  ext = 0.95
  Xmin = - Lx/2*ext
  Xmax =  Lx/2*ext
  Ymin = - Ly/2*ext  
  Ymax =  Ly/2*ext

  polygons = []
  final = []
  #pyclipper.SCALING_FACTOR=1e3

  #while (area_miss > 1e-4):
  buf = 0.0
  nn = 0
  confs = []
  #while nn < Np:

  area = area_min  
  Na = Na_p[0]


  if argv.setdefault('manual',False):
   centers = argv['centers']
  else:
   centers = []
   for nn in range(Np):
    x = np.random.uniform(0,1)
    y = np.random.uniform(0,1) 
    centers.append([x,y])

  for center in centers:
    x = Lx*(center[0]-0.5)
    y = Ly*(center[1]-0.5)
    #Create polygons
    r = math.sqrt(2.0*area/Na/math.sin(2.0 * math.pi/Na))  
    dphi = 2.0*math.pi/Na; 
    r2 = r
    poly_clip = []    
    for ka in range(Na):
     ph =  dphi/2 + (ka-1) * dphi
     px  = x + r * math.cos(ph)
     py  = y + r * math.sin(ph)
     poly_clip.append([px,py])
    p1 = Polygon(poly_clip)

    if p1.intersects(frame):
     nn += 1

     #Write configuration-------------
     for kp in range(len(pbc)):
      cx = x + pbc[kp][0]
      cy = y + pbc[kp][1]
      if cx < Lx/2.0 and cx > -Lx/2.0:
       if cy < Ly/2.0 and cy > -Ly/2.0:
        confs.append([cx,cy])
      #---------------------------------  

     #Check for collision before repeating it             
     collision = []
     if not len(polygons) == 0:
      for n_g,poly_group in enumerate(polygons):
       for n_p,poly in enumerate(poly_group):
        p2 = Polygon(poly)
        if p1.intersects(p2.buffer(buf)):
         collision.append([n_g,n_p])


     if len(collision) == 0 :

      #Add the pore and its repeatitions--------------------------
      poly_group = []
      area_tmp +=area;
      area_miss = area_tot - area_tmp
      for kp in range(len(pbc)):
       p2 = np.array(poly_clip).copy()
       p2[:,0] += pbc[kp][0]
       p2[:,1] += pbc[kp][1]
       p = Polygon(p2)
       if p.intersects(frame):
        poly_group.append(p2)
      polygons.append(poly_group)
      print('Coverage: ' + str(area_tmp/area_tot) + ' %')
      #------------------------------------------------------------

     else:

      if argv['overlap']:
       #Create the multipolygon------
       tmp = [p1]
       print(collision)
       for p in collision:
        tmp.append(Polygon(polygons[p[0]][p[1]]).buffer(buf))
       m = MultiPolygon(tmp)
       total = cascaded_union(m)
       new_p = list(total.exterior.coords) #this is the contour

       #DELETE PORES---------------------------
       n = 0
       gg = []
       for p in collision:
        if not p[0] in gg:
         area_tmp -=Polygon(polygons[p[0]-n][0]).area
         gg.append(p[0])
         del polygons[p[0]-n]
         n +=1
       #---------------------------------------

       poly_group = []
       area_tmp +=total.area;
       area_miss = area_tot - area_tmp
       for kp in range(len(pbc)):
        p2 = np.array(new_p).copy()
        p2[:,0] += pbc[kp][0]
        p2[:,1] += pbc[kp][1]
        p = Polygon(p2)
        if p.intersects(frame):
         if kp == 0 or (kp > 0 and not p.intersects(Polygon(new_p))):
          poly_group.append(p2)
       polygons.append(poly_group)
       print('Coverage (overlap): ' + str(area_tmp/area_tot) + ' %')

  #Create polys
  polys = []
  for poly_group in polygons:
   for poly in poly_group:
    polys.append(poly)
  #-----------------------
 
 
  if argv.setdefault('save_configuration',False):
   np.array(confs).dump(file('conf.dat','w+'))

  
  #return frame_tmp,polygons_final
  return frame_tmp,polys

