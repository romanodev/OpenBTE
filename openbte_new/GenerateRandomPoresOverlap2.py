from __future__ import print_function
import os,sys
import numpy as np
import random
import math
#from utils_disorder import *
from shapely.ops import cascaded_union
from shapely.geometry import Point
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon

random.seed(3)




def consolidate(polys,argv):

  
  d_min =  argv['step']/2.0
 
  n = len(polys)
  conn = {}
  inflate = np.zeros(n)
  for n1 in range(n):   
   p1 = polys[n1]
   for n2 in range(n1+1,n):   
    p2 = polys[n2]
    if p1.distance(p2) < d_min :
     inflate[n1] = d_min/2.0
     inflate[n2] = d_min/2.0
     mn =  min([n1,n2])
     Mn =  max([n1,n2])
     conn.setdefault(mn,[]).append(Mn)

  #Create new pores-----
  new_p = []
  for n,poly in enumerate(polys):   
   poly = poly.buffer(inflate[n], resolution=1,mitre_limit=20.0,join_style=2)
   new_p.append(poly)


  new_p2 = []
  #lone pores-----
  for n in np.where(inflate == 0)[0]:
  #for n in range(len(new_p)):
   new_p2.append(new_p[n])
 

  #merge pores-----
  n_coll = 0
  for n in conn.keys():
   n_coll +=1
   tmp = [new_p[n]]
   for p in conn[n]:
    tmp.append(polys[p])
   m = MultiPolygon(tmp)
   new_p2.append(cascaded_union(m))

  #-----------------------
 
  return new_p2,n_coll







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
  Nx = 1
  Ny = 1
  phi_mean =  argv.setdefault('porosity',0.3)
  spread =  argv.setdefault('spread',0.0)
  


  frame_tmp = []
  frame_tmp.append([float(-Lxp * Nx)/2,float(Lyp * Ny)/2])
  frame_tmp.append([float(Lxp * Nx)/2,float(Lyp * Ny)/2])
  frame_tmp.append([float(Lxp * Nx)/2,float(-Lyp * Ny)/2])
  frame_tmp.append([float(-Lxp * Nx)/2,float(-Lyp * Ny)/2])
  frame = Polygon(frame_tmp)

  
  Lx = Lxp*Nx
  Ly = Lyp*Ny
  area_tot = Lx*Ly

  
  #area_tmp = 0.0;

 
  #area_max = Lx*phi/float(Np)

  
  #area_min = Lx*Ly*phi/float(Np)

  #area_bounds = range(2);
  #area_bounds[0] = area_min;
  #area_bounds[1] = area_max;
     
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
  area_miss = area_tot 
 
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

  final = []
  #pyclipper.SCALING_FACTOR=1e3

  #while (area_miss > 1e-4):
  buf = 0.0
  nn = 0
  confs = []
  #while nn < Np:

  #area = area_min  
  Na = Na_p[0]


  if argv.setdefault('manual',False):
   centers = argv['centers']
  else:
   centers = []
   for nn in range(Np):
    x = np.random.uniform(0,1)
    y = np.random.uniform(0,1) 
    centers.append([x,y])

  #Write configuration-------------
  #for kp in range(len(pbc)):
  # cx = x + pbc[kp][0]
  # cy = y + pbc[kp][1]
  # if cx < Lx/2.0 and cx > -Lx/2.0:
  #  if cy < Ly/2.0 and cy > -Ly/2.0:
  #   confs.append([cx,cy])
  #---------------------------------

  thin = Polygon(frame_tmp).buffer(0.01).difference(frame)
  d_min =  argv['step']/2.0
  #Create first pores----
  polygons = []
  area_tmp = 0.0
  for nc,center in enumerate(centers):
   phi = random.uniform(phi_mean - spread, phi_mean + spread)
   area = Lx*Ly*phi/float(Np)

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
  #----------------------

   #Buffer------------------------------------------------------------
   inflate = False
   if p1.distance(thin) < d_min and p1.distance(thin) > 0.0:
    inflate = True
      
   collision = []
   max_d = 0
   if not len(polygons) == 0:
    for n_g,poly_group in enumerate(polygons):
     for n_p,poly in enumerate(poly_group):
      p2 = Polygon(poly)
      if p1.intersects(p2):
       collision.append([n_g,n_p])
      else:
       if p1.distance(p2) < d_min :
        inflate = True
        collision.append([n_g,n_p])

   if inflate:
    p1 = p1.buffer(d_min, resolution=1,mitre_limit=20.0,join_style=2)
    #-------------------------------------------------------------------

   if len(collision) == 0 :
   
     tmp = np.array(list(p1.exterior.coords))
     poly_group = []
     for kp in range(len(pbc)):
      p2 = tmp.copy()
      p2[:,0] += pbc[kp][0]
      p2[:,1] += pbc[kp][1]
      p = Polygon(p2)
      poly_group.append(p2)
     polygons.append(poly_group)
     #------------------------------------------------------------
   else:
     if argv['overlap']:
      #Create the multipolygon------
      tmp = [p1]
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
      for kp in range(len(pbc)):
       p2 = np.array(new_p).copy()
       p2[:,0] += pbc[kp][0]
       p2[:,1] += pbc[kp][1]
       p = Polygon(p2)
       if kp == 0 or (kp > 0 and not p.intersects(Polygon(new_p))):
        poly_group.append(p2)
      polygons.append(poly_group)




  #flattening poligons----
  polys = []
  for poly_group in polygons:
   for poly in poly_group:
    p = Polygon(poly)
    polys.append(p)

  n_coll = 1
  while n_coll > 0:
   polys,n_coll = consolidate(polys,argv)


  #Create polys
  polys_cut = []
  for p in polys:
   if p.intersects(frame):
    polys_cut.append(p.exterior.coords)

   #np.array(confs).dump(file('conf.dat','w+'))

  
  #return frame_tmp,polygons_final
  return frame_tmp,polys_cut

