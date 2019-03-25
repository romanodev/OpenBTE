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
import matplotlib
if not matplotlib.get_backend() == 'Qt5Agg': matplotlib.use('Qt5Agg')

import networkx as nx
import matplotlib.pylab as plt

#random.seed(3)

def frame_consolidate(argv,polys,frame_tmp):

  d_min =  argv['step']/2.0
  #d_min =  0.0

  frame = Polygon(frame_tmp)
  thin = Polygon(frame_tmp).buffer(1e-4).difference(frame)

  new_p = []
  for nn,p in enumerate(polys):

   inflate = 0
   dd = p.distance(thin)
   if dd < d_min:
    inflate = d_min
    #inflate[nn] = d_min

  #new_p = []
  #for n,poly in enumerate(polys):
   poly = p.buffer(inflate, resolution=1,mitre_limit=20.0,join_style=2)
   new_p.append(poly)


   #new_p2 = []
   #lone pores-----
   #for n in np.where(collision == 0)[0]:
   # new_p2.append(new_p[n])


   #merge pores-----
   #n_coll = 0
   #for n in conn.keys():
   # n_coll +=1
   # tmp = [new_p[n]]
   # for p in conn[n]:
   #  tmp.append(new_p[p])
   # m = MultiPolygon(tmp)
   # new_p2.append(cascaded_union(m))

  #-----------------------
  return new_p








  # coords = p.exterior.coords
  # for n in range(len(coords)-1):
  #  plt.plot([coords[n][0],coords[n+1][0]],[coords[n][1],coords[n+1][1]],color='black')

  # mm = np.mean(coords,axis=0)

  # plt.text(mm[0],mm[1],str(nn))


  #for n in range(4):
  #  n1 = n%4
  #  n2 = (n+1)%4
  #  plt.plot([frame_tmp[n1][0],frame_tmp[n2][0]],[frame_tmp[n1][1],frame_tmp[n2][1]],color='red')


  #plt.axis('equal')
  #plt.show()

  #n = len(polys)
  #newp = []
  #for n1 in range(n):
  # p1 = polys[n1]
  # dd = p1.distance(thin)
  # if dd ==0:
  #  print(n1)
  # if dd < d_min and dd > 0:
  #  print(n1)


  #return newp

def adjust_position(x,y,r,Na,Lx,Ly,d_min):


 poly_clip = []
 dphi = 2.0*math.pi/Na;
 for ka in range(Na):
  ph =  dphi/2 + (ka-1) * dphi
  px  = x + r * math.cos(ph)
  py  = y + r * math.sin(ph)
  poly_clip.append([px,py])
 p = Polygon(poly_clip)


 delta = 1e-4

 tmp = []
 tmp.append([-Lx/2-delta,Ly/2])
 tmp.append([-Lx/2,Ly/2])
 tmp.append([-Lx/2,-Ly/2])
 tmp.append([-Lx/2-delta,-Ly/2])
 L = Polygon(tmp)

 tmp = []
 tmp.append([Lx/2,Ly/2])
 tmp.append([Lx/2+delta,Ly/2])
 tmp.append([Lx/2+delta,-Ly/2])
 tmp.append([Lx/2,-Ly/2])
 R = Polygon(tmp)


 tmp = []
 tmp.append([-Lx/2,Ly/2])
 tmp.append([-Lx/2,Ly/2+delta])
 tmp.append([Lx/2,Ly/2+delta])
 tmp.append([Lx/2,Ly/2])
 U = Polygon(tmp)


 tmp = []
 tmp.append([-Lx/2,-Ly/2])
 tmp.append([-Lx/2,-Ly/2-delta])
 tmp.append([Lx/2,-Ly/2-delta])
 tmp.append([Lx/2,-Ly/2])
 B = Polygon(tmp)


 dx = 0
 dy = 0

 d = p.distance(U)
 if d < d_min and d>0:
  dy += 2*d_min

 d = p.distance(B)
 if d < d_min and d>0:
  dy -= 2*d_min

 d = p.distance(L)
 if d < d_min and d>0:
  dx -= 2*d_min

 d = p.distance(R)
 if d < d_min and d>0:
  dx += 2*d_min


 return dx,dy



def consolidate(polys,argv):

  d_min =  argv['step']/20.0
  #d_min = 0

  #d_min = 0
  n = len(polys)
  conn = {}
  inflate = np.zeros(n)
  collision = np.zeros(n)
  for n1 in range(n):
   p1 = polys[n1]

   for n2 in range(n1+1,n):
    mn =  min([n1,n2])
    Mn =  max([n1,n2])
    p2 = polys[n2]
    if p1.intersects(p2):
     conn.setdefault(mn,[]).append(Mn)
     collision[n1] = 1
     collision[n2] = 1

    dd = p1.distance(p2)
    if dd < d_min and dd > 0:
      inflate[n1] = max([inflate[n1],d_min/2.0])
      inflate[n2] = max([inflate[n2],d_min/2.0])
      conn.setdefault(mn,[]).append(Mn)
      collision[n1] = 1
      collision[n2] = 1


  #Create new pores-----
  new_p = []
  for n,poly in enumerate(polys):
   poly = poly.buffer(inflate[n], resolution=1,mitre_limit=20.0,join_style=2)
   new_p.append(poly)


  new_p2 = []
  #lone pores-----
  for n in np.where(collision == 0)[0]:
   new_p2.append(new_p[n])


  #Cluster conn-------
  G = nx.Graph()
  for n in conn.keys():
   G.add_node(n)
   for m in conn[n]:
    G.add_edge(n,m)
  components = list(nx.connected_components(G))
  #-----------------------

  n_coll = 0
  for c in components:
   n_coll +=1
   tmp = []
   for p in c:
    tmp.append(new_p[p])
   m = MultiPolygon(tmp)
   p = cascaded_union(m)
   new_p2.append(p)


  new_p3 = []
  for p in new_p2:
   pp = p.buffer(d_min, 1, join_style=2).buffer(-d_min, 1, join_style=2)
   new_p3.append(pp)

  #-----------------------
  return new_p3,n_coll




def GenerateRandomPoresOverlap(argv):


  d_min =  argv['step']/20.0
  delta_pore = argv.setdefault('pore_distance',0.2)

  argv.setdefault('shape','square')
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


  Lx = Lxp
  Ly = Lyp
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

  if argv.setdefault('load_configuration',False):
   centers = np.load('conf.dat')
  elif argv.setdefault('manual',False):
   centers = argv['centers'] 
  else:
   centers = []
   for nn in range(Np):
    x = np.random.uniform(0,1)
    y = np.random.uniform(0,1)
    centers.append([x,y])


  dphi = 2.0*math.pi/Na;
  #Fill polys-----
  polys = []

  for center in centers:
   x = Lx*(center[0]-0.5)
   y = Ly*(center[1]-0.5)
   phi = np.random.uniform(phi_mean - spread, phi_mean + spread)
   area = Lx*Ly*phi/float(Np)
   r = math.sqrt(2.0*area/Na/math.sin(2.0 * math.pi/Na))

   (dx,dy) = adjust_position(x,y,r,Na,Lx,Ly,d_min)

   x +=dx
   y +=dy

   for kp in range(len(pbc)):
    poly_clip = []
    for ka in range(Na):
     ph =  dphi/2 + (ka-1) * dphi
     px  = x + r * math.cos(ph) + pbc[kp][0]
     py  = y + r * math.sin(ph) + pbc[kp][1]
     poly_clip.append([px,py])

    polys.append(Polygon(poly_clip))




  n_coll = 1
  while n_coll > 0:
   polys,n_coll = consolidate(polys,argv)
  #polys,n_coll = consolidate(polys,argv)
  # polys,n_coll = consolidate(polys,argv)
  #polys,n_coll = consolidate(polys,argv)
  #polys,n_coll = consolidate(polys,argv)
  # print(n_coll)


  #n_coll = 1
  #while n_coll > 0:
  # polys,n_coll = consolidate(polys,argv)
  #Create polys
  area = 0.0
  eps = argv['step']/2.0
  polys_cut = []
  for p in polys:
   if p.intersects(frame):
    polys_cut.append(list(p.exterior.coords)[:-1])
    area += p.area

  if argv['save_configuration'] and not argv['load_configuration']:
   #np.array(centers).dump(open('conf.dat','w+'))
   np.array(centers).dump('conf.dat')



  #return frame_tmp,polygons_final

  #scaling to 0-1
  #polygons = []
  #for poly_tmp in polys_cut:
  # poly = []
  # for p in poly_tmp:
  #   poly.append([(p[0]-Lx/2)/Lx,(p[1]-Ly/2)/Ly])  
  # polygons.append(poly) 


  return frame_tmp,polys_cut
