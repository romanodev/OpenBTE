from __future__ import print_function
import os,sys
import numpy as np
import random
import math
import copy
from shapely.ops import cascaded_union
from shapely.geometry import Point
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
from shapely.affinity import translate
import networkx as nx
import matplotlib.pylab as plt
from .special_shapes import *

r''' (superseded by adjust)
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

   poly = p.buffer(inflate, resolution=1,mitre_limit=20.0,join_style=2)
   new_p.append(poly)


  return new_p
'''


def adjust_position(x,y,r,Na,Lx,Ly,d_min):


 poly_clip = []
 dphi = 2.0*math.pi/Na;
 for ka in range(Na):
  ph =  dphi/2 + (ka-1) * dphi
  px  = x + r * math.cos(ph)
  py  = y + r * math.sin(ph)
  poly_clip.append([px,py])
 p = Polygon(poly_clip)


 delta = 1e-3

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


def make_polygon(x,y,A,**options):

   Na = options['Na']
   dphi = 2.0*math.pi/Na;
   r = math.sqrt(2.0*A/Na/math.sin(2.0 * math.pi/Na))
   poly_clip = []
   for ka in range(Na):
     ph =  dphi/2 + (ka-1) * dphi
     px  = x + r * math.cos(ph) 
     py  = y + r * math.sin(ph) 
     poly_clip.append([px,py])

   return poly_clip  




def GenerateRandomPoresOverlap(argv):

  d_min =  argv['step']/20.0
  delta_pore = argv.setdefault('pore_distance',0.2)

  #argv.setdefault('load_configuration',False)
  #argv.setdefault('save_configuration',False)
  argv.setdefault('shape','square')


  if argv['shape'] == 'smoothed_square':
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
    


  Np = argv['Np']

  random_phi =  argv.setdefault('random_angle',True)

  #Lxp =  argv['lx']*argv.setdefault('repeat_x',1)
  #Lyp =  argv['ly']*argv.setdefault('repeat_y',1)
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

  buf = 0.0
  nn = 0
  confs = []

  #Na = Na_p[0]

  if argv.setdefault('load_configuration',False):
   centers = np.load(argv.setdefault('configuration_file','conf.dat'),allow_pickle=True)
  elif argv.setdefault('manual',False):
   centers = argv['centers']
  else:
   centers = []
   for nn in range(Np):
    x = np.random.uniform(0,1)
    y = np.random.uniform(0,1)
    centers.append([x,y])

  #dphi = 2.0*math.pi/Na;
  #Fill polys-----
  polys = []

  tt= []
  dmin=1e-1
  for center in centers:
   x = Lx*(center[0]-0.5)
   y = Ly*(center[1]-0.5)
   phi = np.random.uniform(phi_mean - spread, phi_mean + spread)
   area = Lx*Ly*phi/float(Np)


   poly_clip = make_pore(x,y,area,**options)

   #r = math.sqrt(2.0*area/Na/math.sin(2.0 * math.pi/Na))
   #poly_clip = []
   #for ka in range(Na):
   #  ph =  dphi/2 + (ka-1) * dphi
   #  px  = x + r * math.cos(ph) 
   #  py  = y + r * math.sin(ph) 
   #  poly_clip.append([px,py])

   p = Polygon(poly_clip)
   a = p.intersection(frame).area == p.area
   dx = 0
   dy = 0

   #move pores if they are too close to the boundaries
   if argv.setdefault('adjust',True):
    p1 = copy.deepcopy(translate(p,xoff=dmin))
    b = p1.intersection(frame).area == p1.area
    if a != b:
     dx = -dmin
    p1 = copy.deepcopy(translate(p,xoff=-dmin))
    b = p1.intersection(frame).area == p1.area
    if a != b:
     dx = +dmin
    p1 = copy.deepcopy(translate(p,yoff=dmin))
    b = p1.intersection(frame).area == p1.area
    if a != b:
     dy = -dmin
    p1 = copy.deepcopy(translate(p,yoff=-dmin))
    b = p1.intersection(frame).area == p1.area
    if a != b:
     dy = +dmin

   x += dx
   y += dy
   tt.append([x,y])

   for kp in range(len(pbc)):
    poly_clip = make_pore(x + pbc[kp][0],y+pbc[kp][1],area,**options)
    polys.append(Polygon(poly_clip))


  #Consolidate pores that are very close---

  #self.frame_consolidate(argv,polys,frame_tmp):
      

  if argv.setdefault('consolidate',True) :
   n_coll = 1
  else:
   n_coll = 0  
  while n_coll > 0:
   polys,n_coll = consolidate(polys,argv)
  #------------------------------


  #compute distance between pores---
  mind = 1e9
  n = len(polys)
  for p1 in range(n):
   for p2 in range(p1+1,n):
     poly1 = polys[p1]
     poly2 = polys[p2]
     mind = min([mind,poly1.distance(poly2)])
  #--------------------------------

  area = 0.0
  eps = argv['step']/2.0
  polys_cut = []
  for p in polys:

   if p.intersects(frame):
    
    new = list(p.exterior.coords)[:-1]   
    polys_cut.append(new)
    area += p.area


  if argv.setdefault('save_configuration',False) and not argv['load_configuration']:
   np.array(centers).dump(argv.setdefault('configuration_file','conf.dat'))

  return frame_tmp,polys_cut,np.array(centers),mind

