import os,sys
import numpy as np
import pyclipper 
from pyclipper import *
import random
import math
from utils_disorder import *



def GenerateRandomPores(argv):

  delta_pore = argv.setdefault('pore_distance',0.2)

  if argv['shape'] == 'circle':
   Na_p = [24]

  if argv['shape'] == 'square':
   Na_p = [4]

  random_phi =  argv.setdefault('random_angle',False)

  
  frame = argv['frame']
  Lxp =  frame[0]
  Lyp =  frame[1]
  Nx =  argv.setdefault('Nx',4)
  Ny =  argv.setdefault('Ny',4)
  phi =  argv.setdefault('porosity',0.3)

  frame = []
  frame.append([float(-Lxp * Nx)/2,float(Lyp * Ny)/2])
  frame.append([float(Lxp * Nx)/2,float(Lyp * Ny)/2])
  frame.append([float(Lxp * Nx)/2,float(-Lyp * Ny)/2])
  frame.append([float(-Lxp * Nx)/2,float(-Lyp * Ny)/2])

  phi_vec = []
  if random_phi == 0 :
   phi_vec = [0.0]
  else :
   NPhi = 48
   deltaPhi = 360.0/float(NPhi);
   for k in range(NPhi) :
    phi_vec.append(k*deltaPhi)

  
  Lx = Lxp*Nx
  Ly = Lyp*Ny
  area_tot = Lx*Ly*phi
  area_tmp = 0.0;


  phi_min = phi 
  phi_max = phi
  area_max = Lxp * Lyp*phi_max
  area_min = Lxp * Lyp*phi_min
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
 
     
  
  #Periodic Vectors
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
  polygons_info = []
  #pyclipper.SCALING_FACTOR=1e3
  while (area_miss > 1e-4):
       
   collision = 1 
   nt = 0   
   while collision == 1 : 
 
    #This part helps ensuring to cover the given porosity
    if (area_miss > area_bounds[1]+1e-10):
       area = np.random.uniform(area_bounds[0],min(area_bounds[1],area_miss - area_bounds[0]))
    else :
      if (area_miss > 2.0 * area_bounds[0]):
       area = random.uniform(area_bounds[0],area_miss - area_bounds[0])
      else :
       area = area_miss;
    
    #Generate a random pore
    if len(Na_p) > 1:
     Na = Na_p[np.random.randint(0,len(Na_p)-1)];   
    else:
     Na = Na_p[0]

    
    if len(phi_vec) > 1:
      phi0 = phi_vec[np.random.randint(0,len(phi_vec)-1)]
    else:
      phi0 = phi_vec[0]

    r = math.sqrt(2.0*area/Na/math.sin(2.0 * math.pi/Na))  
    dphi = 2.0*math.pi/Na; 
    r2 = r*(1.0+delta_pore)
    x = np.random.uniform(Xmin,Xmax)
    y = np.random.uniform(Ymin,Ymax) 
     
    #check if it is in the frame
    poly_clip = []    
    for ka in range(Na+1):
     ph =  dphi/2 + (ka-1) * dphi + phi0*math.pi/180.0;
     px  = x + r * math.cos(ph)
     py  = y + r * math.sin(ph)
     poly_clip.append([px,py])
    c = pyclipper.Pyclipper()
    c.AddPath(scale_to_clipper(poly_clip), pyclipper.PT_CLIP,True) 
    c.AddPath(scale_to_clipper(frame),pyclipper.PT_SUBJECT,True) 
    solution = scale_from_clipper(c.Execute(pyclipper.CT_INTERSECTION,pyclipper.PFT_EVENODD,pyclipper.PFT_EVENODD))
    if not solution ==  [] : #The (extended) pore is inside the frame

     #consider an inflated pore to see if it intersects other pores
     r3 = r*(1+0.5)
     poly_clip = []
     for ka in range(Na):
      ph =  dphi/2 + (ka-1) * dphi + phi0*math.pi/180.0;
      px  = x + r3 * math.cos(ph)
      py  = y + r3 * math.sin(ph)
      poly_clip.append([px,py])

     #Check for collision             
     collision = 0
     if not len(polygons) == 0:
      for kp in range(len(polygons)):
       c = pyclipper.Pyclipper()
       c.AddPath(scale_to_clipper(poly_clip), pyclipper.PT_CLIP,True) 
       c.AddPath(scale_to_clipper(polygons[kp]),pyclipper.PT_SUBJECT,True) 
       solution = scale_from_clipper(c.Execute(pyclipper.CT_INTERSECTION,pyclipper.PFT_EVENODD,pyclipper.PFT_EVENODD))

       if not solution == []  :
        collision = 1
        break

     #If there is no collision with existing pores
     if collision == 0 : 
      #Add pore to polygon
      p = []
      for ka in range(Na):
       ph =  dphi/2 + (ka-1) * dphi + phi0*math.pi/180.0;
       px  = x + r * math.cos(ph)
       py  = y + r * math.sin(ph)
       p.append([px,py])
      polygons.append(p)
      polygons_info.append([x,y,r,Na,phi0])
      area_tmp +=area;
      area_miss = area_tot - area_tmp
      print('Coverage: ' + str(area_tmp/area_tot) + ' %')

      for kp in range(len(pbc)):
       p = []
       for ka in range(Na):
        ph =  dphi/2 + (ka-1) * dphi + phi0*math.pi/180.0;
        px  = x + r * math.cos(ph) + pbc[kp][0]
        py  = y + r * math.sin(ph) + pbc[kp][1]
        p.append([px,py])
       polygons.append(p)
       polygons_info.append([x+pbc[kp][0],y+pbc[kp][1],r,Na,phi0])

  #Fill up pores, add only the ones that actually are in the frame
  pores = []
  NP = len(polygons)
  polygons_final = []
  polygons_final_ext = []
  polygons_info_final = []
  configuration = []
  for k in range(NP) :
    #Create external pore
    info = polygons_info[k]
    x = info[0]
    y = info[1]
    r = info[2]
    Na = info[3]
    phi0 = info[4]

    dphi = 2.0*math.pi/Na; 
    poly_clip = []
    for ka in range(Na):
     ph =  dphi/2 + (ka-1) * 2.0*math.pi/Na + phi0*math.pi/180.0;
     px  = x + r * math.cos(ph)
     py  = y + r * math.sin(ph)
     poly_clip.append([px,py])

    c = pyclipper.Pyclipper()
    c.AddPath(scale_to_clipper(poly_clip), pyclipper.PT_CLIP,True) 
    c.AddPath(scale_to_clipper(frame),pyclipper.PT_SUBJECT,True) 
    solution = scale_from_clipper(c.Execute(pyclipper.CT_INTERSECTION,pyclipper.PFT_EVENODD,pyclipper.PFT_EVENODD))

    if not solution ==  [] :
     polygons_final.append(polygons[k])
     polygons_info_final.append(polygons_info[k])
     if x < Lx/2.0 and x>-Lx/2.0:
      if y < Ly/2.0 and y>-Ly/2.0:
       configuration.append([x,y])
  #polygons_info = polygons_info_final
  #polygons = polygons_final

  if argv.setdefault('save_configuration',False):
   np.array(configuration).dump(file('conf.dat','w+'))
  #return frame,polygons_final,polygons_info_final
  return frame,polygons_final

