from __future__ import print_function
import xml.etree.ElementTree as ET
import os,sys
import numpy as np
import random
import math
from matplotlib.pylab import *
import pyclipper
from pyclipper import *
import subprocess

def line_exists(l,lines):

 for n,line in enumerate(lines) :
   if (line[1] == l[0] and line[2] == l[1]) or (line[1] == l[1] and line[2] == l[0]) :
     return line
 return 0

def point_exists(p,points):

 for n,point in enumerate(points) :
   if abs(point[0] - p[0]) < 1e-4 and abs(point[1] - p[1])<1e-4 :
     return n

 return -1


def get_line_from_points(p1,p2,all_lines,points):

 #Get indexes
 for t,p in enumerate(points):
  if p == p1 :
   t1 = t 
  if p == p2 :
   t2 = t 


 for t,l in enumerate(all_lines):
  if (t1 == l[1] and t2 == l[2]) :  
   line = [l[0],t1,t2]
   break
  if (t2 == l[1] and t1 == l[2])  :
   line = [l[0],t2,t1]
   break


 return line


def rev_rotate(l,n):
    return l[-n:] + l[:-n]


def rotate(l,n):
    return l[n:] + l[:n]


def GetSampleList(output_dir) :
   tmp =  glob.glob(output_dir + '/*.json')
   index = 0
   if len(tmp) > 0:
    for n in range(len(tmp)) :
     tmp2 = tmp[n].split('/')
     tmp3 = tmp2[-1].split('.')
     tmp4 = tmp3[0].split('_') 
     index = max([int(tmp4[1]),index])
    index +=1

   return index

 

def plot_region(region):

   for n in range(len(region)) :
    n1 = n
    n2 = (n+1)%len(region)
    plot([region[n1][0],region[n2][0]],[region[n1][1],region[n2][1]],color='black')
   xlim([-3,3])
   ylim([-3,3])

def FindPeriodic(control,lines,points):
 
 
 p1c = points[control[1]]
 p2c = points[control[2]]
 for n in range(len(lines)):
   p1 = points[lines[n][1]]
   p2 = points[lines[n][2]]
   #Positive  
   pd1 = [p1c[0]-p1[0],p1c[1]-p1[1]]  
   pd2 = [p2c[0]-p2[0],p2c[1]-p2[1]]  
   if abs(pd1[0] - pd2[0]) < 1e-6 and abs(pd1[1] - pd2[1]) <1e-6 :
     return lines[n][0]
   
   #Negative  
   pd1 = [p1c[0]-p2[0],p1c[1]-p2[1]]   
   pd2 = [p2c[0]-p1[0],p2c[1]-p1[1]]   
   if abs(pd1[0] - pd2[0])<1e-6 and abs(pd1[1] - pd2[1])<1e-6 :
     return -lines[n][0]


 #If no periodic line has been found
 print("NO PERIODIC LINES HAS BEEN FOUND")
 quit()
 
def LineOrdering(lines) :

  new_lines = []
  new_lines.append(lines[0])
  check = np.ones(len(lines))
  check[0] = 0
  n_check = 0 
  while len(lines) > len(new_lines) :
   n_check +=1 
   line = new_lines[len( new_lines)-1]
   for k in range(len(lines)):
    if check[k] == 1:
     if lines[k][1] == line[2]  :
      check[k] = 0
      new_lines.append([lines[k][0],lines[k][1],lines[k][2]])
      break;
     if lines[k][2] == line[2]  :
      check[k] = 0
      new_lines.append([-lines[k][0],lines[k][2],lines[k][1]])
      break;
   if n_check > len(lines) :
    print('CANNOT FIND A ORDERED LOOP')
    print(lines)
    quit()
  return new_lines



def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def mesh(polygons,frame,argv):

  mesh_ext = argv['step']
  include_pores = argv.setdefault('include_pores',False)
  #Lx = argv['Lx']*argv['Nx']
  #Ly = argv['Ly']*argv['Ny']
  #if argv['submodel'] == 'staggered':
  # Lx *=sqrt(2)
  # Ly *=sqrt(2)

  Lx = -frame[0][0]*2.0
  Ly = frame[0][1]*2.0
  Lz = argv['lz']
  

  store = open('mesh.geo', 'w+')
  #CREATE BULK SURFACE------------------------------------
  c = pyclipper.Pyclipper()
  c.AddPath(scale_to_clipper(frame), pyclipper.PT_SUBJECT,True) 
  c.AddPaths(scale_to_clipper(polygons),pyclipper.PT_CLIP,True) 
  solution = scale_from_clipper(c.Execute(pyclipper.CT_DIFFERENCE,pyclipper.PFT_EVENODD,pyclipper.PFT_EVENODD))

  ng = len(solution)
  points = []
  lines = []
  lines.append([0,0,0])
  loop = 0
  ss = 0
  ll = 1
  line_contour = []
  pore_regions = []

  Z = [-Lz/2,Lz/2]

  C1 = []
  C2 = []
  C3 = []
  C4 = []
  delta = 1e-4
  pore_surfaces = []
  upper_loops = []
  lower_loops = []
  lateral_surfaces = []
  lateral_loops = []
  horizontal_surfaces = []
  for g,region in enumerate(solution) :

   #np = len(region)
   for kk,z in enumerate(Z):
    p_start = len(points)
    l_start = ll
  
    region_points = []
    for p,point in enumerate(region) :
     new_point = [round(point[0],4),round(point[1],4),round(z,4)]
     if not new_point in points:
      points.append(new_point)
      region_points.append(new_point)
      store.write( 'Point('+str(len(points)-1) +') = {' + str(new_point[0]) +','+ str(new_point[1])+',' + str(z) + ','+ str(mesh_ext) +'};\n')

    np = len(region_points)
      #Write Points
    #for p,point in enumerate(region) :
    # points.append(point)
     
    #Write Horizonal Lines 
    l_tmp = []
    for p,point in enumerate(region_points) :
     p1 = p_start + p
     p2 = p_start + (p+1)%len(region_points)
     lines.append([ll,p1,p2])
     store.write( 'Line('+str(ll) +') = {' + str(p1) +','+ str(p2)+'};\n')
     l_tmp.append(ll)
     ss += 1
     ll += 1
     
    #create horizontal loops----------
    strc = 'Line Loop(' + str(loop)+ ') = {'
    for n,p in enumerate(l_tmp) :
     strc +=str(p)
     if n == np-1:
      strc += '};\n'
     else :
      strc += ','
    store.write(strc)

    #strc = 'Plane Surface(' + str(ss)+ ') = {'+ str(loop) + '};\n' 
    #horizontal_surfaces.append(ss)
    #store.write(strc)
    if kk == 0: 
     lower_loops.append(loop)
    else:
     upper_loops.append(loop)
    
    loop +=1
    #----------------------

   #Write Vertical Lines 
   for p in range(len(region_points)) :
    p1 = p_start-np + p
    p2 = p_start + p
    lines.append([ll,p1,p2])
    store.write( 'Line('+str(ll) +') = {' + str(p1) +','+ str(p2)+'};\n')
    ll += 1

   #Create vertical loops
   for p in range(np) :
    l1 =   l_start - np + p
    l2 =   l_start  + np + (p+1)%np
    l3 = - (l_start + p)
    l4 = - (l_start  + np + p)
    strc = 'Line Loop(' + str(loop)+ ') = {' + str(l1) + ',' + str(l2) + ',' + str(l3) + ',' + str(l4) + '};\n'
    store.write(strc)
    strc = 'Plane Surface(' + str(ss)+ ') = {'+ str(loop) + '};\n' 
    store.write(strc)
    if g > 0:
     pore_surfaces.append(ss)
    else:
     lateral_surfaces.append(ss)
     lateral_loops.append([l1,l2,l3,l4])
     #------------Get the contacts-----------------
     p1 = points[lines[abs(l1)][1]]
     p2 = points[lines[abs(l1)][2]]
     p3 = points[lines[abs(l2)][2]]
     pores = True
     if abs(p1[0]-Lx/2) < delta and abs(p2[0]-Lx/2)< delta and  abs(p3[0]-Lx/2)< delta:
      C1.append([ss,l1,l2,l3,l4])      
      pores = False
     if abs(p1[0]+Lx/2) < delta and abs(p2[0]+Lx/2)< delta and abs(p3[0]+Lx/2)< delta:
      C2.append([ss,l1,l2,l3,l4])      
      pores = False
     if abs(p1[1]-Ly/2) < delta and abs(p2[1]-Ly/2)< delta and  abs(p3[1]-Ly/2)< delta:
      C3.append([ss,l1,l2,l3,l4])      
      pores = False
     if abs(p1[1]+Ly/2) < delta and abs(p2[1]+Ly/2)< delta and  abs(p3[1]+Ly/2)< delta:
      C4.append([ss,l1,l2,l3,l4])      
      pores = False
     if pores:
      pore_surfaces.append(ss)
     #--------------------------------------------
    ss += 1
    loop +=1
  
  #Create horizontal loops
  strc = 'Plane Surface(' + str(ss)+ ') = {'
  for n,p in enumerate(upper_loops) :
   strc +=str(p)
   if n == ng-1:
    strc += '};\n'
   else :
    strc += ','
  store.write(strc)
  ls = [ss]
  ss +=1

  #Create horizontal loops
  strc = 'Plane Surface(' + str(ss)+ ') = {'
  for n,p in enumerate(lower_loops) :
   strc +=str(p)
   if n == ng-1:
    strc += '};\n'
   else :
    strc += ','
  store.write(strc)
  hs = [ss]
  ss +=1




  #GENERALIZE THIS-----------------------------------------------------------------
  
  Nc1 = len(C1)
  Nc2 = len(C2)
  additional_boundary = []
  
  if argv.setdefault('Periodic',[True,True,True])[0]:
   Px1 = []
   Px2 = []
   assert Nc1 == Nc2 
   for n in range(Nc1):
    i1 = n
    i2 = Nc1-1-n
    store.write('Periodic Surface ' + str(C1[i1][0]) + ' {') 
    for n,k in enumerate(C1[i1][1:5]):
     store.write(str(k))
     if not n == 3:
      store.write(',')
    store.write('} = ' + str(C2[i2][0]) + ' {')
    tmp = C2[i2][1:5]
    store.write(str(-tmp[0]) + ',')
    store.write(str(-tmp[3]) + ',')
    store.write(str(-tmp[2]) + ',')
    store.write(str(-tmp[1]) + '};\n')
    Px1.append(C1[i1][0])
    Px2.append(C2[i1][0])

   strc = r'''Physical Surface("Periodic_1") = {'''
   for n,p in enumerate(Px1) :
    strc +=str(p)
    if n == len(Px1)-1:
     strc += '};\n'
    else :
     strc += ','
   store.write(strc)
   strc = r'''Physical Surface("Periodic_2") = {'''
   for n,p in enumerate(Px2) :
    strc +=str(p)
    if n == len(Px2)-1:
     strc += '};\n'
    else :
     strc += ','
   store.write(strc)
  else:
   for n in range(Nc1):
    additional_boundary.append(C1[n][0])
   for n in range(Nc2):
    additional_boundary.append(C2[n][0])

 
 

  #store.write(r'''Physical Surface("Periodic_1") = {''' + str(C1[i1][0]) + '};\n') 
  #store.write(r'''Physical Surface("Periodic_2") = {''' + str(C2[i2][0]) + '};\n') 
  #---------------------------------
 
  #periodic contats---------------------------------- 
  Nc3 = len(C3)
  Nc4 = len(C4)
  if argv.setdefault('Periodic',[True,True,True])[1]:
   Px3 = []
   Px4 = []
   assert Nc3 == Nc4 
   for n in range(Nc3):
    i3 = n
    i4 = Nc3-1-n
   
    store.write('Periodic Surface ' + str(C3[i3][0]) + ' {')
    for n,k in enumerate(C3[i3][1:5]):
     store.write(str(k))
     if not n == 3:
      store.write(',')
    store.write('} = ' + str(C4[i4][0]) + ' {')
    tmp = C4[i4][1:5]
    store.write(str(-tmp[0]) + ',')
    store.write(str(-tmp[3]) + ',')
    store.write(str(-tmp[2]) + ',')
    store.write(str(-tmp[1]) + '};\n')
    Px3.append(C3[i3][0])
    Px4.append(C4[i4][0])
  
   strc = r'''Physical Surface("Periodic_3") = {'''
   for n,p in enumerate(Px3) :
    strc +=str(p)
    if n == len(Px3)-1:
     strc += '};\n'
    else :
     strc += ','
   store.write(strc)
   strc = r'''Physical Surface("Periodic_4") = {'''
   for n,p in enumerate(Px4) :
    strc +=str(p)
    if n == len(Px4)-1:
     strc += '};\n'
    else :
     strc += ','
   store.write(strc)
  else:
   for n in range(Nc1):
    additional_boundary.append(C3[n][0])
   for n in range(Nc2):
    additional_boundary.append(C3[n][0])

  #Create Physical Surfaces
  boundary_surfaces = pore_surfaces + hs + ls + additional_boundary
  strc = r'''Physical Surface('Boundary') = {'''
  for n,p in enumerate(boundary_surfaces) :
   strc +=str(p)
   if n == len(boundary_surfaces)-1:
    strc += '};\n'
   else :
    strc += ','
  store.write(strc)


  #store.write(r'''Physical Surface("Periodic_1") = {''' + str(C1[i1][0]) + '};\n') 


  #store.write(r'''Physical Surface("Periodic_3") = {''' + str(C3[i3][0]) + '};\n') 
  #store.write(r'''Physical Surface("Periodic_4") = {''' + str(C4[i4][0]) + '};\n') 
  #-----------------------------------------------------------------------------

  #Create Physical Bulk
  surfaces = lateral_surfaces + pore_surfaces + hs + ls
  strc = r'''Surface Loop(''' + str(ss) + ') = {'
  for n,p in enumerate(surfaces) :
   strc +=str(p)
   if n == len(surfaces)-1:
    strc += '};\n'
   else :
    strc += ','
  store.write(strc)
  strc = r'''Volume(0) = {''' + str(ss) + '};\n'
  store.write(strc)
  strc = r'''Physical Volume('Bulk') = {0};'''
  store.write(strc)
