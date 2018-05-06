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
  store = open('mesh.geo', 'w+')
  #CREATE BULK SURFACE------------------------------------
  c = pyclipper.Pyclipper()
  c.AddPath(scale_to_clipper(frame), pyclipper.PT_SUBJECT,True) 
  c.AddPaths(scale_to_clipper(polygons),pyclipper.PT_CLIP,True) 
  solution = scale_from_clipper(c.Execute(pyclipper.CT_DIFFERENCE,pyclipper.PFT_EVENODD,pyclipper.PFT_EVENODD))

  points = []
  lines = []
  loop = 0
  ss = 0
  ll = 1
  line_contour = []
  pore_regions = []
  #check_points = []
  p_start = 0
  for g,region in enumerate(solution) :
   #p_start = len(points)

   #Write Points
   
   region_points = []
   for p,point in enumerate(region) :

    new_point = [round(point[0],4),round(point[1],4)]
    if not new_point in points:
     points.append(new_point)
     #region_points.append(new_point)
     region_points.append(len(points)-1)
     store.write( 'Point('+str(len(points)-1) +') = {' + str(new_point[0]) +','+ str(new_point[1])+',0,'+ str(mesh_ext) +'};\n')
    else:
     region_points.append(points.index(new_point))

   #Write lines   
   local_line_list = []
   for p,point in enumerate(region_points) :
    #---
    #-------
    p1 = region_points[p]
    p2 = region_points[(p+1)%len(region_points)]
    if not p1 == p2:
     lines.append([ll,p1,p2])
     local_line_list.append(ll)
     store.write( 'Line('+str(ll) +') = {' + str(p1) +','+ str(p2)+'};\n')
     line_contour.append([ll,p1,p2])
     ll += 1
    else:
     print('here')

   #create the loop
   strc = 'Line Loop(' + str(loop)+ ') = {'
   #for p,point in enumerate(region_points) :
   for local_line in local_line_list: 
    #strc +=str(p_start+p+1)
    strc +=str(local_line)
    #if p == len(region_points)-1:
    if local_line == local_line_list[-1]:
     strc += '};\n'
    else :
     strc += ','

   store.write(strc)
   loop +=1
   #p_start += len(region)
   

  #Create Bulk Surface
  #Get corners----------------------------------------------------------
  corner_list = [] 
  for r1,region1 in enumerate(solution) :
   intersect = False
   for r2,region2 in enumerate(solution) :
    if not r2 == r1:
     c = pyclipper.Pyclipper()
     c.AddPath(scale_to_clipper(region1), pyclipper.PT_SUBJECT,True) 
     c.AddPaths(scale_to_clipper([region2]),pyclipper.PT_CLIP,True) 
     test = scale_from_clipper(c.Execute(pyclipper.CT_INTERSECTION,pyclipper.PFT_EVENODD,pyclipper.PFT_EVENODD))
     if len(test) > 0 :
      intersect = True
      break
   if intersect == False:
    corner_list.append(r1)
  
  #Get frame --------------------------------------------------------- 
  for r1,region1 in enumerate(solution) :
   is_frame = True
   for r2,region2 in enumerate(solution) :
    if not r2 == r1 and (not r2 in corner_list):
     c = pyclipper.Pyclipper()
     c.AddPath(scale_to_clipper(region1), pyclipper.PT_SUBJECT,True) 
     c.AddPaths(scale_to_clipper([region2]),pyclipper.PT_CLIP,True) 
     test = scale_from_clipper(c.Execute(pyclipper.CT_INTERSECTION,pyclipper.PFT_EVENODD,pyclipper.PFT_EVENODD))
     if len(test) == 0 :
      is_frame = False
      break
   if is_frame:
    frame_index = r1
    break
  #-------------------------------------

  #----------------------
  no_corner=False
  for r,region in enumerate(solution) :
   if (not r in corner_list):
    no_corner=True
    break 

  if no_corner:
   nostart = False
   separate_list = []
   strc = 'Plane Surface(' + str(ss)+ ') = {' 
   for r,region in enumerate(solution) :
  
    if (not r in corner_list):
     if nostart : strc +=','
     strc += str(r)
     nostart = True
  
   store.write(strc + '};\n')
   ss += 1

  #Add piece of corners (if any)---------------------------------------
  for r in corner_list:
   strc = 'Plane Surface(' + str(ss)+ ') = {' + str(r) + '};\n'
   store.write(strc)
   ss += 1

  #Collect Bulk Surfaces-------------
  tmp = ''
  for s in range(ss):
   tmp += str(s)
   if s < ss-1: tmp +=','

  strc = r'''Physical Surface('Bulk') = {''' + tmp + '};\n' 
  store.write(strc)
  #ss += 1

  #------------------------------------------------------------------- 
  #CREATE PORE SURFACE------------------------------------
  c = pyclipper.Pyclipper()
  c.AddPath(scale_to_clipper(frame), pyclipper.PT_CLIP,True) 
  c.AddPaths(scale_to_clipper(polygons),pyclipper.PT_SUBJECT,True) 
  solution = scale_from_clipper(c.Execute(pyclipper.CT_INTERSECTION,pyclipper.PFT_EVENODD,pyclipper.PFT_EVENODD))

  pore_wall = []
  pore_surface = []
  for region in solution:
    tmp = []
    for point in region:
     #Create Points
     index = point_exists(point,points)
     #index = point_exists(point,new_points)
     tmp.append(index)
    line_loop = []
    #do_add = 0 
    for n in range(len(tmp)):
     n1 = n
     n2 = (n+1)%len(tmp)
     line = [tmp[n1],tmp[n2]]
 
     index =  line_exists(line,lines)
     if not index==0:
      pore_wall.append(index[0])
      line_loop.append(index)
     
 
    #Add Physical Pore 
    if include_pores:
     new_lines = LineOrdering(line_loop) 
     strc = 'Line Loop(' + str(loop)+ ') = {'
     for p,line in enumerate(new_lines) :
      strc +=str(line[0])
      if p == len(new_lines)-1:
       strc += '};\n'
      else :
       strc += ','
     store.write(strc)
     
     strc = 'Plane Surface(' + str(ss)+ ') = {' + str(loop) + '};\n'
     store.write(strc)
     pore_regions.append(ss)
     ss +=1
     loop +=1

  if include_pores :
   #Create Pores Physical Surfaces---------------
   strc = r'''Physical Surface('Pores2') = {''' 
   for t,r in enumerate(pore_regions) :
    strc += str(r)
    if t == len(pore_regions)-1:
     strc += '};\n'
    else :
     strc += ','
   store.write(strc)
  #---------------------------------------------

  frame = np.array(frame)
  Maxy = np.max(frame[:,1])
  Miny = np.min(frame[:,1])
  Maxx= np.max(frame[:,0])
  Minx = np.min(frame[:,0])


  lower = []
  upper = []
  hot = []
  cold = []
  
  if include_pores :
   line_check = lines 
  else :
   line_check = line_contour

  deltax = (Maxx-Minx)/1e5
  deltay = (Maxy-Miny)/1e5


  for l in line_check:
   p1 = points[l[1]]
   p2 = points[l[2]]
 
   if abs(p1[0] - Minx)<deltax and  abs(p2[0] - Minx)<deltax :
    hot.append(l)
   if abs(p1[0] - Maxx)<deltax and  abs(p2[0] - Maxx)<deltax :
    cold.append(l)
   if abs(p1[1] - Miny)<deltay and  abs(p2[1] - Miny)<deltay :
    lower.append(l)
   if abs(p1[1] - Maxy)<deltay and  abs(p2[1] - Maxy)<deltay :
    upper.append(l)


  additional_boundary = []
  if argv.setdefault('Periodic',[True,True,True])[0]:
   strc = r'''Physical Line('Periodic_1') = {'''
   for n in range(len(hot)-1):
    strc += str(hot[n][0]) + ','
   strc += str(hot[-1][0]) + '};\n'
   store.write(strc)
  
   strc = r'''Physical Line('Periodic_2') = {'''
   for n in range(len(cold)-1):
    strc += str(cold[n][0]) + ','
   strc += str(cold[-1][0]) + '};\n'
   store.write(strc)
  else:
   for k in hot:
    additional_boundary.append(k[0])
   for k in cold:
    additional_boundary.append(k[0])

  if argv.setdefault('Periodic',[True,True,True])[1]:
   strc = r'''Physical Line('Periodic_3') = {'''
   for n in range(len(upper)-1):
    strc += str(upper[n][0]) + ','
   strc += str(upper[-1][0]) + '};\n'
   store.write(strc)

   strc = r'''Physical Line('Periodic_4') = {'''
   for n in range(len(lower)-1):
    strc += str(lower[n][0]) + ','
   strc += str(lower[-1][0]) + '};\n'
   store.write(strc)
  else:
   for k in lower:
    additional_boundary.append(k[0])
   for k in upper:
    additional_boundary.append(k[0])

  
  #Collect Wall
  boundary_surfaces = pore_wall + additional_boundary
  strc = r'''Physical Line('Boundary') = {''' 
  for r,region in enumerate(boundary_surfaces) :
   strc += str(region) 
   if r == len(boundary_surfaces)-1:
    strc += '};\n'
    store.write(strc)
   else :
    strc += ','



  for n in range(len(hot)):
   periodic = FindPeriodic(hot[n],cold,points)
   strc = 'Periodic Line{' + str(hot[n][0]) + '}={' + str(periodic) + '};\n'
   store.write(strc)
  
  for n in range(len(lower)):
   periodic = FindPeriodic(lower[n],upper,points)
   strc = 'Periodic Line{' + str(lower[n][0]) + '}={' + str(periodic) + '};\n'
   store.write(strc)
  
#-------------------------------------------------------
  store.close()
 

