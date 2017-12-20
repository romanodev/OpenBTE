import xml.etree.ElementTree as ET
import os,sys
import numpy as np
import random
import math
from matplotlib.pylab import *
import pyclipper
import subprocess

def create_structured_bulk(argv):


 Lx = argv['frame'][0]
 Ly = argv['frame'][1]
 width = argv['width']
 layers_x = int(Lx/mesh_ext)
 layers_y = int(Ly/mesh_ext)
 store = open("mesh.geo", 'w+')
 store.write( 'Point(0) = {' + str(-Lx/2.0) +','+ str(-Ly/2.0)+',0,'+ str(mesh_ext) +'};\n')
 store.write('out[] = Extrude{0,''' + str(Ly) + r''',0}{ Point{0};Layers{''' + str(layers_x) + r'''};Recombine;};''' + '\n')
 store.write('out[] = Extrude{' + str(Lx/2 - width/2) + r''',0,0}{ Line{out[0]};Layers{''' + str(layers_y) + r'''};Recombine;};''' + '\n')
 store.write('out[] = Extrude{' + str(width) + r''',0,0}{ Line{out[1]};Layers{''' + str(layers_y) + r'''};Recombine;};''' + '\n')
 store.write('out[] = Extrude{' + str(Lx/2-width/2) + r''',0,0}{ Line{out[1]};Layers{''' + str(layers_y) + r'''};Recombine;};''' + '\n')


 quit()

 #Create Surface
 strc = r'''Physical Surface('bulk') = {out[1]};'''+'\n' 
 store.write(strc)


 bs = []
 if argv.setdefault('Periodic',[True,True,True])[1] :
  strc = r'''Physical Line('Periodic_1') = {1};''' + '\n'
  store.write(strc)
  strc = r'''Physical Line('Periodic_2') = {2};''' + '\n'
  store.write(strc)
 else:
  bs.append(1)
  bs.append(3)

 if argv.setdefault('Periodic',[True,True,True])[0] :
  strc = r'''Physical Line('Periodic_3') = {3};''' + '\n'
  store.write(strc)
  strc = r'''Physical Line('Periodic_4') = {4};''' +'\n'
  store.write(strc)
 else:
  bs.append(2)
  bs.append(4)
  
 if len(bs) > 0:
  strc = r'''Physical Line('Boundary') = {''' 
  for p,side in enumerate(bs) :
   strc +=str(side)
   if p == len(bs)-1:
    strc += '};\n'
   else :
    strc += ','
  store.write(strc)

 strc = 'Periodic Line{1}={-3};\n'
 store.write(strc)
 strc = 'Periodic Line{2}={-4};\n'
 store.write(strc)
  
 store.close()

def create_unstructured_bulk(argv):

  Lx = argv['frame'][0]
  Ly = argv['frame'][1]
  width = argv['width']
  frame = []
  frame.append([float(-Lx)/2,float(Ly)/2])
  frame.append([float(-width/2),float(Ly)/2])
  frame.append([float(width/2),float(Ly)/2])
  frame.append([float(Lx)/2,float(Ly)/2])
  frame.append([float(Lx)/2,float(-Ly)/2])
  frame.append([float(width/2),float(-Ly)/2])
  frame.append([float(-width/2),float(-Ly)/2])
  frame.append([float(-Lx)/2,float(-Ly)/2])


  m1 = argv['step_bulk']
  m2 = argv['step_interface']
  mesh_step = [m1,m2,m2,m1,m1,m2,m2,m1]

  points = []
  lines = []

  store = open("mesh.geo", 'w+')

  for k,p in enumerate(frame) :
   store.write( 'Point('+str(k) +') = {' + str(p[0]) +','+ str(p[1])+',0,'+ str(mesh_step[k]) +'};\n')

  ll = 0 
  for k,p in enumerate(frame) :
   p1 = k
   p2 = (k+1)%len(frame)
   ll += 1
   store.write( 'Line('+str(ll) +') = {' + str(p1) +','+ str(p2)+'};\n')

  #Buffer line----
  ll += 1
  store.write( 'Line('+str(ll) +') = {1,6};\n')
  #Buffer line----

  ll += 1
  store.write( 'Line('+str(ll) +') = {5,2};\n')
  
  store.write('Line Loop(0) = {1,9,7,8};\n')
  store.write('Line Loop(1) = {3,4,5,10};\n')
  store.write('Line Loop(2) = {2,-10,6,-9};\n')
  store.write('Plane Surface(0) = {0};\n') 
  store.write('Plane Surface(1) = {1};\n') 
  store.write('Plane Surface(2) = {2};\n') 
  store.write(r'''Physical Surface('bulk') = {0,1,2};'''+'\n')

  bs = []
  if argv.setdefault('Periodic',[True,True,True])[1] :
   strc = r'''Physical Line('Periodic_1') = {1,2,3};''' + '\n'
   store.write(strc)
   strc = r'''Physical Line('Periodic_2') = {5,6,7};''' + '\n'
   store.write(strc)
  else:
   bs.append(1)
   bs.append(3)

  if argv.setdefault('Periodic',[True,True,True])[0] :
   strc = r'''Physical Line('Periodic_3') = {4};''' + '\n'
   store.write(strc)
   strc = r'''Physical Line('Periodic_4') = {8};''' +'\n'
   store.write(strc)
  else:
   bs.append(2)
   bs.append(4)
  

  if len(bs) > 0:
   strc = r'''Physical Line('Boundary') = {''' 
   for p,side in enumerate(bs) :
    strc +=str(side)
    if p == len(bs)-1:
     strc += '};\n'
    else :
     strc += ','
   store.write(strc)


  store.write('Periodic Line{1}={-7};\n')
  store.write('Periodic Line{2}={-6};\n')
  store.write('Periodic Line{3}={-5};\n')
  strc = 'Periodic Line{8}={-4};\n'
  store.write(strc)
  
#-------------------------------------------------------
  store.close()
 



def mesh(argv):

  if argv.setdefault('unstructured',True):
   create_unstructured_bulk(argv)
  else:
   create_structured_bulk(argv)




