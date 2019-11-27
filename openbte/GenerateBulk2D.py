import xml.etree.ElementTree as ET
import os,sys
import numpy as np
import random
import math
from matplotlib.pylab import *
import subprocess

def create_structured_bulk(argv):



 res = argv.setdefault('reservoirs',False)
 direction = argv['direction']


 Lx = argv['lx']
 Ly = argv['ly']
 mesh_ext = argv['step']
 layers_x = int(Lx/mesh_ext)
 layers_y = int(Ly/mesh_ext)
 store = open("mesh.geo", 'w+')
 store.write('h='+str(mesh_ext) + ';\n')

 store.write( 'Point(0) = {' + str(-Lx/2.0) +','+ str(-Ly/2.0)+',0,h};\n')
 store.write('out[] = Extrude{0,''' + str(Ly) + r''',0}{ Point{0};Layers{''' + str(layers_x) + r'''};Recombine;};''' + '\n')
 store.write('out[] = Extrude{' + str(Lx) + r''',0,0}{ Line{out[0]};Layers{''' + str(layers_y) + r'''};Recombine;};''' + '\n')

 #Create Surface
 strc = r'''Physical Surface('bulk') = {out[1]};'''+'\n'
 store.write(strc)


 bs = []
 if argv.setdefault('Periodic',[True,True,True])[1] :
  if res and direction=='x':
   strc = r'''Physical Line('Hot') = {1};''' + '\n'
   store.write(strc)
   strc = r'''Physical Line('Cold') = {2};''' +'\n'
   store.write(strc)
  else:
   strc = r'''Physical Line('Periodic_1') = {1};''' + '\n'
   store.write(strc)
   strc = r'''Physical Line('Periodic_2') = {2};''' + '\n'
   store.write(strc)
 else:
  bs.append(1)
  bs.append(3)

 if argv.setdefault('Periodic',[True,True,True])[0] :
  if res and direction=='y':
   strc = r'''Physical Line('Cold') = {3};''' + '\n'
   store.write(strc)
   strc = r'''Physical Line('Hot') = {4};''' +'\n'
   store.write(strc)
  else:
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

  #res = argv.setdefault('reservoirs',False)
  direction = argv['direction']

  Lx = float(argv['lx'])
  Ly = float(argv['ly'])
  frame = []
  frame.append([-Lx/2,Ly/2])
  frame.append([Lx/2,Ly/2])
  frame.append([Lx/2,-Ly/2])
  frame.append([-Lx/2,-Ly/2])

  refine = argv.setdefault('refine',False)

  mesh_ext = argv['step']
  points = []
  lines = []

  store = open("mesh.geo", 'w+')
  store.write('h='+str(mesh_ext) + ';\n')

  for k,p in enumerate(frame) :
   store.write( 'Point('+str(k) +') = {' + str(p[0]) +','+ str(p[1])+',0,h};\n')

  if refine:
   delta = argv.setdefault('delta',Lx/10)  
   frame_small = []
   frame_small.append([-Lx/2+delta,Ly/2])
   frame_small.append([Lx/2-delta,Ly/2])
   frame_small.append([Lx/2-delta,-Ly/2])
   frame_small.append([-Lx/2+delta,-Ly/2])
   for k,p in enumerate(frame_small) :
    store.write( 'Point('+str(k+4) +') = {' + str(p[0]) +','+ str(p[1])+',0,h};\n')


  ll = 0
  for k,p in enumerate(frame) :
   p1 = k
   p2 = (k+1)%len(frame)
   ll += 1
   store.write( 'Line('+str(ll) +') = {' + str(p1) +','+ str(p2)+'};\n')

  strc = 'Line Loop(5) = {'
  for p,point in enumerate(frame) :
   strc +=str(p+1)
   if p == len(frame)-1:
    strc += '};\n'
   else :
    strc += ','
  store.write(strc)

  #Create Surface
  strc = 'Plane Surface(10) = {5};\n'
  store.write(strc)
  strc = r'''Physical Surface('Matrix') = {10};'''+'\n'
  store.write(strc)

  bs = []
  if argv.setdefault('Periodic',[True,True,True])[1] :
    strc = r'''Physical Line('Periodic_1') = {1};''' + '\n'
    store.write(strc)
    strc = r'''Physical Line('Periodic_2') = {3};''' + '\n'
    store.write(strc)
  else:
   if direction=='y':
    strc = r'''Physical Line('Cold') = {2};''' + '\n'
    store.write(strc)
    strc = r'''Physical Line('Hot') = {4};''' +'\n'
    store.write(strc)
   else:
    bs.append(1)
    bs.append(3)

  if argv.setdefault('Periodic',[True,True,True])[0] :
    strc = r'''Physical Line('Periodic_3') = {2};''' + '\n'
    store.write(strc)
    strc = r'''Physical Line('Periodic_4') = {4};''' +'\n'
    store.write(strc)
  else:
   if direction=='x':
    strc = r'''Physical Line('Cold') = {2};''' + '\n'
    store.write(strc)
    strc = r'''Physical Line('Hot') = {4};''' +'\n'
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

#-------------------------------------------------------
  store.close()

def create_unstructured_bulk_refine(argv):

  #res = argv.setdefault('reservoirs',False)
  direction = argv['direction']

  Lx = float(argv['lx'])
  Ly = float(argv['ly'])
  frame = []
  frame.append([-Lx/2,Ly/2])
  frame.append([Lx/2,Ly/2])
  frame.append([Lx/2,-Ly/2])
  frame.append([-Lx/2,-Ly/2])

  refine = argv.setdefault('refine',False)

  mesh_ext = argv['step']
  points = []
  lines = []

  store = open("mesh.geo", 'w+')
  store.write('h='+str(mesh_ext) + ';\n')
  store.write('h2='+str(argv['step2']) + ';\n')

  for k,p in enumerate(frame) :
   store.write( 'Point('+str(k) +') = {' + str(p[0]) +','+ str(p[1])+',0,h2};\n')

  delta = argv.setdefault('delta',Lx/10)  
  frame_small = []
  frame_small.append([-Lx/2+delta,Ly/2])
  frame_small.append([Lx/2-delta,Ly/2])
  frame_small.append([Lx/2-delta,-Ly/2])
  frame_small.append([-Lx/2+delta,-Ly/2]) 
  for k,p in enumerate(frame_small) :
    store.write( 'Point('+str(k+4) +') = {' + str(p[0]) +','+ str(p[1])+',0,h};\n')
  ll = 1


  store.write( 'Line('+str(ll) +') = {' + str(4) +','+ str(5)+'};\n'); ll+=1
  store.write( 'Line('+str(ll) +') = {' + str(5) +','+ str(6)+'};\n'); ll+=1
  store.write( 'Line('+str(ll) +') = {' + str(6) +','+ str(7)+'};\n'); ll+=1
  store.write( 'Line('+str(ll) +') = {' + str(7) +','+ str(4)+'};\n'); ll+=1
  store.write( 'Line('+str(ll) +') = {' + str(5) +','+ str(1)+'};\n'); ll+=1
  store.write( 'Line('+str(ll) +') = {' + str(1) +','+ str(2)+'};\n'); ll+=1
  store.write( 'Line('+str(ll) +') = {' + str(2) +','+ str(6)+'};\n'); ll+=1
  store.write( 'Line('+str(ll) +') = {' + str(7) +','+ str(3)+'};\n'); ll+=1
  store.write( 'Line('+str(ll) +') = {' + str(3) +','+ str(0)+'};\n'); ll+=1
  store.write( 'Line('+str(ll) +') = {' + str(0) +','+ str(4)+'};\n'); ll+=1
  store.write( 'Line Loop('+str(ll) +') = {1,2,3,4};'); ll+=1
  store.write( 'Line Loop('+str(ll) +') = {5,6,7,-2};'); ll+=1
  store.write( 'Line Loop('+str(ll) +') = {4,-10,-9,-8};'); ll+=1
  store.write('Plane Surface(10) = {11};\n')
  store.write('Plane Surface(11) = {12};\n')
  store.write('Plane Surface(12) = {13};\n')
  strc = r'''Physical Surface('Matrix') = {10};'''+'\n'
  store.write(r'''Physical Surface('Matrix') = {10,11,12};''')


  bs = []
  if argv.setdefault('Periodic',[True,True,True])[1] :
    strc = r'''Physical Line('Periodic_1') = {10,1,5};''' + '\n'
    store.write(strc)
    strc = r'''Physical Line('Periodic_2') = {7,3,8};''' + '\n'
    store.write(strc)
  else:
   if direction=='y':
    strc = r'''Physical Line('Cold') = {7,3,8};''' + '\n'
    store.write(strc)
    strc = r'''Physical Line('Hot') = {10,1,5};''' +'\n'
    store.write(strc)
   else:
    bs +=  [10,1,5,7,3,8]
  
  if argv.setdefault('Periodic',[True,True,True])[0] :
    strc = r'''Physical Line('Periodic_1') = {9};''' + '\n'
    store.write(strc)
    strc = r'''Physical Line('Periodic_2') = {6};''' + '\n'
    store.write(strc)
  else:
   if direction=='x':
    strc = r'''Physical Line('Cold') = {6};''' + '\n'
    store.write(strc)
    strc = r'''Physical Line('Hot') = {9};''' +'\n'
    store.write(strc)
   else:
    bs +=  [9,6]
  

  strc = 'Periodic Line{1}={-3};\n'
  store.write(strc)
  strc = 'Periodic Line{5}={-7};\n'
  store.write(strc)
  strc = 'Periodic Line{10}={-8};\n'
  store.write(strc)
  strc = 'Periodic Line{9}={-6};\n'
  store.write(strc)


  store.close()



def mesh(argv):


  if argv.setdefault('refine',False):
    create_unstructured_bulk_refine(argv)
  else :  
   if argv.setdefault('unstructured',True):
    create_unstructured_bulk(argv)
   else:
    create_structured_bulk(argv)


