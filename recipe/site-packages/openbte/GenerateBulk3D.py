import os,sys
import numpy as np
import random
import math
import subprocess


def mesh(argv):


  Lx = argv['lx']
  Ly = argv['ly']
  Lz = argv['lz']



  frame = []
  frame.append([float(-Lx)/2,float(Ly)/2])
  frame.append([float(Lx)/2,float(Ly)/2])
  frame.append([float(Lx)/2,float(-Ly)/2])
  frame.append([float(-Lx)/2,float(-Ly)/2])


  mesh_ext = argv['step']
  Nx = float(argv.setdefault('Nx',1))
  Ny = float(argv.setdefault('Ny',1))
  Nz = float(argv.setdefault('Nz',1))

  Lx *=Nx
  Ly *=Ny
  Lz *=Nz


  store = open('mesh.geo', 'w+')

  points  = [[-Lx/2,-Ly/2,-Lz/2],\
             [-Lx/2,+Ly/2,-Lz/2],\
             [Lx/2,Ly/2,-Lz/2],\
             [Lx/2,-Ly/2,-Lz/2],\
             [-Lx/2,-Ly/2,Lz/2],\
             [-Lx/2,+Ly/2,Lz/2],\
             [Lx/2,Ly/2,Lz/2],\
             [Lx/2,-Ly/2,Lz/2]]

  #Write points
  for p,point in enumerate(points):
   store.write( 'Point('+str(p) +') = {' + str(point[0]) +','+\
                                          str(point[1])+',' +\
                                          str(point[2]) + ','+\
                                          str(mesh_ext) +'};\n')

  #Write lines
  ll = 1
  store.write( 'Line('+str(ll) +') = {' + str(0) +','+ str(1)+'};\n')
  ll += 1
  store.write( 'Line('+str(ll) +') = {' + str(1) +','+ str(2)+'};\n')
  ll += 1
  store.write( 'Line('+str(ll) +') = {' + str(2) +','+ str(3)+'};\n')
  ll += 1
  store.write( 'Line('+str(ll) +') = {' + str(3) +','+ str(0)+'};\n')
  ll += 1
  store.write( 'Line('+str(ll) +') = {' + str(4) +','+ str(5)+'};\n')
  ll += 1
  store.write( 'Line('+str(ll) +') = {' + str(5) +','+ str(6)+'};\n')
  ll += 1
  store.write( 'Line('+str(ll) +') = {' + str(6) +','+ str(7)+'};\n')
  ll += 1
  store.write( 'Line('+str(ll) +') = {' + str(7) +','+ str(4)+'};\n')
  ll += 1
  store.write( 'Line('+str(ll) +') = {' + str(0) +','+ str(4)+'};\n')
  ll += 1
  store.write( 'Line('+str(ll) +') = {' + str(1) +','+ str(5)+'};\n')
  ll += 1
  store.write( 'Line('+str(ll) +') = {' + str(2) +','+ str(6)+'};\n')
  ll += 1
  store.write( 'Line('+str(ll) +') = {' + str(3) +','+ str(7)+'};\n')
  ll += 1

  #Write Surfaces
  loops = [[1,2,3,4],\
           [5,6,7,8],\
           [9,5,-10,-1],\
           [10,6,-11,-2],\
           [12,-7,-11,3],\
           [4,9,-8,-12]]

  for nloop,loop in enumerate(loops) :
   index = nloop + 12
   strc = 'Line Loop(' + str(index)+ ') = {'
   for n,p in enumerate(loop) :
    strc +=str(p)
    if n == len(loop)-1:
     strc += '};\n'
    else :
     strc += ','
   store.write(strc)
   strc = 'Plane Surface(' + str(nloop+1)+ ') = {'+ str(index) + '};\n'
   store.write(strc)

  strc = 'Surface Loop(' + str(12)+ ') = {-1,3,4,5,6,2};\n'
  store.write(strc)
  strc = 'Volume(0)={12};\n'
  store.write(strc)
  strc = 'Physical Volume("Matrix")={0};\n'
  store.write(strc)



  #Write Periodic Surfaces--------------------------------------------
  strc = 'Periodic Surface 1 {1,2,3,4} = 2 {5,6,7,8};\n'
  store.write(strc)
  strc = 'Periodic Surface 3 {9,5,-10,-1} = 5 {12,-7,-11,3};\n'
  store.write(strc)
  strc = 'Periodic Surface 4 {10,6,-11,-2} = 6 {9,-8,-12,4};\n'
  store.write(strc)
  #------------------------------------------------------------------


  boundary_surfaces = []
  if argv.setdefault('Periodic',[True,True,True])[0]:
   strc = r'''Physical Surface('Periodic_1') = {3} ;''' + '\n'
   store.write(strc)
   strc = r'''Physical Surface('Periodic_2') = {5} ;''' + '\n'
   store.write(strc)
  else:
   boundary_surfaces.append(3)
   boundary_surfaces.append(5)

  if argv.setdefault('Periodic',[True,True,True])[1]:
   strc = r'''Physical Surface('Periodic_3') = {4} ;''' + '\n'
   store.write(strc)
   strc = r'''Physical Surface('Periodic_4') = {6} ;''' + '\n'
   store.write(strc)
  else:
   boundary_surfaces.append(4)
   boundary_surfaces.append(6)

  if argv.setdefault('Periodic',[True,True,True])[2]:
   strc = r'''Physical Surface('Periodic_5') = {1} ;''' + '\n'
   store.write(strc)
   strc = r'''Physical Surface('Periodic_6') = {2} ;''' +'\n'
   store.write(strc)
  else:
   boundary_surfaces.append(1)
   boundary_surfaces.append(2)


  #write Boundary surfaces
  if len(boundary_surfaces) > 0:
   strc = r'''Physical Surface('Boundary') = {'''
   for p,side in enumerate(boundary_surfaces) :
    strc +=str(side)
    if p == len(boundary_surfaces)-1:
     strc += '};\n'
    else :
     strc += ','
   store.write(strc)


   store.close()
