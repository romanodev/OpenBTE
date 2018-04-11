import os,sys
import numpy as np
import pyclipper 
import random
import math

def nanowire(argv):

  L = argv['length']
  D = argv['diameter']
  base = argv['base']
  if base == 'circle':
   Na = 24
  if base == 'square':
   Na = 4
  if base == 'triangle':
   Na = 3

  m = argv['step']
  store = open('mesh.geo', 'w+')
  dphi = 2.0*math.pi/Na;
  p = []
  argv.setdefault('periodic',True)
   
  #points   
  for ka in range(Na):
   ph =  dphi/2 + (ka-1) * dphi 
   py  = D/2 * math.cos(ph)
   pz  = D/2 * math.sin(ph)
   store.write( 'Point('+str(ka) +') = {0,'+ str(py)+',' + str(pz) + ','+ str(m) +'};\n')
   store.write( 'Point('+str(ka+Na) +') = {'+str(L) + ',' + str(py) +','+ str(pz)+ ','+ str(m) +'};\n')

  #Lines
  for ka in range(Na):
   store.write('Line(' + str(ka + 1) +') = {' + str(ka) +',' +str((ka+1)%Na)+'};\n')
   store.write('Line(' + str(ka+Na+1) +') = {' + str(ka+Na) +',' +str(Na+(ka+1)%Na)+'};\n')
   store.write('Line(' + str(2*Na+ka+1) +') = {' + str(ka) +',' +str(ka+Na)+'};\n')

  #Surfaces UP
  store.write('Line Loop('+str(3*Na) + ')={') 
  for ka in range(Na):
   store.write(str(ka+1))
   if not ka == Na-1:
    store.write(',')
  store.write('};\n')
  store.write('Plane Surface(1)={' + str(3*Na)+'};\n')

  if argv['periodic']:
   contact_1 = 'Periodic_1'
   contact_2 = 'Periodic_2'
  else:
   contact_1 = 'Hot'
   contact_2 = 'Cold'

  store.write(r'''Physical Surface("''' + contact_1 + r'''")={1};''' + '\n')

  #Surfaces DOWN
  store.write('Line Loop('+str(3*Na+1) + ')={') 
  for ka in range(Na):
   store.write(str(ka+Na+1))
   if not ka == Na-1:
    store.write(',')
  store.write('};\n')
  store.write('Plane Surface(2)={' + str(3*Na+1)+'};\n')
  store.write(r'''Physical Surface("''' + contact_2 + r'''")={2};''' + '\n')


  #Periodic surfaces----------------
  if argv['periodic']:
   store.write('Periodic Surface 2 {') 
   for ka in range(Na):
    store.write(str(ka+Na+1))
    if not ka == Na-1:
     store.write(',')
   store.write('} = 1 {')
   for ka in range(Na):
    store.write(str(ka+1))
    if not ka == Na-1:
     store.write(',')
   store.write('};\n')
  #---------------------------------


  #Surfaces Lateral
  for ka in range(Na):
   store.write('Line Loop('+str(3*Na+ka+2) + ')={' + str(ka+1) + ','\
                                                 + str(2*Na+(ka+1)%Na+1) + ',-'\
                                                 + str(ka+1 + Na) + ',-'\
                                                 + str(2*Na+ka+1) + '};\n')

   store.write('Plane Surface(' + str(ka+3) + ')={' + str(3*Na+ka+2)+'};\n')



  store.write(r'''Physical Surface("Boundary")={''')
  for ka in range(Na):
   store.write(str(ka+3))
   if not ka == Na-1:
    store.write(',')
  store.write('};\n')

  store.write(r'''Surface Loop(0)={1,2,''')
  for ka in range(Na):
   store.write(str(ka+3))
   if not ka == Na-1:
    store.write(',')
  store.write('};\n')
   
  store.write(r'''Volume(0)={0};''' + '\n')
  store.write(r'''Physical Volume("Bulk")={0};''')

  store.close()

  return 3


