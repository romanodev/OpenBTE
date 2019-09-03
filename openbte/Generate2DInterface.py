import numpy as np


def Generate2DInterface(argv):

 Lx = argv['lx']
 Ly = argv['ly']
 step1 = argv['step1']
 step2 = argv['step2']
 delta = argv['delta']


 store = open("mesh.geo", 'w+')

 store.write( 'Point(1) = {' + str(-Lx/2.0) +','+ str(-Ly/2.0)+',0,'+ str(step1) +'};\n')
 store.write( 'Point(2) = {' + str(-delta/2) +','+ str(-Ly/2.0)+',0,'+ str(step1) +'};\n')
 store.write( 'Point(3) = {' + str(0) +','+ str(-Ly/2.0)+',0,'+ str(step2) +'};\n')
 store.write( 'Point(4) = {' + str(delta/2) +','+ str(-Ly/2.0)+',0,'+ str(step1) +'};\n')
 store.write( 'Point(5) = {' + str(Lx/2.0) +','+ str(-Ly/2.0)+',0,'+ str(step1) +'};\n')
 store.write( 'Point(6) = {' + str(Lx/2.0) +','+ str(Ly/2.0)+',0,'+ str(step1) +'};\n')
 store.write( 'Point(7) = {' + str(delta/2) +','+ str(Ly/2.0)+',0,'+ str(step1) +'};\n')
 store.write( 'Point(8) = {' + str(0) +','+ str(Ly/2.0)+',0,'+ str(step2) +'};\n')
 store.write( 'Point(9) = {' + str(-delta/2) +','+ str(Ly/2.0)+',0,'+ str(step1) +'};\n')
 store.write( 'Point(10) = {' + str(-Lx/2.0) +','+ str(Ly/2.0)+',0,'+ str(step1) +'};\n')

 for k in range(9):
  store.write( 'Line('+str(k+1) +') = {' + str(k+1) +','+ str(k+2)+'};\n')
 store.write( 'Line('+str(10) +') = {' + str(10) +','+ str(1)+'};\n')


 store.write( 'Line('+str(11) +') = {' + str(2) +','+ str(9)+'};\n')


 #Building disordered interface---
 #store.write( 'Line('+str(12) +') = {' + str(3) +','+ str(8)+'};\n')
 interface = []

 ps = 3
 N = 50
 dx_old = 0
 pp = 11
 ll = 12
 pp_old = 8
 for n in range(N):
  yy = Ly/2 - (n+1)*Ly/(N+1) 
  store.write( 'Point(' + str(pp) + r') = {' + str(dx_old) +','+ str(yy)+',0,'+ str(step1) +'};\n')
  store.write( 'Line(' + str(ll) + r') = {' + str(pp_old) +','+ str(pp) + '};\n')
  interface.append(ll)
  pp +=1
  ll +=1

  dx = (np.random.rand()-0.5)/2
  if n == N-1:
   store.write( 'Point(' + str(pp) + r') = {' + str(0) +','+ str(yy)+',0,'+ str(step1) +'};\n')
  else:
   store.write( 'Point(' + str(pp) + r') = {' + str(dx) +','+ str(yy)+',0,'+ str(step1) +'};\n')
  store.write( 'Line(' + str(ll) + r') = {' + str(pp-1) +','+ str(pp) + '};\n')
  interface.append(ll)
  pp_old = pp
  pp +=1
  ll +=1
  dx_old = dx


 #store.write( 'Line(' + str(ll) + r') = {' + str(pp-1) +','+ str(pp) + '};\n')
 #ll +=1
 #store.write( 'Point(' + str(pp) + r') = {' + str(0) +','+ str(yy)+',0,'+ str(step1) +'};\n')
 #pp +=1
 store.write( 'Line(' + str(ll) + r') = {' + str(pp-1) +','+ str(3) + '};\n')
 interface.append(ll)
 ll +=1

 #Write final line--- 
 #pe = 12
 #store.write( 'Line(' + str(11+N+1) + r') = {' + str(ps) +','+ str(pe)'};\n')
 #------------------
 #--------------------------------

 store.write( 'Line('+str(ll) +') = {' + str(4) +','+ str(7)+'};\n')
 
 store.write(r'''Line Loop(15) = {1,11,9,10};''' + '\n')
 store.write(r'''Plane Surface(1)= {15};'''+ '\n')

 #First side
 strc = r'''Line Loop(16) = {-8,'''
 for n,i in enumerate(interface):
  strc += str(i) + ','
 strc += '-2,11};\n'
 store.write(strc)
 store.write(r'''Plane Surface(2)= {16};'''+ '\n')
 #----------------------

 #Second side
 strc = r'''Line Loop(17) = {7,'''
 for n,i in enumerate(interface):
  strc += str(i) + ','
 strc += '3,' + str(2*N+13) + '};\n'
 store.write(strc)
 store.write(r'''Plane Surface(3)= {17};'''+'\n')
 #----------------------

 #last side 
 store.write(r'''Line Loop(18) = {6,''' + str(-2*N-13) + r''',4,5};''' + '\n')
 store.write(r'''Plane Surface(4)= {18};''' +'\n')


 strc = r'''Physical Surface('Matrix') = {1,2}''' + ';\n'
 store.write(strc)
 strc = r'''Physical Surface('Inclusion') = {3,4}''' + ';\n'
 store.write(strc)
 

 #store.write(r'''Line Loop(16) = {2,12,8,-11};''' + '\n')

 #store.write(r'''Line Loop(17) = {3,13,7,-12};''' + '\n')

 
 #store.write(r'''Physical Surface("Matrix")= {1,2,3,4};'''+ '\n')
 strc = (r'''Physical Line("Interface")= {''')
 for n,i in enumerate(interface):
  strc += str(i) 
  if n == len(interface)-1:
   strc += '};\n'
  else:
   strc += ','
 store.write(strc)

 store.write(r'''Physical Line("Periodic_1")= {10};'''+'\n')
 store.write(r'''Physical Line("Periodic_2")= {5};'''+'\n')
 store.write(r'''Physical Line("Periodic_3")= {1,2,3,4};'''+'\n')
 store.write(r'''Physical Line("Periodic_4")= {6,7,8,9};'''+'\n')

 store.write(r'''Periodic Line{10}={-5};''' + '\n')
 store.write(r'''Periodic Line{4}={-6};''' + '\n')
 store.write(r'''Periodic Line{3}={-8};''' + '\n')
 store.write(r'''Periodic Line{2}={-7};''' + '\n')
 store.write(r'''Periodic Line{1}={-6};''' + '\n')

 store.close()



  
 

