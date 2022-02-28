from shapely.ops import unary_union
from shapely.geometry import Point,LineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
import shapely
import numpy as np
from .shape import *
from .utils import *
import subprocess
import os
import time

def generate_disk(argv):
  
  strc ='''

SetFactory("OpenCASCADE");

Mesh.CharacteristicLengthMin = STEP;
Mesh.CharacteristicLengthMax = STEP;

Circle(1) = {0,0,0, R1};
Circle(2) = {0,0,0, R2};

Line Loop(1) = {1};
Line Loop(2) = {2};

Plane Surface(1) = {1};
Plane Surface(2) = {1,2};


Physical Surface('GENERATION_Q') = {1};
Physical Surface('Bulk') = {2};
Physical Line('ISO_0.0') = {2};

'''


  strc = strc.replace('STEP',str(argv['step']))
  strc = strc.replace('R1',str(argv['Rh']))
  strc = strc.replace('R2',str(argv['R']))
  strc = strc.replace('Q',str(argv['heat_source']))

  store = open('mesh.geo', 'w+')

  store.write(strc)  

  store.close()
  with open(os.devnull, 'w') as devnull:
    output = subprocess.check_output("gmsh -optimize_netgen -format msh2 -2 mesh.geo -o mesh.msh".split(), stderr=devnull)




def generate_bulk_2D(argv):

  direction = argv.setdefault('direction','x')
  argv.setdefault('boundary',['Periodic','Periodic','Periodic'])
  argv['heat_source']=[]

  frame = generate_frame(**argv)
  Lx = float(argv['lx'])
  Ly = float(argv.setdefault('ly',Lx))

  mesh_ext = argv['step']
  points = []
  lines = []

  store = open("mesh.geo", 'w+')
  store.write('h='+str(mesh_ext) + ';\n')

  for k,p in enumerate(frame) :
   store.write( 'Point('+str(k) +') = {' + str(p[0]) +','+ str(p[1])+',0,h};\n')

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
  if argv['boundary'][1] == 'Periodic':
    strc = r'''Physical Line('Periodic_1') = {1};''' + '\n'
    store.write(strc)
    strc = r'''Physical Line('Periodic_2') = {3};''' + '\n'
    store.write(strc)
    strc = 'Periodic Line{1}={-3};\n'
    store.write(strc)
  else:
   if direction=='y':
    strc = r'''Physical Line('Cold') = {3};''' + '\n'
    store.write(strc)
    strc = r'''Physical Line('Hot') = {1};''' +'\n'
    store.write(strc)
   else:
    bs.append(1)
    bs.append(3)
  

  if argv['boundary'][0] =='Periodic':
    strc = r'''Physical Line('Periodic_3') = {2};''' + '\n'
    store.write(strc)
    strc = r'''Physical Line('Periodic_4') = {4};''' +'\n'
    store.write(strc)
    strc = 'Periodic Line{2}={-4};\n'
    store.write(strc)
  else:
   if direction=='x':
    strc = r'''Physical Line('Hot') = {2};''' + '\n'
    store.write(strc)
    strc = r'''Physical Line('Cold') = {4};''' +'\n'
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


  if argv.setdefault('structured',False):
      store.write('Transfinite Surface {10};\n')
      store.write('Recombine Surface {10};\n')

#-------------------------------------------------------
  store.close()
  
  with open(os.devnull, 'w') as devnull:
    output = subprocess.check_output("gmsh -format msh2 -2 mesh.geo -o mesh.msh".split(), stderr=devnull)


class Mesher(object):

 def __init__(self,argv):

  #argv['dmin'] = 0.0
  model = argv.setdefault('model','lattice')

  if model == 'lattice':   
   argv.setdefault('ly',argv['lx'])
   #create polygons-----
   self.add_symmetry(argv) 
   shapes = get_shape(argv)

   scale = [1,argv['ly']/argv['lx']]
   argv['base'] = [[tmp[0]*scale[0],tmp[1]*scale[1]]  for tmp in argv['base']]
   polygons = [translate_shape(shapes[n],i,**argv) for n,i in enumerate(argv['base'])]
   argv.update({'polygons':np.array(polygons,dtype=object)})


   repeat_merge_scale(argv)


  elif model == 'custom':
    repeat_merge_scale(argv)

  elif model =='bulk':

    if argv.setdefault('lz',0) == 0:
     return generate_bulk_2D(argv)
    else :
     return self.generate_bulk_3D(argv)

  elif model == 'random':
      while argv['dmin'] < argv.setdefault('dmin_prescribed',0.02):  
       argv['base'] = np.random.rand(argv.setdefault('Np',5),2)-0.5 

       self.add_symmetry(argv) 
       self.add_polygons(argv)
       repeat_merge_scale(argv)

  elif model =='disk':
    return generate_disk(argv)
  #---------------------
  #if not argv.setdefault('only_geo',False):
  if argv.setdefault('lz',0) == 0:
    self.generate_mesh_2D_new(argv)
  else:   
    self.generate_mesh_3D(argv)


 def generate_bulk_3D(self,argv):


  Lx = argv['lx']
  Ly = argv['ly']
  Lz = argv['lz']

  frame = generate_frame(**argv)

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
   strc = r'''Physical Surface('Periodic_1') = {5} ;''' + '\n'
   store.write(strc)
   strc = r'''Physical Surface('Periodic_2') = {3} ;''' + '\n'
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
   strc = r'''Physical Surface('Periodic_5') = {2} ;''' + '\n'
   store.write(strc)
   strc = r'''Physical Surface('Periodic_6') = {1} ;''' +'\n'
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
  with open(os.devnull, 'w') as devnull:
    output = subprocess.check_output("gmsh -optimize_netgen -format msh2 -3 mesh.geo -o mesh.msh".split(), stderr=devnull)


 def add_polygons(self,argv):

   shapes = get_shape(argv)
   polygons = [translate_shape(shapes[n],i) for n,i in enumerate(argv['base'])]

   argv.update({'polygons':np.array(polygons)})


 def add_symmetry(self,argv):

  if argv.setdefault('invariance',False):
      argv['base'] =  np.append(argv['base'],[[0,0]],axis=0)

  if argv.setdefault('reflection',False):
     
      argv['base']=np.array(argv['base'])
      base = argv['base'].copy()
      tmp = base/2 + np.array([0.25,0.25])
      base = argv['base'].copy()
      tmp =  np.append(tmp,-base/2 - [0.25,0.25],axis=0)
      base = argv['base'].copy()
      base[:,1] = -base[:,1]
      tmp =  np.append(tmp,base/2 - [-0.25,0.25],axis=0)
      base = argv['base'].copy()
      base[:,0] = -base[:,0]
      tmp =  np.append(tmp,base/2 - [0.25,-0.25],axis=0)
      argv['base'] = tmp




 def create_loop(self,loops,line_list,store):

   #create external loop
   strc = 'Line Loop(' + str(loops)+ ') = {'
   for n in range(len(line_list)):
    strc +=str(line_list[n])
    if n == len(line_list)-1:
     strc += '};\n'
    else:
     strc += ','
   store.write(strc)

   return strc


 def generate_mesh_2D_new(self,argv): 

  #Create polygons 
  #polygons = unary_union([Polygon(poly)  for poly in argv['polygons']])
  polygons = [Polygon(poly)  for poly in argv['polygons']]
  polygons_union = unary_union(polygons)

  #Create frame
  frame = Polygon(generate_frame(**argv))

  #Creat heat source--
  heat_source = argv.setdefault('heat_source',len(argv['polygons'])*[None])


  step = argv['step']
  lx = argv['lx']
  ly = argv['ly']
  self.lx = lx
  self.ly = ly

  #Init data
  points = []
  lines = []
  loops = 1000
  ss = 1
  pore_wall = []
  bulk_surface = []
  #inclusions = []
  #---------

  #Start File--
  store = open('mesh.geo', 'w+')
  store.write('h='+str(step) + ';\n')
  store.write('lx='+str(lx) + ';\n')
  store.write('ly='+str(ly) + ';\n')

  #polygons = frame.intersection(polygons)
  heat_source_surface = []
  heat_source_lines = []
  internal_loops = []
  kg = 0
  for n,polygon in enumerate(polygons):

      polygon_intersected = frame.intersection(polygon) 
      pp = list(polygon_intersected.exterior.coords)[:-1]
      cross =  abs(polygon.area - polygon_intersected.area)/polygon.area >1e-4

      if not heat_source[n] == None: #Heat source
         line_list = self.create_line_list(pp,points,lines,store)
         heat_source_lines += line_list
         loops +=1
         self.create_loop(loops,line_list,store)
         self.create_surface_old([loops],ss,store)
         heat_source_surface.append(str(ss))
         strc = r'''Physical Surface('GENERATION_''' + str(heat_source[n])+ "') = {" + str(heat_source_surface[kg]) + '''};\n''';store.write(strc)
         ss += 1
         kg += 1
         if not cross:
          internal_loops.append(loops)

      
      if heat_source[n] == None and not cross: #Internal pore
         line_list = self.create_line_list(pp,points,lines,store)
         loops +=1
         self.create_loop(loops,line_list,store)
         internal_loops.append(loops)


  #----------------------------------------------------
  #Create bulk region--
  diff = frame.difference(polygons_union)
  pp = list(diff.exterior.coords)[:-1]
  line_list = self.create_line_list(pp,points,lines,store)
  loops +=1
  self.create_loop(loops,line_list,store)
  self.create_surface_old([loops] + internal_loops,ss,store)
  #bulk_surface = heat_source_surface + [ss]
  bulk_surface = [ss]
  ss += 1
  #------------------------------------------------------
 
  strc = r'''Physical Surface('Matrix') = {'''
  for n,s in enumerate(bulk_surface):
   strc += str(s)
   if n == len(bulk_surface)-1:
     strc += '};\n'
   else:
     strc += ','
  store.write(strc)

  left = []
  right = []
  upper = []
  lower = []

  pul = np.array([-self.lx/2.0,self.ly/2.0])
  pur = np.array([self.lx/2.0,self.ly/2.0])
  pll = np.array([-self.lx/2.0,-self.ly/2.0])
  plr = np.array([self.lx/2.0,-self.ly/2.0])


  delta = 1e-12
  pore_wall = []
  for l,line in enumerate(lines):
 
   pl = (np.array(points[line[0]])+np.array(points[line[1]]))/2.0
   
   is_on_boundary = True
   if self.compute_line_point_distance(pul,pur,pl) < delta:     
     upper.append(l+1)
     is_on_boundary = False
     
   if self.compute_line_point_distance(pll,plr,pl) < delta:
     lower.append(l+1)
     is_on_boundary = False 
     
   if self.compute_line_point_distance(plr,pur,pl) < delta:
     left.append(l+1)
     is_on_boundary = False 
     
   if self.compute_line_point_distance(pll,pul,pl) < delta:
     right.append(l+1)
     is_on_boundary = False   
     
   if is_on_boundary and (not l+1 in heat_source_lines):
    pore_wall.append(l+1)



  #Setting the temperature---
  boundary_bc = argv.setdefault('boundary',['Periodic','Periodic','Periodic'])
  direction        = argv.setdefault('direction','x')
  additional_boundary = []
  boundary_conditions = {}
  if direction == 'x':     

     #bc along x
     if boundary_bc[0] == 'Isothermal':
       boundary_conditions['ISO_-0.5'] = left
       boundary_conditions['ISO_0.5']  = right
     elif boundary_bc[0] == 'Periodic':
       boundary_conditions['Periodic_1'] = left
       boundary_conditions['Periodic_2'] = right
     elif boundary_bc[0] == 'Diffuse':
       additional_boundary += left + right
     else:
       raise Exception('No applied gradient direction recognized')  

     
     #bc along y
     if boundary_bc[1] == 'Isothermal':
       boundary_conditions['ISO_0.0']   = lower
       boundary_conditions['ISO_0.0']  += upper
     elif boundary_bc[1] == 'Periodic':
       boundary_conditions['Periodic_3'] = upper
       boundary_conditions['Periodic_4'] = lower
     elif boundary_bc[1] == 'Diffuse':
       additional_boundary += upper + lower
     else:
       raise Exception('No applied gradient direction recognized')  


  elif direction == 'y':    

     #bc along x
     if boundary_bc[0] == 'Isothermal':
       boundary_conditions['ISO_0.0']  = left
       boundary_conditions['ISO_0.0']  = right
     elif boundary_bc[0] == 'Periodic':
       boundary_conditions['Periodic_1'] = left
       boundary_conditions['Periodic_2'] = right
     elif boundary_bc[0] == 'Diffuse':
       additional_boundary += left + right
     else:
       raise Exception('No applied gradient direction recognized')  


     #bc along y
     if boundary_bc[1] == 'Isothermal':
       boundary_conditions['ISO_-0.5'] = lower
       boundary_conditions['ISO_0.5']  = upper
     elif boundary_bc[1] == 'Periodic':
       boundary_conditions['Periodic_3'] = upper
       boundary_conditions['Periodic_4'] = lower
     elif boundary_bc[1] == 'Diffuse':
       additional_boundary += lower + upper
     else:
       raise Exception('No applied gradient direction recognized')  

  elif direction == '-1':     
     if boundary_bc[0] == 'Isothermal':
       boundary_conditions['ISO_0.0']  = left
       boundary_conditions['ISO_0.0'] += right
     if boundary_bc[1] == 'Isothermal':
       boundary_conditions['ISO_0.0'] += lower
       boundary_conditions['ISO_0.0'] += upper

  else:
    raise Exception('No applied gradient direction recognized')  
  #-------------------------

  #Write to file
  for key, value in boundary_conditions.items():
   store.write('Physical Line("' + key + '") = {' + ','.join([str(s)  for s in value]) + '};')



  #Thermalizing Boundaries---
  interface_surfaces = []
  boundary_surfaces  =  pore_wall + additional_boundary

  #Interface surface
  if len(boundary_surfaces)> 0:
   strc = r'''Physical Line('Boundary') = {'''
   for r,region in enumerate(boundary_surfaces) :
    strc += str(region)
    if r == len(boundary_surfaces)-1:
     strc += '};\n'
     store.write(strc)
    else :
     strc += ','


  if len(interface_surfaces)> 0:
   strc = r'''Physical Line('Interface') = {'''
   for r,region in enumerate(interface_surfaces) :
    strc += str(region)
    if r == len(interface_surfaces)-1:
     strc += '};\n'
     store.write(strc)
    else :
     strc += ','
  

#-------------------------------------------------------
  store.close()

  with open(os.devnull, 'w') as devnull:
    output = subprocess.check_output("gmsh -format msh2 -2 mesh.geo -o mesh.msh".split(), stderr=devnull)


 def generate_mesh_2D(self,argv): 

  polygons = argv['polygons']

  mesh_ext = argv['step']
  
  heat_source = argv.setdefault('heat_source',len(polygons)*[None])

  #Create the frame
  frame = generate_frame(**argv)
  Frame = Polygon(frame)

  #CREATE BULK SURFACE------------------------------------
  store = open('mesh.geo', 'w+')
  store.write('h='+str(mesh_ext) + ';\n')
  store.write('lx='+str(argv['lx']) + ';\n')
  store.write('ly='+str(argv['ly']) + ';\n')
  points = []
  lines = []
  loops = 1000
  ss = 0

  polypores = []
  for poly in polygons:
   polypores.append(Polygon(poly))

  #Get boundary wall
  self.lx = abs(frame[0][0])*2.0
  self.ly = abs(frame[0][1])*2.0
  pore_wall = []
  delta = 1e-2
  #-------------------------

  bulk = Frame.difference(unary_union(polypores))
  if not (isinstance(bulk, shapely.geometry.multipolygon.MultiPolygon)):
   bulk = [bulk]

  bulk_surface = []
  inclusions = []

  for r,region in enumerate(bulk):
   
    pp = list(region.exterior.coords)[:-1]
    line_list = self.create_line_list(pp,points,lines,store)

    loops +=1
    local_loops = [loops]
    self.create_loop(loops,line_list,store)

    #Create internal loops-------------
    iloop = []
    pore_lines = []
    for interior in region.interiors:
     pp = list(interior.coords)[:-1]
     line_list = self.create_line_list(pp,points,lines,store)
     pore_lines.append(line_list)
     loops +=1
     iloop.append(loops)
     local_loops.append(loops)
     self.create_loop(loops,line_list,store)
    #---------------------------------
    ss +=1
    bulk_surface.append(ss)
    self.create_surface_old(local_loops,ss,store)
    ss +=1

  #Create Inclusion surfaces--------
  #if argv.setdefault('inclusion',False):
  kg = 0
  pore_wall2 = []
  for ng,g in enumerate(heat_source):
      if not g == None:  
         self.create_surface_old([iloop[argv['corr'][ng]]],ss,store)
         strc = r'''Physical Surface('GENERATION_''' + str(kg)+ "') = {" + str(ss) + '''};\n'''
         kg += 1
         store.write(strc)
         ss +=1
      else:   
        pore_wall2 += pore_lines[argv['corr'][ng]]  
  #-------------------------------

  strc = r'''Physical Surface('Matrix') = {'''
  for n,s in enumerate(bulk_surface):
   strc += str(s)
   if n == len(bulk_surface)-1:
     strc += '};\n'
   else:
     strc += ','
  store.write(strc)


  hot = []
  cold = []
  upper = []
  lower = []

  pul = np.array([-self.lx/2.0,self.ly/2.0])
  pur = np.array([self.lx/2.0,self.ly/2.0])
  pll = np.array([-self.lx/2.0,-self.ly/2.0])
  plr = np.array([self.lx/2.0,-self.ly/2.0])


  delta = 1e-12
  pore_wall = []
  for l,line in enumerate(lines):
 
   pl = (np.array(points[line[0]])+np.array(points[line[1]]))/2.0
   
   is_on_boundary = True
   if self.compute_line_point_distance(pul,pur,pl) < delta:     
     upper.append(l+1)
     is_on_boundary = False
     
   if self.compute_line_point_distance(pll,plr,pl) < delta:
     lower.append(l+1)
     is_on_boundary = False 
     
   if self.compute_line_point_distance(plr,pur,pl) < delta:
     cold.append(l+1)
     is_on_boundary = False 
     
   if self.compute_line_point_distance(pll,pul,pl) < delta:
     hot.append(l+1)
     is_on_boundary = False   
     
   if is_on_boundary:
    pore_wall.append(l+1)


  #Rationale: pore_wall trackes all the surfaces/lines that are NOT on the boundary.
  #pore_wall2, on the other side, tracks all the surfaces/lines that are actually pores (as opposed to heat source)
  #In some cases, however, 
  #print(pore_wall)
  #print(pore_wall2)
  #quit()

  additional_boundary = []
  argv.setdefault('Periodic',[True,True,True])
  if argv['Periodic'][0]:
   strc = r'''Physical Line('Periodic_2') = {'''
   for n in range(len(hot)-1):
    strc += str(hot[n]) + ','
   strc += str(hot[-1]) + '};\n'
   store.write(strc)

   strc = r'''Physical Line('Periodic_1') = {'''
   for n in range(len(cold)-1):
    strc += str(cold[n]) + ','
   strc += str(cold[-1]) + '};\n'
   store.write(strc)
  else:
   for k in hot:
    additional_boundary.append(k)
   for k in cold:
    additional_boundary.append(k)

  if argv.setdefault('Periodic',[True,True,True])[1] and len(upper) > 0:
   strc = r'''Physical Line('Periodic_3') = {'''
   for n in range(len(upper)-1):
    strc += str(upper[n]) + ','
   strc += str(upper[-1]) + '};\n'
   store.write(strc)

   strc = r'''Physical Line('Periodic_4') = {'''
   for n in range(len(lower)-1):
    strc += str(lower[n]) + ','
   strc += str(lower[-1]) + '};\n'
   store.write(strc)
  else:
   for k in lower:
    additional_boundary.append(k)
   for k in upper:
    additional_boundary.append(k)

  #Collect Wall
  #if argv.setdefault('inclusion',False):
  # interface_surfaces = pore_wall
  # boundary_surfaces = additional_boundary
  #else:

  interface_surfaces = []
  boundary_surfaces  =  pore_wall2 + additional_boundary

  #Interface surface
  if len(boundary_surfaces)> 0:
   strc = r'''Physical Line('Boundary') = {'''
   for r,region in enumerate(boundary_surfaces) :
    strc += str(region)
    if r == len(boundary_surfaces)-1:
     strc += '};\n'
     store.write(strc)
    else :
     strc += ','


  if len(interface_surfaces)> 0:
   strc = r'''Physical Line('Interface') = {'''
   for r,region in enumerate(interface_surfaces) :
    strc += str(region)
    if r == len(interface_surfaces)-1:
     strc += '};\n'
     store.write(strc)
    else :
     strc += ','

  if argv['Periodic'][0]:
   for n in range(len(hot)):
    periodic = self.FindPeriodic(hot[n],cold,points,lines) 
    if not periodic == -1:
     strc = 'Periodic Line{' + str(hot[n]) + '}={' + str(periodic) + '};\n'
     store.write(strc)

  if argv['Periodic'][1]:
   for n in range(len(lower)):
    periodic = self.FindPeriodic(lower[n],upper,points,lines)
    if not periodic == -1:
     strc = 'Periodic Line{' + str(lower[n]) + '}={' + str(periodic) + '};\n'
     store.write(strc)

#-------------------------------------------------------
  store.close()

  with open(os.devnull, 'w') as devnull:
    output = subprocess.check_output("gmsh -format msh2 -2 mesh.geo -o mesh.msh".split(), stderr=devnull)


 def FindPeriodic(self,control,others,points,lines):
  
  delta = 1e-3
  p1c = np.array(points[lines[control-1][0]])
  p2c = np.array(points[lines[control-1][1]])
  pc = (p1c + p2c)/2.0
 
  for n in range(len(others)):
   p1o = np.array(points[lines[others[n]-1][0]])
   p2o = np.array(points[lines[others[n]-1][1]])
   po = (p1o + p2o)/2.0   
  
  
   if np.linalg.norm(np.dot(p1c-p2c,pc-po)) < delta:     
    line1 = LineString([p1c,p1o])
    line2 = LineString([p2c,p2o])     
    if line1.intersects(line2):
     return -others[n]
    else:
     return others[n]

  #If no periodic line has been found
  print("NO PERIODIC LINES HAS BEEN FOUND")
  return -1

 def create_surface_old(self,ss,bulk_surface,store):

  strc = '''Plane Surface(''' + str(bulk_surface) + ')= {'
  for n,s in enumerate(ss):
   strc += str(s)
   if n == len(ss)-1:
     strc += '};\n'
   else:
     strc += ','
  store.write(strc)


 def compute_line_point_distance(self,p1,p2,p3):
   return np.linalg.norm(np.cross(p2-p1, p3-p1))/np.linalg.norm(p2-p1)


 def generate_mesh_3D(self,argv): 

  self.step = argv['step']
  self.points = []
  self.lines = []
  self.loops = []
  self.surfaces = []
  self.argv = argv
  self.common(z=-self.argv['lz']/2)

  self.common(z=self.argv['lz']/2)

  self.merge() 
  self.points = np.array(self.points)
  self.lines = np.array(self.lines)
  self.loops = np.array(self.loops,dtype=object)
  
  self.apply_periodic_mesh()
 
  self.write_geo()

     
 def get_surface_centroid(self,ss):
     
  
   s = self.surfaces[ss][0]  
    
   pts = []
   for n in self.loops[s]:
    if n > 0:
        i = 0
    else:
        i = 1
    pts.append(self.points[self.lines[abs(n)-1][i]])
   pts = np.array(pts) 
   
   return np.mean(pts,axis=0)
   
     

 def isperiodic(self,s1,s2):

   c1 = self.get_surface_centroid(s1)
   c2 = self.get_surface_centroid(s2)
   
   
   if abs(np.linalg.norm(c1-c2) - self.argv['lx'])<1e-4 and \
      np.linalg.norm(np.cross(c1-c2,[1,0,0])) < 1e-4: 
      return 0,c1-c2
   

   if abs(np.linalg.norm(c1-c2) - self.argv['ly'])<1e-4 and \
      np.linalg.norm(np.cross(c1-c2,[0,1,0])) < 1e-4: 
      return 1,c1-c2
   
   if abs(np.linalg.norm(c1-c2) - self.argv['lz'])<1e-4 and \
      np.linalg.norm(np.cross(c1-c2,[0,0,1])) < 1e-4: 
      return 2,c1-c2
   
  
    
   return None,None
   
 
   
    
 def get_points_from_surface(self,s1):
      
   ll = self.surfaces[s1][0]
   pts = []
   ind = []
   for n in self.loops[ll]:
    if n > 0:
        i = 0
    else:
        i = 1
    pts.append(self.points[self.lines[abs(n)-1][i]])
    
    ind.append(self.lines[abs(n)-1][i])

   return np.array(pts),ind
     
     
 def apply_periodic_mesh(self):

     #Find periodic surfaces----
     self.periodic = {}
     for s1 in range(len(self.surfaces)):
      for s2 in range(s1+1,len(self.surfaces)):
         (a,per) = self.isperiodic(s1,s2)
         if not a == None and self.argv['Boundary'][a]=='Periodic':
           
           corr = {}
           pts1,ind1 = self.get_points_from_surface(s1)
           pts2,ind2 = self.get_points_from_surface(s2)

           for n1,p1 in zip(ind1,pts1):
            for n2,p2 in zip(ind2,pts2):
             if np.linalg.norm(p1-p2-per) < 1e-3:
                corr[n1] = n2
                break            
           #-----------------------------------------
           loop_1 = self.loops[self.surfaces[s1][0]] 
           loop_2 = []
           for ll in loop_1 :
             pts = self.lines[abs(ll)-1]
             if ll > 0:
              p1 = pts[0]; p2 = pts[1] 
             else:
              p1 = pts[1]; p2 = pts[0]   
             loop_2.append(self.line_exists_ordered(corr[p1],corr[p2]))             
           self.periodic[s1] = [s2,loop_1,loop_2,np.array(per/np.linalg.norm(per))]  
           
           break  
     
      
        
       

 def merge(self):

   #Create lines
   Nh = int(len(self.points)/2)
   for n in range(Nh):
     self.lines.append([n,n+Nh])  

   Ns = int(len(self.surfaces)/2)
   self.surfaces3D = []
   for s in range(Ns):
    tmp2 = [s,s+Ns]  

    for loop in self.surfaces[s]:

     #Create vertical loop    
     for l in self.loops[loop]:
      i1 = self.lines[l-1][0];  i2 = self.lines[l-1][1]
      j1 = i1 + Nh; j2 = i2 + Nh
      tmp = []
      tmp.append(self.line_exists_ordered(i1,i2))
      tmp.append(self.line_exists_ordered(i2,j2))
      tmp.append(self.line_exists_ordered(j2,j1))
      tmp.append(self.line_exists_ordered(j1,i1))
      self.loops.append(tmp)
      self.surfaces.append([len(self.loops)-1])
      tmp2.append(len(self.surfaces)-1)

 
    self.surfaces3D.append(tmp2)

 
 #def apply_periodic_mesh(self):

 def write_geo(self):

   
   store = open('mesh.geo', 'w+')

   
   #Write points-----
   for p,point in enumerate(self.points):
    store.write( 'Point('+str(p) +') = {' + str(point[0]) +','+ str(point[1])+',' + str(point[2])+','+ str(self.step) +'};\n')
   
   #Write lines----
   for l,line in enumerate(self.lines):
     store.write( 'Line('+str(l+1) +') = {' + str(line[0]) +','+ str(line[1])+'};\n')
      
   #Write loops------
   nl = len(self.lines)+1
   for ll,loop in enumerate(self.loops):  
    strc = 'Line Loop(' + str(ll+nl)+ ') = {'
    for n,l in enumerate(loop):
     strc +=str(l)
     if n == len(loop)-1:
      strc += '};\n'
     else:
      strc += ','
    store.write(strc)
   
   
   #Write surface-----
   nll = len(self.loops)
   for s,surface in enumerate(self.surfaces):
    strc = '''Plane Surface(''' +str(s+nll+nl) + ''')= {''' 
    for n,ll in enumerate(surface):
      strc += str(ll+nl)  
      if n == len(surface)-1:
       strc += '};\n'
      else:
       strc += ','    
    store.write(strc)
    

   boundary = list(range(len(self.surfaces)))
   #Apply periodicity
   vec = {}

   for key, value in self.periodic.items():
   
     boundary.remove(key)  
     boundary.remove(value[0])  

     value[3] = np.round(value[3],4)     
     store.write('Periodic Surface ' + str(key+nl+nll) + ' {') 
     
     for n,k in enumerate(value[1]):
      store.write(str(k))
      if n < len(value[1])-1:
       store.write(',')
     store.write('} = ' + str(value[0]+nl+nll) + ' {')
     
     for n,k in enumerate(value[2]):
      store.write(str(k))
      if n < len(value[2])-1:
       store.write(',')
     store.write('};\n')
    
     value[0] += nl+nll
     tt = key +  nl + nll
     if np.array_equal(np.array([1,0,0]),value[3]) : 
        vec.setdefault(2,[]).append(value[0]) 
        vec.setdefault(1,[]).append(tt) 
     
     if np.array_equal(np.array([-1,0,0]),value[3]): 
        vec.setdefault(1,[]).append(value[0]) 
        vec.setdefault(2,[]).append(tt) 
        
     if np.array_equal(np.array([0,1,0]),value[3]) : 
        vec.setdefault(3,[]).append(tt) 
        vec.setdefault(4,[]).append(value[0])
        
     if np.array_equal(np.array([0,-1,0]),value[3]) : 
        vec.setdefault(4,[]).append(tt) 
        vec.setdefault(3,[]).append(value[0])
        

     if np.array_equal(np.array([0,0,1]),value[3]) : 
        vec.setdefault(5,[]).append(value[0]) 
        vec.setdefault(6,[]).append(tt)   
        
     if np.array_equal(np.array([0,0,-1]),value[3]) : 
        vec.setdefault(6,[]).append(value[0]) 
        vec.setdefault(5,[]).append(tt) 
   for key, value in vec.items():
     strc = r'''Physical Surface("Periodic_''' + str(key) + '''") = {'''
     for n,p in enumerate(value) :
      strc +=str(p)
      if n == len(value)-1:
       strc += '};\n'
      else :
       strc += ','
     store.write(strc)

   strc = r'''Physical Surface("Boundary") = {'''
   for n,p in enumerate(boundary) :
      strc +=str(p+nl+nll)
      if n == len(boundary)-1:
       strc += '};\n'
      else :
       strc += ','
   store.write(strc)
    
  
   #this part needs to change---

   index = nll+nl +  len(self.surfaces3D)
   for k,surfaces in enumerate(self.surfaces3D):
       
    #------------------
    strc = r'''Surface Loop(''' + str(nll+nl + k) + ') = {'
    for s,surface in enumerate(surfaces) :
     strc +=str(nll+nl+surface)
     if s == len(surfaces)-1:
      strc += '};\n'
     else :
      strc += ','
    store.write(strc)
    
    strc = r'''Volume(''' + str(index + k) + ''') = {''' + str(nll+nl+k) + '};\n'
    store.write(strc)
  
   strc = r'''Physical Volume('Bulk') = {'''
   for k in range(len(self.surfaces3D)) :
     strc +=str(k+index)
     if k == len(self.surfaces3D)-1:
      strc += '};\n'
     else :
      strc += ','
   store.write(strc)

   #---------------------------------------------
    
   store.close()


   with open(os.devnull, 'w') as devnull:
    output = subprocess.check_output("gmsh -optimize_netgen -format msh2 -3 mesh.geo -o mesh.msh".split(), stderr=devnull)
    #output = subprocess.check_output("gmsh -format msh2 -3 mesh.geo -o mesh.msh".split(), stderr=devnull)


 def line_exists_ordered(self,p1,p2):
     
  l = [p1,p2]   
  for n,line in enumerate(self.lines) :
   if (line[0] == l[0] and line[1] == l[1]) :
    return n+1
   if (line[0] == l[1] and line[1] == l[0]) :
    return -(n+1)
  return 0

 def already_included(self,p):

  for n,point in enumerate(self.points):
   d = np.linalg.norm(np.array(point)-np.array(p))
   if d < 1e-12:
    return n
  return -1


 def already_included_old(self,all_points,new_point,p_list):


  dd = 1e-12
  I = []
  if len(all_points) > 0:
   I = np.where(np.linalg.norm(np.array(new_point)-np.array(all_points),axis=1)<dd)[0]
  if len(I) > 0: 
     return I[0]
  else: 
     return -1



 # return loop
 def line_exists_ordered_old(self,l,lines):
  for n,line in enumerate(lines) :
   if (line[0] == l[0] and line[1] == l[1]) :
    return n+1
   if (line[0] == l[1] and line[1] == l[0]) :
    return -(n+1)
  return 0

 def create_line_list(self,pp,points,lines,store):

   p_list = []
   for k,p in enumerate(pp):
    tmp = self.already_included_old(points,p,p_list)
    if tmp == -1:    
      points.append(p)
      p_list.append(len(points)-1)
      point  = p
      store.write( 'Point('+str(len(points)-len(p_list)+k) +') = {' + str(point[0]/self.lx) +'*lx,'+ str(point[1]/self.ly)+'*ly,0,h};\n')
    else:
      p_list.append(tmp)

   #for k,p in enumerate(p_list):
   #  point = points[-len(p_list)+k] 
   #  store.write( 'Point('+str(len(points)-len(p_list)+k) +') = {' + str(point[0]/self.lx) +'*lx,'+ str(point[1]/self.ly)+'*ly,0,h};\n')

   line_list = []
   for l in range(len(p_list)):
    p1 = p_list[l]
    p2 = p_list[(l+1)%len(p_list)]
    if not p1 == p2:
     tmp = self.line_exists_ordered_old([p1,p2],lines)

     if tmp == 0 : #create line
      lines.append([p1,p2])
      store.write( 'Line('+str(len(lines)) +') = {' + str(p1) +','+ str(p2)+'};\n')
      line_list.append(len(lines))
     else:
      line_list.append(tmp)

   
   return line_list


 


 def create_lines_and_loop_from_points(self,pp,z):
     
   p_list = []  
   for p in pp:
    p = [p[0],p[1],z]   
    tmp = self.already_included(p) 
    if tmp == -1:
      self.points.append(p)
      p_list.append(len(self.points)-1)
    else:
      p_list.append(tmp)

   line_list = []
   for l in range(len(p_list)):
    p1 = p_list[l]
    p2 = p_list[(l+1)%len(p_list)]
    if not p1 == p2:
     tmp = self.line_exists_ordered(p1,p2)
     if tmp == 0 : #craete line
      self.lines.append([p1,p2])
      line_list.append(len(self.lines))
     else:
      line_list.append(tmp)
      
   self.loops.append(line_list)  
      

 def common(self,z):

  Frame = Polygon(generate_frame(**self.argv))
  polygons = self.argv['polygons']

  polypores = [ Polygon(poly)  for poly in polygons]
  
  MP = MultiPolygon(polypores)
  bulk = Frame.difference(unary_union(MP))
  if not (isinstance(bulk, shapely.geometry.multipolygon.MultiPolygon)):
   bulk = [bulk]

  for region in bulk:
   loop_start = len(self.loops)  
   pp = list(region.exterior.coords)[:-1]
   self.create_lines_and_loop_from_points(pp,z) 
   #From pores 
   for interior in region.interiors:
    pp = list(interior.coords)[:-1]
    self.create_lines_and_loop_from_points(pp,z)
   loop_end = len(self.loops) 

   self.surfaces.append(range(loop_start,loop_end))

    







 

  




