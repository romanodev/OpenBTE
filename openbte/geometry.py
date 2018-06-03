from __future__ import print_function
from __future__ import absolute_import
import os,sys
import numpy as np
import subprocess
from mpi4py import MPI
from pyvtk import *
from .GenerateSquareLatticePores import *
from .GenerateHexagonalLatticePores import *
#from .GenerateCustomPores import *
#from GenerateRandomPores import *
#from GenerateRandomPoresOverlap import *
from . import GenerateMesh2D 
from . import GenerateMesh3D 
from . import GenerateBulk2D 
from . import GenerateBulk3D 
#import GenerateInterface2D 
#from nanowire import *
import deepdish as dd
from scipy.sparse import csc_matrix

class Geometry(object):

 def __init__(self,**argv):


  direction = argv.setdefault('direction','x')
  if direction == 'x':self.direction = 0
  if direction == 'y':self.direction = 1
  if direction == 'z':self.direction = 2


  geo_type = argv['type']

  if geo_type == 'load':
   self.state = dd.io.load(argv.setdefault('filename','geometry')+ '.hdf5')
   self._update_data()
  else:
   if MPI.COMM_WORLD.Get_rank() == 0:
    #porous-----
    #if geo_type == 'porous/random' or \
    if  geo_type == 'porous/square_lattice' or\
        geo_type == 'porous/hexagonal_lattice' or\
        geo_type == 'porous/custom':

     if geo_type == 'porous/square_lattice':
      frame,polygons = GenerateSquareLatticePores(argv) 
   
     if geo_type == 'porous/hexagonal_lattice':
      frame,polygons = GenerateHexagonalLatticePores(argv) 

     #if geo_type == 'porous/custom':
     # frame,polygons = GenerateCustomPores(argv) 

     #if geo_type == 'porous/random':
     # frame,polygons = GenerateRandomPoresOverlap(argv) 
     # polygons = np.array(polygons)

     if 'lz' in argv.keys():
      GenerateMesh3D.mesh(polygons,frame,argv)
      self.dim = 3
     else:
      GenerateMesh2D.mesh(polygons,frame,argv)
      self.dim = 2

    #if argv['model'] == 'nanowire':
    # nanowire(argv)
    # self.dim = 3
    #Interface----EXPERIMENTAL
    #if argv['model'] == '2DInterface':
    #  GenerateInterface2D.mesh(argv)
    #  self.dim = 2

    #bulk----------
    if geo_type == 'bulk':
     if 'lz' in argv.keys() == 3:
      GenerateBulk3D.mesh(argv)
      self.dim = 3
     else:
      GenerateBulk2D.mesh(argv)
      self.dim = 2
     #-----------------------------------
    data = self.compute_mesh_data()
    dd.io.save('geometry.hdf5', data)
  MPI.COMM_WORLD.Barrier()


 def compute_mesh_data(self):

    
    self.import_mesh()

    self.compute_elem_map()
    self.compute_elem_volumes()
    self.compute_side_areas()
    self.compute_side_normals()
    self.compute_elem_centroids()
    self.compute_side_centroids()
    self.compute_least_square_weigths()
    self.compute_connecting_matrix()
    self.compute_interpolation_weigths()
    self.compute_contact_areas()
    self.compute_boundary_condition_data()
    

    data = {'side_list':self.side_list,\
          'exlude':self.exlude,\
          'elem_side_map':self.elem_side_map,\
          'side_elem_map':self.side_elem_map,\
          'side_node_map':self.side_node_map,\
          'node_elem_map':self.node_elem_map,\
          'elem_map':self.elem_map,\
          'nodes':self.nodes,\
          'sides':self.sides,\
          'elems':self.elems,\
          'A':self.A,\
          'side_periodicity':self.side_periodicity,\
          'dim':self.dim,\
          'weigths':self.weigths,\
          'size':self.size,\
          'c_areas':self.c_areas,\
          'interp_weigths':self.interp_weigths,\
          'elem_centroids':self.elem_centroids,\
          'side_centroids':self.side_centroids,\
          'elem_volumes':self.elem_volumes,\
          'periodic_nodes':self.periodic_nodes,\
          'side_areas':self.side_areas,\
          'pairs':self.pairs,\
          'boundary_elements':self.boundary_elements,\
          'side_normals':self.side_normals,\
          'grad_direction':self.direction,\
          'area_flux':self.area_flux,\
          'flux_sides':self.flux_sides,\
          'side_periodic_value':self.side_periodic_value}

    return data

 def compute_connecting_matrix(self):

   nc = len(self.elems)
   row_tmp = []
   col_tmp = []
   data_tmp = [] 
   data_tmp_b = [] 

   for ll in self.side_list['active'] :
    if not ll in self.side_list['Boundary'] and \
       not ll in self.side_list['Hot'] and\
       not ll in self.side_list['Cold']:

     elems = self.get_elems_from_side(ll)
     kc1 = elems[0]
     kc2 = elems[1]  
     row_tmp.append(kc1)
     col_tmp.append(kc2)
     data_tmp.append(1)
     row_tmp.append(kc2)
     col_tmp.append(kc1)
     data_tmp.append(1)
  
   self.A = csc_matrix( (np.array(data_tmp),(np.array(row_tmp),np.array(col_tmp))), shape=(nc,nc) ).todense()


 def compute_elem_map(self):

  self.elem_map = {}
  for elem1 in self.elem_side_map:
    for side in self.elem_side_map[elem1]:
     elem2 = self.get_neighbor_elem(elem1,side)
     self.elem_map.setdefault(elem1,[]).append(elem2)


 def get_side_orthognal_direction(self,side):

   elem = self.side_elem_map[side][0]
   c1 = self.get_elem_centroid(elem)
   c2 = self.get_side_centroid(side)
   normal = self.compute_side_normal(elem,side)
   area = self.compute_side_area(side)
   Af = area*normal
   dist = c2-c1
   v_orth = np.dot(Af,Af)/np.dot(Af,dist)
   return v_orth


 def get_decomposed_directions(self,elem_1,elem_2,rot = np.eye(3)):

   side = self.get_side_between_two_elements(elem_1,elem_2)  
   normal = self.compute_side_normal(elem_1,side)
   area = self.compute_side_area(side)
   Af = area*normal
   c1 = self.get_elem_centroid(elem_1)
   c2 = self.get_next_elem_centroid(elem_1,side)
   dist = c2-c1
   v_orth = np.dot(Af,np.dot(rot,Af))/np.dot(Af,dist)
   v_non_orth = np.dot(rot,Af) - dist*v_orth
   return v_orth,v_non_orth


 def get_side_coeff(self,side):

  elem = self.side_elem_map[side][0]
  vol = self.get_elem_volume(elem)
  normal = self.compute_side_normal(elem,side)
  area = self.get_side_area(side)
  return normal*area/vol,elem #This is specific to boundaries

 def get_coeff(self,elem_1,elem_2):
   
   side = self.get_side_between_two_elements(elem_1,elem_2)  
   vol = self.get_elem_volume(elem_1)  
   normal = self.compute_side_normal(elem_1,side)
   area = self.compute_side_area(side)

    
   return normal*area/vol

 def get_af(self,elem_1,elem_2):

   side = self.get_side_between_two_elements(elem_1,elem_2)  
   normal = self.compute_side_normal(elem_1,side)
   area = self.compute_side_area(side)
   return normal*area


 def get_normal_between_elems(self,elem_1,elem_2):

   side = self.get_side_between_two_elements(elem_1,elem_2)  
   normal = self.compute_side_normal(elem_1,side)
   return normal


 def get_side_between_two_elements(self,elem_1,elem_2):   

    for side_1 in self.elem_side_map[elem_1]:
     for side_2 in self.elem_side_map[elem_2]: 
      if side_1 == side_2:
       return side_1

    print('no adjicents elems')
    quit()



 def compute_gradient_on_side(self,x,ll,side_periodic_value):

   #Diff in the temp
   normal = self.get_side_normal(0,ll)
   elems = self.side_elem_map[ll]
   kc1 = elems[0]
   temp_1 = x[kc1]
   #Build tempreature matrix--------------------------
   diff_temp = np.zeros(self.dim + 1)
   grad = np.zeros(3)
   for s in self.elem_side_map[kc1]: 
    if s in self.side_list['Boundary']:
     temp_2 = temp_1
    else:
     kc2 = self.get_neighbor_elem(kc1,s)
     temp_2 = x[kc2] + self.get_side_periodic_value(s,kc2,side_periodic_value)

    ind1 = self.elem_side_map[kc1].index(s)
    diff_t = temp_2 - temp_1
    diff_temp[ind1] = diff_t
    #-----------------------------------------------
    #COMPUTE GRADIENT
   tmp = np.dot(self.weigths[kc1],diff_temp)
   grad[0] = tmp[0] #THESE HAS TO BE POSITIVE
   grad[1] = tmp[1]
   if self.dim == 3:
    grad[2] = 0.0

   return grad



 def compute_thermal_conductivity_from_scalar(self,x,flux_sides,side_periodic_value,factor = np.eye(3)):

   #Diff in the temp
   kappa = 0.0
   for ll in flux_sides :
    normal = self.get_side_normal(0,ll)
    area_loc = self.get_side_area(ll)
    elems = self.side_elem_map[ll]
    kc1 = elems[0]
    temp_1 = x[kc1]
    #Build tempreature matrix--------------------------
    diff_temp = np.zeros(self.dim + 1)
    grad = np.zeros(self.dim)
    for s in self.elem_side_map[kc1]: 
     if s in self.side_list['Boundary']:
      temp_2 = temp_2
     else:
      kc2 = self.get_neighbor_elem(kc1,s)
      temp_2 = x[kc2] + self.get_side_periodic_value(s,kc2,side_periodic_value)

     ind1 = self.elem_side_map[kc1].index(s)
     diff_t = temp_2 - temp_1
     diff_temp[ind1] = diff_t
    #-----------------------------------------------
    #COMPUTE GRADIENT
    tmp = np.dot(self.weigths[kc1],diff_temp)
    grad[0] = tmp[0] #THESE HAS TO BE POSITIVE
    grad[1] = tmp[1]
    if self.dim == 3:
     grad[2] = tmp[2]

    #COMPUTE THERMAL CONDUCTIVITY------------------
    for i in range(self.dim):
     for j in range(self.dim):
      kappa -= normal[i]*factor[i][j]*grad[j]*area_loc

   return kappa


 def compute_thermal_conductivity_outer(self,vector,scalar,flux_sides):

   #Here we simply invert Fourier's law
   factor = np.eye(3)
   power = 0.0
   for ns in flux_sides:
    ne = self.side_elem_map[ns][0]
    normal = self.get_side_normal(0,ns)
    area_loc = self.get_side_area(ns)
    power_loc = 0.0
    for i in range(self.dim):
     for j in range(self.dim):
      power_loc -= normal[i]*factor[i][j]*vector[j]*scalar[ne]*area_loc
    power += power_loc
   return power

 def compute_thermal_conductivity(self,grad,flux_sides,factor=np.eye(3)):

   #Here we simply invert Fourier's law
   power = 0.0
   for ns in flux_sides:
    ne = self.side_elem_map[ns][0]
    normal = self.get_side_normal(0,ns)
    area_loc = self.get_side_area(ns)
    power_loc = 0.0
    for i in range(3):
     for j in range(3):
      power_loc -= normal[i]* factor[i][j] * grad[ne][j]*area_loc
    power += power_loc

   return power



 def compute_average_temperature(self,x,flux_sides):

   #Here we simply invert Fourier's law
   temp = 0.0
   for ns in flux_sides:
    ne = self.side_elem_map[ns][0]
    area_loc = self.get_side_area(ns)
    temp += area_loc * x[ne]

   return temp


  


 def compute_contact_areas(self):

  self.c_areas = np.zeros(3)  

  nodesT = self.nodes.copy().T

  minx = min(nodesT[0])
  maxx = max(nodesT[0])
  miny = min(nodesT[1])
  maxy = max(nodesT[1])
  minz = min(nodesT[2])
  maxz = max(nodesT[2])
  if self.dim == 3:
   self.c_areas[0] = (maxy - miny) * (maxz - minz)
   self.c_areas[1] = (maxz - minz) * (maxx - minx)
   self.c_areas[2] = (maxy - miny) * (maxx - minx)
  else :
   self.c_areas[0] = maxy - miny
   self.c_areas[1] = maxx - minx
   self.c_areas[2] = 0


 def get_side_area(self,side):
  return self.side_areas[side]

 def get_elem_volume(self,elem):
  return self.elem_volumes[elem]

 def compute_elem_volumes(self):

  self.elem_volumes = np.zeros(len(self.elems))
  for k in range(len(self.elems)):
   self.elem_volumes[k] = self.compute_elem_volume(k)




 def compute_side_areas(self):
  self.side_areas = np.zeros(len(self.sides))
  for k in range(len(self.sides)):
   self.side_areas[k] = self.compute_side_area(k)

 
 def get_dim(self):
   return self.dim

 def _update_data(self):   
    self.nodes = self.state['nodes']
    self.dim = self.state['dim']
    self.A = self.state['A']
    self.sides = self.state['sides']
    self.elems = self.state['elems']
    self.side_periodicity = self.state['side_periodicity']
    self.size = self.state['size']
    self.dim = self.state['dim']
    self.weigths = self.state['weigths']
    self.elem_side_map = self.state['elem_side_map']
    self.side_node_map = self.state['side_node_map']
    self.node_elem_map = self.state['node_elem_map']
    self.side_elem_map = self.state['side_elem_map']
    self.side_list = self.state['side_list']
    self.interp_weigths = self.state['interp_weigths']
    self.elem_centroids = self.state['elem_centroids']
    self.c_areas = self.state['c_areas']
    self.side_centroids = self.state['side_centroids']
    self.periodic_nodes = self.state['periodic_nodes']
    self.elem_volumes = self.state['elem_volumes']
    self.boundary_elements = self.state['boundary_elements']
    self.side_normals = self.state['side_normals']
    self.side_areas = self.state['side_areas']
    self.exlude = self.state['exlude']     
    self.pairs = self.state['pairs']
    self.direction = self.state['grad_direction']
    self.area_flux = self.state['area_flux']
    self.flux_sides = self.state['flux_sides']
    self.side_periodic_value = self.state['side_periodic_value']
    self.kappa_factor = self.size[self.direction]/self.area_flux

    

 def import_mesh(self):

  #Create mesh---
  a=subprocess.check_output(['gmsh','-' + str(self.dim),'mesh.geo','-o','mesh.msh'])

  #-------------

  f = open('mesh.msh','r')
  #Read boundary conditions------------------
  f.readline()
  f.readline()
  f.readline()
  f.readline()
  blabels = {}
  nb =  int(f.readline())-1
  for n in range(nb):
   tmp = f.readline().split()
   l = tmp[2].replace('"',r'')
   blabels.update({int(tmp[1]):l})
  #------------------------------------------

  #import nodes------------------------
  f.readline()
  f.readline()
  f.readline()
  n_nodes = int(f.readline())
  nodes = []
  for n in range(n_nodes):
   tmp = f.readline().split()
   nodes.append([float(tmp[1]),float(tmp[2]),float(tmp[3])])
   
  nodes = np.array(nodes) 

  if self.dim == 3:
   self.size = np.array([ max(nodes[:,0]) - min(nodes[:,0]),\
                  max(nodes[:,1]) - min(nodes[:,1]),\
                  max(nodes[:,2]) - min(nodes[:,2])])
  else:
   self.size = np.array([ max(nodes[:,0]) - min(nodes[:,0]),\
                  max(nodes[:,1]) - min(nodes[:,1]),0])

  self.nodes = nodes
  #import elements and create map
  self.side_elem_map = {}
  self.elem_side_map = {}
  f.readline()
  f.readline()
  n_tot = int(f.readline())
  self.sides = []
  self.elems = []
  b_sides = []
  nr = []

  self.node_side_map = {}
  self.side_node_map = {}
  self.node_elem_map = {}
  self.side_list = {}
  node_side_map = {}
  for n in range(n_tot):
   tmp = f.readline().split()
   #Get sides------------------------------------------------------------   
   if self.dim == 3 and int(tmp[1]) == 2: #2D area
     n = sorted([int(tmp[5])-1,int(tmp[6])-1,int(tmp[7])-1])

   if self.dim == 2 and int(tmp[1]) == 1: #1D area
     n = sorted([int(tmp[5])-1,int(tmp[6])-1])

   b_sides.append(n)
   nr.append(int(tmp[3]))
   
   if self.dim == 3 and int(tmp[1]) == 4: #3D Elem
     node_indexes = [int(tmp[5])-1,int(tmp[6])-1,int(tmp[7])-1,int(tmp[8])-1]
     n = sorted(node_indexes)
     self.elems.append(n)
     perm_n =[[n[0],n[1],n[2]],\
             [n[0],n[1],n[3]],\
             [n[1],n[2],n[3]],\
             [n[0],n[2],n[3]]]
     self._update_map(perm_n,b_sides,blabels,nr,node_indexes)

   
   if self.dim == 2 and int(tmp[1]) == 2: #2D Elem
     node_indexes = [int(tmp[5])-1,int(tmp[6])-1,int(tmp[7])-1]
     n = sorted(node_indexes)
     self.elems.append(n)
     perm_n =[[n[0],n[1]],\
             [n[0],n[2]],\
             [n[1],n[2]]]
     self._update_map(perm_n,b_sides,blabels,nr,node_indexes)

   if self.dim == 2 and int(tmp[1]) == 3: #2D Elem
     n = [int(tmp[5])-1,int(tmp[6])-1,int(tmp[7])-1,int(tmp[8])-1]

     #n = sorted(node_indexes)
     self.elems.append(n)
     perm_n =[ sorted([n[0],n[1]]),\
              sorted([n[1],n[2]]),\
              sorted([n[2],n[3]]),\
              sorted([n[3],n[0]])]
     self._update_map(perm_n,b_sides,blabels,nr,n)


  #------------------------------------------------------------
  #Set default for hot and cold
  self.side_list.setdefault('Hot',[])
  self.side_list.setdefault('Cold',[])

  #Apply Periodic Boundary Conditions
  self.side_list.update({'active':range(len(self.sides))})  
  self.side_periodicity = np.zeros((len(self.sides),2,3))
  group_1 = []
  group_2 = []
  self.exlude = []

  self.pairs = [] #global (all periodic pairs)

  self.side_list.setdefault('Boundary',[])
  self.periodic_nodes = [] 
  for label in self.side_list.keys():

   #Add Cold and Hot to Boundary
   #if label == "Hot" or label == "Cold":
   # tmp = self.side_list.setdefault('Boundary',[])+self.side_list[label]
   # self.side_list['Boundary'] = tmp 

   
   if str(label.split('_')[0]) == 'Periodic':
    if not int(label.split('_')[1])%2==0: 
     contact_1 = label
     contact_2 = 'Periodic_' + str(int(label.split('_')[1])+1)
     group_1 = self.side_list[contact_1]
     group_2 = self.side_list[contact_2]
     pairs = []
     
     #compute tangential unity vector
     tmp = self.nodes[self.sides[group_2[0]][0]] - self.nodes[self.sides[group_2[0]][1]]
     t = tmp/np.linalg.norm(tmp)
     n = len(group_1)
     for s1 in group_1:
      d_min = 1e6
      for s in group_2:
       c1 = self.compute_side_centroid(s1)
       c2 = self.compute_side_centroid(s)
       d = np.linalg.norm(c2-c1)
       if d < d_min:
        d_min = d
        pp = c1-c2
        s2 = s

      #if abs(np.dot(pp,self.nodes[self.sides[s1][0]]-self.nodes[self.sides[s1][1]]))>1e-4:
      pairs.append([s1,s2])   
      self.side_periodicity[s1][1] = pp
      self.side_periodicity[s2][1] = -pp

      if np.linalg.norm(self.nodes[self.sides[s1][0]] - self.nodes[self.sides[s2][0]]) == np.linalg.norm(pp):
       self.periodic_nodes.append([self.sides[s1][0],self.sides[s2][0]])
       self.periodic_nodes.append([self.sides[s1][1],self.sides[s2][1]])
      else:
       self.periodic_nodes.append([self.sides[s1][0],self.sides[s2][1]])
       self.periodic_nodes.append([self.sides[s1][1],self.sides[s2][0]])


     plot_sides = False
     if plot_sides:
      for s in pairs: 
       c1 = self.compute_side_centroid(s[0])
       c2 = self.compute_side_centroid(s[1])
       plot([c1[0],c2[0]],[c1[1],c2[1]],color='r')
      show()
     #Amend map
     for s in pairs: 
      s1 = s[0]
      s2 = s[1]
   
      #Change side in elem 2
      elem2 = self.side_elem_map[s2][0]
      index = self.elem_side_map[elem2].index(s2)
      self.elem_side_map[elem2][index] = s1
      self.side_elem_map[s1].append(elem2)

      #To get hflux sides right-----
      self.side_elem_map[s2].append(self.side_elem_map[s1][0])      

      #-------------------------
 
      #Delete s2 from active list
      self.side_list['active'].remove(s2)
      self.exlude.append(s2)
      tmp = self.side_list.setdefault('Inactive',[]) + [s2]
      self.side_list['Inactive'] = tmp 

     #Polish sides
     tmp = self.side_list.setdefault('Periodic',[])+self.side_list[contact_1]
     self.side_list['Periodic'] = tmp 


     del self.side_list[contact_1]
     del self.side_list[contact_2]
     self.pairs += pairs 

  #Create boundary_elements--------------------
  self.boundary_elements = []
  for ll in self.side_list['Boundary']:
   self.boundary_elements.append(self.side_elem_map[ll][0])

  #delete MESH-----
  #a=subprocess.check_output(['rm','-f','mesh.geo'])
  #a=subprocess.check_output(['rm','-f','mesh.msh'])
  
  #----------------

 
 def get_elems_from_side(self,ll):

  return self.side_elem_map[ll]


 def _update_map(self,perm_n,b_sides,blabels,nr,node_indexes):


    #UPDATE NODE_ELEM_MAP-------------------
    for n in node_indexes:
     self.node_elem_map.setdefault(n,[]).append(len(self.elems)-1)
    #---------------------------------------

    for new_n in perm_n :
     #Use the node_side_map---------
     index = -1
     sides = []
     for k in new_n:
      if index == -1:
       if k in self.node_side_map.keys():
        for s in self.node_side_map[k]:
         if np.allclose(self.sides[s],new_n):
          index = s 
          break

     if index == -1:
      self.sides.append(new_n)
      index = len(self.sides)-1

      #add boundary--------------------------------------
      try:
       k = b_sides.index(new_n)
     

       self.side_list.setdefault(blabels[nr[k]],[]).append(index)
      except:
       a = 1
      #------------------------------------------------------
      for i in range(len(new_n)):
       self.node_side_map.setdefault(new_n[i],[]).append(index)
       self.side_node_map.setdefault(index,[]).append(new_n[i])

     #------------------------------------
     self.side_elem_map.setdefault(index,[]).append(len(self.elems)-1)
     self.elem_side_map.setdefault(len(self.elems)-1,[]).append(index)





 def get_elem_boundary_area_normal(self,elem) :

    sides = self.elem_side_map[elem]
    is_here = 0
    for s in sides:
     if s in self.side_list['Boundary'] :
      normal = self.compute_side_normal(elem,s)
      is_here = 1
      break
    if is_here == 0 : print('ERROR: side is not a boundary')
    return normal


 def compute_side_normals(self):

  self.side_normals = np.zeros((len(self.sides),2,3))
  for s in range(len(self.sides)):
   elems = self.side_elem_map[s]
   self.side_normals[s][0] = self.compute_side_normal(elems[0],s)
   if len(elems)>1:
    self.side_normals[s][1] = self.compute_side_normal(elems[1],s)

 def compute_side_normal(self,ne,ns):

  #Get generic normal--------------------
  v1 = self.nodes[self.sides[ns][1]]-self.nodes[self.sides[ns][0]]
  if self.dim == 3:
   v2 = self.nodes[self.sides[ns][2]]-self.nodes[self.sides[ns][1]]
  else :
   v2 = np.array([0,0,1])
  v = np.cross(v1,v2)
  normal = v/np.linalg.norm(v)
  #-------------------------------------

  #Check sign
  ind = self.side_elem_map[ns].index(ne)
  c_el   = self.compute_centroid(self.elems[ne])
  c_side = self.compute_centroid(self.sides[ns]) -  self.side_periodicity[ns][ind]

  c = (c_side - c_el)
  if np.dot(normal,c) < 0: normal = - normal
 

  return normal

 def get_n_elems(self):
  return len(self.elems)

 def compute_side_area(self,ns):

  p = self.nodes[self.sides[ns]]
  if self.dim == 2:
    return np.linalg.norm(p[1]-p[0])
  else:
   v = np.cross(p[1]-p[0],p[1]-p[2])
   normal = v/np.linalg.norm(v)
   tmp = np.zeros(3)
   for i in range(len(p)):
    vi1 = p[i]
    vi2 = p[(i+1)%len(p)]
    tmp += np.cross(vi1, vi2)

   result = np.dot(tmp,normal)
   return abs(result/2)


 def write_vtk(self,file_name,data) : 

   points = self.nodes
   el = self.elems
   cells = []
   for k in range(len(el)):
    if self.dim == 3:
     cells.append(el[k][0:4])
    else:
     cells.append(el[k][0:3])

   if self.dim == 3:
    vtk = VtkData(UnstructuredGrid(points,tetra=cells),data)
   else:
    vtk = VtkData(UnstructuredGrid(points,triangle=cells),data)
   vtk.tofile(file_name,'ascii')


 def get_side_normal(self,ne,ns):
  return self.side_normals[ns,ne]


 def get_elem_centroid(self,ne):
  return self.elem_centroids[ne]

 def get_side_centroid(self,ns):
  return self.side_centroids[ns]



 def compute_side_centroids(self):

  self.side_centroids = np.zeros((len(self.sides),3))
  for s in range(len(self.sides)):
   self.side_centroids[s] = self.compute_side_centroid(s)

 def compute_side_centroid(self,ns):

  nodes = self.nodes[self.sides[ns]]

  centroid = np.zeros(3)
  for p in nodes:
   centroid += p
  return centroid/len(nodes)



 def compute_elem_centroids(self):

  self.elem_centroids = np.zeros((len(self.elems),3))
  for elem in range(len(self.elems)):
   self.elem_centroids[elem] = self.compute_elem_centroid(elem)


 def compute_elem_centroid(self,ne):
  nodes = self.nodes[self.elems[ne]]
  centroid = np.zeros(3)
  for p in nodes:
   centroid += p

  return centroid/len(nodes)

 def compute_elem_volume(self,kc1):

  if self.dim == 3: #Assuming Tetraedron
   ns = self.elems[kc1]
   m = np.ones((4,4))   
   m[0,0:3] = self.nodes[ns[0]]
   m[1,0:3] = self.nodes[ns[1]]
   m[2,0:3] = self.nodes[ns[2]]
   m[3,0:3] = self.nodes[ns[3]]
   return abs(1.0/6.0 * np.linalg.det(m))

  if self.dim == 2: #Assuming Tetraedron

   points = self.nodes[self.elems[kc1]]
   x = []; y= []
   for p in points:
    x.append(p[0])
    y.append(p[1])

   return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


   return volume


 def get_side_periodic_value(self,ll,elem) :
  
    ind = self.side_elem_map[ll].index(elem)
    return self.side_periodic_value[ll][ind]


 def get_next_elem_centroid(self,elem,side):
  
  centroid1 = self.get_elem_centroid(elem)
  elem2 = self.get_neighbor_elem(elem,side)
  centroid2 = self.get_elem_centroid(elem2)
  ind1 = self.side_elem_map[side].index(elem)
  ind2 = self.side_elem_map[side].index(elem2)
  centroid = centroid2 - self.side_periodicity[side][ind1] + self.side_periodicity[side][ind2]
  return centroid


 def compute_centroid(self,side):
   
  node = np.zeros(3)
  for p in side:
   node += self.nodes[p]

  return node/len(side)


 def get_neighbor_elem(self,elem,ll) :
     
    if not (elem in self.side_elem_map[ll]) : print('error')

    for tmp in self.side_elem_map[ll] :
       if not (tmp == elem) :
         return tmp 


 def compute_interpolation_weigths(self):

  self.interp_weigths = np.zeros(len(self.sides))
  for ll in self.side_list['active']:
   if not (ll in (self.side_list['Boundary'] + self.side_list['Hot'] + self.side_list['Cold'])) :
    e0 = self.side_elem_map[ll][0]
    e1 = self.side_elem_map[ll][1]
    P0 = self.get_elem_centroid(e0)
    P1 = self.get_next_elem_centroid(e0,ll)
    #---------------------------------------------------------------
    if self.dim == 3:
     #from here: http://geomalgorithms.com/a05-_intersect-1.html
     u = P1 - P0
     n = self.get_side_normal(1,ll)
     node = self.nodes[self.sides[ll][0]]
     w = P0 - node
     s = -np.dot(n,w)/np.dot(n,u)
    else: #dim = 2
     P2 = self.nodes[self.sides[ll][0]]
     P3 = self.nodes[self.sides[ll][1]]
     den = (P3[1] - P2[1])*(P1[0]-P0[0])-(P3[0]-P2[0])*(P1[1]-P0[1])
     num = (P3[0] - P2[0])*(P0[1]-P2[1])-(P3[1]-P2[1])*(P0[0]-P2[0])
     a = num/den
     P = P0 + a * (P1-P0)
     if a < 0.0 or a > 1.0 :
      print(P0)
      print(P1)
      print('ERROR in the skew parameter')
      return
     dist = np.linalg.norm(P1-P0)
     d = np.linalg.norm(P - P1) 
     s = d/dist 
     #---------------------------------------------------------------
    self.interp_weigths[ll] = s


 def get_interpolation_weigths(self,ll):

  return self.interp_weigths[ll]

 def compute_boundary_condition_data(self):

    gradir = self.direction

    if gradir == 0:
     flux_dir = [1,0,0]
     length = self.size[0]

    if gradir == 1:
     flux_dir = [0,1,0]
     length = self.size[1]

    if gradir == 2:
     flux_dir = [0,0,1]
     length = self.size[2]

    delta = 1e-2
    nsides = len(self.sides)
    flux_sides = [] #where flux goes 
    side_value = np.zeros(nsides)
    
    tmp = self.side_list.setdefault('Periodic',[]) + self.exlude


    for kl,ll in enumerate(tmp) :
     normal = self.compute_side_normal(self.side_elem_map[ll][0],ll)
     tmp = np.dot(normal,flux_dir)
     if tmp > delta : #either negative or positive
      flux_sides.append(ll)

     if tmp < - delta :
        side_value[ll] = -1.0
     if tmp > delta :
        side_value[ll] = +1.0

    side_periodic_value = np.zeros((nsides,2))


    if len(self.side_list.setdefault('Periodic',[])) > 0:
     for t,side in enumerate(self.pairs) :
      side_periodic_value[side[0]][0]= side_value[side[0]]
      side_periodic_value[side[0]][1]= side_value[side[1]]

    self.area_flux = abs(np.dot(flux_dir,self.c_areas))
    self.flux_sides = flux_sides
    self.side_periodic_value = side_periodic_value




 def compute_least_square_weigths(self):

   n_el = len(self.elems)
  
   nd = len(self.elems[0])
   diff_dist = np.zeros((n_el,nd,self.dim))


   for ll in self.side_list['active'] :

    elems = self.side_elem_map[ll]
    kc1 = elems[0]
    c1 = self.compute_elem_centroid(kc1)
    ind1 = self.elem_side_map[kc1].index(ll)

    if not ll in (self.side_list['Boundary'] + self.side_list['Hot'] + self.side_list['Cold']): 

     #Diff in the distance
     kc2 = elems[1]
     c2 = self.get_next_elem_centroid(kc1,ll)
     ind2 = self.elem_side_map[kc2].index(ll)
     dist = c2-c1

     for i in range(self.dim):
      diff_dist[kc1][ind1][i] = dist[i]
      diff_dist[kc2][ind2][i] = -dist[i]
    else : 
     dist = self.compute_side_centroid(ll) - c1
     for i in range(self.dim):
      diff_dist[kc1][ind1][i] = dist[i]

   #Compute weights
   self.weigths = []
   for tmp in diff_dist :
    self.weigths.append(np.dot(np.linalg.inv(np.dot(np.transpose(tmp),tmp)),np.transpose(tmp)  ))


 def compute_grad(self,temp,lattice_temp =[]) :


   if len(lattice_temp) == 0: lattice_temp = temp

   n_el = len(self.elems)
   nd = len(self.elems[0])

   if self.dim == 2:
    diff_temp = np.zeros((n_el,nd))
   else:
    diff_temp = np.zeros((n_el,4))

   gradT = np.zeros((n_el,3))

   #Diff in the temp
   for ll in self.side_list['active'] :

    elems = self.side_elem_map[ll]
    kc1 = elems[0]
    #c1 = self.compute_elem_centroid(kc1)
    c1 = self.get_elem_centroid(kc1)
    ind1 = self.elem_side_map[kc1].index(ll)

    if not ll in (self.side_list['Boundary'] + self.side_list['Hot'] + self.side_list['Cold']) : 

     kc2 = elems[1]
     ind2 = self.elem_side_map[kc2].index(ll)
     
     temp_1 = temp[kc1]
     temp_2 = temp[kc2] + self.get_side_periodic_value(ll,kc2)
     diff_t = temp_2 - temp_1    
  
     diff_temp[kc1][ind1]  = diff_t
     diff_temp[kc2][ind2]  = -diff_t 
    else : 
     if ll in self.side_list['Hot'] :
      diff_temp[kc1][ind1]  = 0.5-temp[kc1]

     if ll in self.side_list['Cold'] :
      diff_temp[kc1][ind1]  = -0.5-temp[kc1]

     if ll in self.side_list['Boundary'] :
      diff_temp[kc1][ind1]  = lattice_temp[kc1]-temp[kc1]

   for k in range(n_el) :
    tmp = np.dot(self.weigths[k],diff_temp[k])
    gradT[k][0] = tmp[0] #THESE HAS TO BE POSITIVE
    gradT[k][1] = tmp[1]
    if self.dim == 3:
     gradT[k][2] = tmp[2]

   return gradT
 
