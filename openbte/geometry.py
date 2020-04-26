import numpy as np
from .mesher import *
from .utils import *
import time
import scipy.sparse as sp
import deepdish as dd
import itertools
from .GenerateInterface import *
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD

class Geometry(object):

 def __init__(self,**argv):
  
  if comm.rank == 0:
   if argv.setdefault('model','lattice') == 'interface':
     GenerateInterface(**argv) 
   else:
     Mesher(argv) #this create mesh.msh
     self.dmin = argv['dmin']

  self.import_mesh(**argv)

  if comm.rank == 0:
   data = self.compute_mesh_data(**argv)
   if argv.setdefault('save',True):
     dd.io.save('geometry.h5',data)
  else:  data = None
  self.data = comm.bcast(data,root=0)


 def compute_node_map(self,**argv):

   self.conn = np.zeros(len(self.nodes))
   for k,e in enumerate(self.elems):
     for n in e:
      self.conn[n] +=1


 def compute_mesh_data(self,**argv):


    self.compute_elem_volumes()
    self.compute_side_areas()
    self.compute_side_normals()
    self.compute_elem_centroids()
    self.compute_side_centroids()
    self.compute_least_square_weigths()
    self.compute_connecting_matrix()
    self.compute_interpolation_weigths()
    self.compute_dists()
    self.compute_boundary_condition_data(argv)
    self.compute_node_map()
    self.n_elems = len(self.elems)

    #generate_elem_mat_map

    #generate_frame
    frame = generate_frame(**argv)
    #-----------
    #argv.update({'centroids':self.elem_centroids})
    #self.elem_mat_map = CreateCorrelation(**argv)
    self.elem_mat_map = { i:[0] for i in range(len(self.elems))}


    return {'side_list':self.side_list,\
          'n_elems':np.array([self.n_elems]),\
          'elem_side_map':self.elem_side_map,\
          'side_elem_map':self.side_elem_map,\
          'elem_mat_map':self.elem_mat_map,\
          'elems':self.elems,\
          'sides':self.sides,\
          'size':self.size,\
          'lx':argv['lx'],\
          'conn':self.conn,\
          'ly':argv['ly'],\
          'dim':np.array([self.dim]),\
          'weigths':self.weigths,\
          'kappa_factor':np.array([self.kappa_factor]),\
          'areas':self.side_areas,\
          'nodes':self.nodes,\
          'periodic_values':self.periodic_values,\
          'periodic_side_values':self.periodic_side_values,\
          'interp_weigths':self.interp_weigths,\
          'centroids':self.elem_centroids,\
          'side_centroids':self.side_centroids,\
          'volumes':self.elem_volumes,\
          'kappa_mask':np.array(np.sum(self.B_with_area_old.todense(),axis=0))[0],\
          'B':self.B.todense(),\
          'normals':self.new_normals,\
          'dists':self.dists,\
          'flux_sides':self.flux_sides,\
          'frame':frame,\
          'i':self.i,\
          'j':self.j,\
          'k':self.k,\
          'ip':self.ip,\
          'jp':self.jp,\
          'dp':self.dp,\
          'eb':self.eb,\
          'sb':self.sb,\
          'db':self.db,\
          'dbp':self.dbp,\
          'pv':self.pv,\
          'n_side_per_elem':np.array([len(self.elems[0])])}



 def compute_dists(self):
  self.dists= {}   
  for ll in self.side_list['active']:
   if not (ll in self.side_list['Boundary']):# + self.side_list['Hot'] + self.side_list['Cold'])) :
    elem_1 = self.side_elem_map[ll][0]
    elem_2 = self.side_elem_map[ll][1]
    c1 = self.elem_centroids[elem_1]
    if elem_1 == elem_2:
     c2 = self.get_side_centroid(ll)
    else:    
     c2 = self.get_next_elem_centroid(elem_1,ll)
    dist = c2 - c1
    self.dists.setdefault(elem_1,{}).update({elem_2:dist})



 def compute_connecting_matrix(self):

   nc = len(self.elems)
   ns = len(self.elem_side_map[0])
   self.i = [];self.j = [];self.k = []
   self.eb = [];self.sb = []; self.db = []; self.dbp = []

   data = []

   
   for ll in self.side_list['active'] :
     elems = self.side_elem_map[ll]
     l1 = elems[0]
     l2 = elems[1]
     vol1 = self.elem_volumes[l1]
     vol2 = self.elem_volumes[l2]
     area = self.compute_side_area(ll)
     normal = self.compute_side_normal(l1,ll)
     if not l1 == l2:
       #New development-----------------
       self.i.append(l1)
       self.j.append(l2)
       self.k.append(normal*area/vol1)
       self.j.append(l1)
       self.i.append(l2)
       self.k.append(-normal*area/vol2)
       #---------------------------------

     else:
       s = np.where(np.array(self.elem_side_map[l1])==ll)[0][0]
       self.eb.append(l1)
       self.sb.append(ll)
       self.db.append(normal*area/vol1)
       self.dbp.append(normal)

   self.k = np.array(self.k).T
   self.db = np.array(self.db).T
   self.dbp = np.array(self.dbp).T





 def get_decomposed_directions(self,elem_1,elem_2,rot = np.eye(3)):

    side = self.get_side_between_two_elements(elem_1,elem_2)
    normal = self.compute_side_normal(elem_1,side)
    area = self.compute_side_area(side)
    Af = area*normal
    c1 = self.get_elem_centroid(elem_1)

    if elem_1 == elem_2:
     c2 = self.get_side_centroid(side)
    else:    
     c2 = self.get_next_elem_centroid(elem_1,side)

    dist = c2 - c1

    v_orth = np.dot(normal,np.dot(rot,normal))/np.dot(normal,dist)
    v_non_orth = np.dot(rot,normal) - dist*v_orth
    return area*v_orth,area*v_non_orth



 def get_side_between_two_elements(self,elem_1,elem_2):

   if elem_1 == elem_2: #Boundary side
    for side in self.elem_side_map[elem_1]:
     if side in self.side_list['Boundary'] :
      return side
    print('No boundary side')
    quit()
   else:

    for side_1 in self.elem_side_map[elem_1]:
     for side_2 in self.elem_side_map[elem_2]:
      if side_1 == side_2:
       return side_1

    print('no adjacents elems')
    assert(1==0)
    quit()

 def compute_elem_volumes(self):

  self.elem_volumes = np.zeros(len(self.elems))
  for k in range(len(self.elems)):
   self.elem_volumes[k] = self.compute_elem_volume(k)


 def compute_side_areas(self):
  self.side_areas = np.zeros(len(self.sides))
  for k in range(len(self.sides)):
   self.side_areas[k] = self.compute_side_area(k)



 def import_mesh(self,**argv):

  if comm.rank == 0:
   self.elem_region_map = {}
   self.region_elem_map = {}

   with open('mesh.msh', 'r') as f: lines = f.readlines()

   lines = [l.split()  for l in lines]
   nb = int(lines[4][0])
   current_line = 5
   #Get physical regions
   self.blabels = {int(lines[current_line+i][1]) : lines[current_line+i][2].replace('"',r'') for i in range(nb)}

   current_line += nb+2
   n_nodes = int(lines[current_line][0])
   current_line += 1
   self.nodes = np.array([lines[current_line + n][1:4] for n in range(n_nodes)],float)
   current_line += n_nodes+1 
   #Get size
   self.size = [ np.max(self.nodes[:,i]) - np.min(self.nodes[:,i])   for i in range(3)]
   self.dim = 2 if self.size[2] == 0 else 3
   current_line += 1
   #load elements----
   n_elem_tot = int(lines[current_line][0])

   current_line += 1

   #type of elements
   bulk_type = {2:[2,3],3:[4]}
   face_type = {2:[1],3:[2]}
   #---------------------------


   bulk_tags = [n for n in range(n_elem_tot) if int(lines[current_line + n][1]) in bulk_type[self.dim]] 
   face_tags = [n for n in range(n_elem_tot) if int(lines[current_line + n][1]) in face_type[self.dim]]

   self.elems = [list(np.array(lines[current_line + n][5:],dtype=int)-1) for n in bulk_tags] 


   n_elems = len(self.elems)
   boundary_sides = np.array([ sorted(np.array(lines[current_line + n][5:],dtype=int)) for n in face_tags] ) -1

   #generate sides and maps---
   #side = np.zeros(3)

   tt = time.time()

   tt = time.time()

   #this is the bottleneck.
   b = time.time()

   self.elem_side_map = {}
   self.node_side_map = {}
   self.sides = []
   for k,elem in enumerate(self.elems):
       tmp = list(elem)
       for i in range(len(elem)):
         tmp.append(tmp.pop(0))
         trial = sorted(tmp[:self.dim])
         if not trial in self.sides:
            self.sides.append(trial)
            r = len(self.sides)-1
            for t in trial: self.node_side_map.setdefault(t,[]).append(r) 
         else: r = self.sides.index(trial)  
         self.elem_side_map.setdefault(k,[]).append(r)   
   #----------------------------------------------------      
  if comm.rank == 0:
   #match the boundary sides with the global side.
   physical_boundary = {}
   for n,bs in enumerate(boundary_sides): #match with the boundary side
      side = self.node_side_map[bs[0]]
      for s in side: 
        if np.allclose(np.array(self.sides[s]),np.array(bs),rtol=1e-4,atol=1e-4):
            #The boundary sides (msh2) are listed at the beginning of the elements list
            physical_boundary.setdefault(self.blabels[int(lines[current_line + n][3])],[]).append(s) 
            break
   #--------------------------------------------

   self.side_elem_map = {}
   for key, value in self.elem_side_map.items():
     for v in value:   
      self.side_elem_map.setdefault(v,[]).append(key)

   self.side_list = {}

   #Apply Periodic Boundary Conditions
   self.side_list.update({'active':list(range(len(self.sides)))})
   self.side_periodicity = np.zeros((len(self.sides),2,3))
   group_1 = []
   group_2 = []
   #self.pairs = [] #global (all periodic pairs)

   self.side_list.setdefault('Boundary',[])
   self.side_list.setdefault('Interface',[])
   self.periodic_nodes = {}

   if argv.setdefault('delete_gmsh_files',False):
    os.remove(os.getcwd() + '/mesh.msh')
    os.remove(os.getcwd() + '/mesh.geo')

   self.pairs = []
   for label in list(physical_boundary.keys()):

    if str(label.split('_')[0]) == 'Periodic':
     if not int(label.split('_')[1])%2==0:
      contact_1 = label
      contact_2 = 'Periodic_' + str(int(label.split('_')[1])+1)
      group_1 = physical_boundary[contact_1]
      group_2 = physical_boundary[contact_2]
      for s in group_1:
        c = self.compute_side_centroid(s)
      for s in group_2:
        c = self.compute_side_centroid(s)

      #----create pairs
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
       pairs.append([s1,s2])
       self.side_periodicity[s1][1] = pp
       self.side_periodicity[s2][1] = -pp
      self.pairs +=pairs
      #----------------------------------
      plot_sides = False
      if plot_sides:
       for s in pairs:
        c1 = self.compute_side_centroid(s[0])
        c2 = self.compute_side_centroid(s[1])
        plot([c1[0],c2[0]],[c1[1],c2[1]])
       c1 = self.compute_elem_centroid(4151)
       scatter(c1[0],c1[1],color='r')
       c1 = self.compute_elem_centroid(4154)
       scatter(c1[0],c1[1],color = 'g')
       c1 = self.compute_elem_centroid(4150)
       scatter(c1[0],c1[1],color = 'b')
       show()

      #Amend map
      for s in pairs:
       s1 = s[0]
       s2 = s[1]

       #Change side in elem 2--------------------
       elem2 = self.side_elem_map[s2][0]
       index = self.elem_side_map[elem2].index(s2)
       self.elem_side_map[elem2][index] = s1
       self.side_elem_map[s1].append(elem2)
       self.side_elem_map[s2].append(self.side_elem_map[s1][0])
       self.side_list['active'].remove(s2)
       #-----------------------------------------

      #Polish sides
      [self.side_list.setdefault('Periodic',[]).append(i) for i in physical_boundary[contact_1]]
      [self.side_list.setdefault('Inactive',[]).append(i) for i in physical_boundary[contact_2]]


   if 'Boundary' in physical_boundary.keys(): self.side_list.update({'Boundary':physical_boundary['Boundary']})

   for side in self.side_list['Boundary'] :# + self.side_list['Hot'] + self.side_list['Cold']:
    self.side_elem_map[side].append(self.side_elem_map[side][0])

   self.elem_kappa_map = {}
   self.elem_mat_map = { ne:0 for ne in list(range(len(self.elems)))}



 def compute_side_normals(self):

  self.new_normals = {}   
  self.side_normals = np.zeros((len(self.sides),2,3))
  for s in range(len(self.sides)):
   elems = self.side_elem_map[s]
   self.side_normals[s][0] = self.compute_side_normal(elems[0],s)
   self.new_normals.setdefault(elems[0],{}).update({elems[1]:self.compute_side_normal(elems[0],s)})
   if len(elems)>1:
    self.side_normals[s][1] = self.compute_side_normal(elems[1],s)
    self.new_normals.setdefault(elems[1],{}).update({elems[0]:self.compute_side_normal(elems[1],s)})

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



 def get_next_elem_centroid(self,elem,side):

  centroid1 = self.elem_centroids[elem]
  elem2 = self.get_neighbor_elem(elem,side)
  centroid2 = self.elem_centroids[elem2]
  ind1 = self.side_elem_map[side].index(elem)
  ind2 = self.side_elem_map[side].index(elem2)
  centroid = centroid2 - self.side_periodicity[side][ind1] + self.side_periodicity[side][ind2]
  return centroid

 def compute_centroid(self,side):

  node = np.zeros(3)
  for p in side:
   node += self.nodes[p]

  return node/len(side)


 def get_elem_neighbors(self,elem):

  neighbors = []
  for ll in self.elem_side_map[elem]:
   neighbors.append(self.get_neighbor_elem(elem,ll))
  return neighbors

 def get_neighbor_elem(self,elem,ll) :

    if not (elem in self.side_elem_map[ll]) : print('error, no neighbor',ll,elem)

    for tmp in self.side_elem_map[ll] :
       if not (tmp == elem) :
         return tmp


 def compute_interpolation_weigths(self):

  self.interp_weigths = {}# np.zeros(len(self.sides))
  for ll in self.side_list['active']:
   if not (ll in self.side_list['Boundary']) : # + self.side_list['Hot'] + self.side_list['Cold'])) :
    e0 = self.side_elem_map[ll][0]
    e1 = self.side_elem_map[ll][1]
    
    P0 = self.elem_centroids[e0]
    P1 = self.get_next_elem_centroid(e0,ll)
    #---------------------------------------------------------------
    if self.dim == 3:
     #from here: http://geomalgorithms.com/a05-_intersect-1.html
     u = P1 - P0
     n = self.side_normals[ll,1]
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
      print(ll)
      print(P0)
      print(P1)
      print(e0)
      print(e1)
      print(P2)
      print(P3)
      print('ERROR in the skew parameter')
      return
     dist = np.linalg.norm(P1-P0)
     d = np.linalg.norm(P - P1)
     s = d/dist
     #---------------------------------------------------------------
    self.interp_weigths.update({ll:[s]})
   else:
    self.interp_weigths.update({ll:[1.0]})



 def compute_boundary_condition_data(self,argv):

    direction = argv.setdefault('direction','x')
    if direction == 'x':
     gradir = 0
     applied_grad = [1,0,0]
    if direction == 'y': 
     applied_grad = [0,1,0]
     gradir = 1
    if direction == 'z':
     applied_grad = [0,0,1]
     gradir = 2

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
    side_value = np.zeros(nsides)

    tmp = self.side_list.setdefault('Periodic',[]) + self.side_list.setdefault('Inactive',[])# + self.side_list.setdefault('Cold',[])

    for kl,ll in enumerate(tmp) :
     normal = self.compute_side_normal(self.side_elem_map[ll][0],ll)
     tmp = np.dot(normal,flux_dir)
     #if tmp < -delta : #either negative or positive
     # flux_sides.append(ll)

     if tmp < - delta :
        side_value[ll] = -1.0
     if tmp > delta :
        side_value[ll] = +1.0
     

    side_periodic_value = np.zeros((nsides,2))
    self.periodic_values = {}
    self.periodic_side_values = {}

    n_el = len(self.elems)
    B = sp.dok_matrix((n_el,n_el),dtype=np.float64)

    B_with_area_old = sp.dok_matrix((n_el,n_el),dtype=np.float64)
    self.B_area = np.zeros(n_el,dtype=np.float64)
    
    self.ip = []; self.jp = []; self.dp = []; self.pv = [] 
    if len(self.side_list.setdefault('Periodic',[])) > 0:
     for side in self.pairs:

      area = self.side_areas[side[0]]
      
      side_periodic_value[side[0]][0] = side_value[side[0]]
      side_periodic_value[side[0]][1] = side_value[side[1]]

      i,j = self.side_elem_map[side[0]]

      self.periodic_values.update({i:{j:[side_value[side[0]]]}})
      self.periodic_values.update({j:{i:[side_value[side[1]]]}})
    
      self.periodic_side_values.update({side[0]:side_value[side[0]]})


      normal = self.compute_side_normal(i,side[0])
      
      voli = self.elem_volumes[i]
      volj = self.elem_volumes[j]
 

      B[i,j] = side_value[side[0]]
      B[j,i] = side_value[side[1]]
      self.ip.append(i); self.jp.append(j); self.dp.append(side_value[side[0]]); self.pv.append(normal*area/voli)
      self.ip.append(j); self.jp.append(i); self.dp.append(side_value[side[1]]); self.pv.append(-normal*area/volj)

      if np.linalg.norm(np.cross(self.side_normals[side[0],1],applied_grad)) < 1e-12:
       B_with_area_old[i,j] = abs(side_value[side[0]]*area)
       

    #select flux sides-------------------------------
    self.flux_sides = [] #where flux goes
    total_area = 0
    for ll in self.side_list['Periodic']:
     normal = self.compute_side_normal(self.side_elem_map[ll][0],ll)
     tmp = np.abs(np.dot(normal,flux_dir))
     if tmp > delta : #either negative or positive
       if normal[0] == 1:
           if self.dim == 2:  
            area_flux = self.size[1]
           else: 
            area_flux = self.size[1]*self.size[2]
           total_area += self.side_areas[ll]

       elif normal[1] == 1:
           if self.dim == 2:  
            area_flux = self.size[0]
           else: 
            area_flux = self.size[0]*self.size[2]

       self.flux_sides.append(ll)

    #-----------------------------------------------
    #----------------------------------------------------------------
    self.side_periodic_value = side_periodic_value
    self.B = B.tocoo()
    self.B_with_area_old = B_with_area_old.tocoo()
    self.pv = np.array(self.pv).T
    self.kappa_factor = self.size[gradir]/area_flux
  

 def compute_least_square_weigths(self):

   nd = len(self.elems[0])
   diff_dist = {}

   for ll in self.side_list['active'] :
    elems = self.side_elem_map[ll]
    kc1 = elems[0]
    c1 = self.elem_centroids[kc1]
    ind1 = self.elem_side_map[kc1].index(ll)
    if not ll in self.side_list['Boundary'] :
     #Diff in the distance
     kc2 = elems[1]
     c2 = self.get_next_elem_centroid(kc1,ll)
     ind2 = self.elem_side_map[kc2].index(ll)
     dist = c2-c1

     for i in range(self.dim):
      diff_dist.setdefault(kc1,np.zeros((len(self.elem_side_map[kc1]),self.dim)))[ind1][i] = dist[i]   
      diff_dist.setdefault(kc2,np.zeros((len(self.elem_side_map[kc2]),self.dim)))[ind2][i] = -dist[i]   
    else :
     dist = self.compute_side_centroid(ll) - c1
     for i in range(self.dim):
      #diff_dist[kc1][ind1][i] = dist[i]
      diff_dist.setdefault(kc1,np.zeros((len(self.elem_side_map[kc1]),self.dim)))[ind1][i] = dist[i]   
      
     if ll in self.side_list['Interface']:
      kc2 = elems[1]
      c2 = self.get_next_elem_centroid(kc1,ll)
      dist = self.compute_side_centroid(ll) - c2
      ind2 = self.elem_side_map[kc2].index(ll)
      for i in range(self.dim):
        #diff_dist[kc2][ind2][i] = dist[i]
        diff_dist.setdefault(np.zeros((len(self.elem_side_map[kc2]),self.dim)))[ind2][i] = dist[i]   


   #Compute weights
   self.weigths = {}
   for h in diff_dist.keys() :
    tmp = diff_dist[h]   
    self.weigths[h] = np.dot(np.linalg.inv(np.dot(np.transpose(tmp),tmp)),np.transpose(tmp)  )



