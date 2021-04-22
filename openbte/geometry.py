import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from .mesher import *
from .utils import *
import time
import scipy.sparse as sp
import itertools
from mpi4py import MPI
import time
import numpy.testing as npt
from statistics import mean
from scipy.ndimage.interpolation import shift
from collections import Counter

comm = MPI.COMM_WORLD

def compute_boundary_condition_data(data,**argv):

    
    side_elem_map = data['side_elem_map_vec']
    face_normals = data['face_normals']
    volumes = data['volumes']
    dim = data['dim']

    direction = data['direction']
    if direction == 0:
     gradir = 0
     applied_grad = [1,0,0]
    if direction == 1:
     gradir = 1
     applied_grad = [0,1,0]
    if direction == 2:
     applied_grad = [0,0,1]
     gradir = 2

    if gradir == 0:
     flux_dir = [1,0,0]
     length = data['size'][0]

    if gradir == 1:
     flux_dir = [0,1,0]
     length = data['size'][1]

    if gradir == 2:
     flux_dir = [0,0,1]
     length = data['size'][2]

    delta = 1e-2
    nsides = len(data['sides'])
    side_value = np.zeros(nsides)

    tmp = list(data['periodic_sides']) + list(data['inactive_sides'])

    DeltaT = 1
    for kl,ll in enumerate(tmp) :
     Ee1,e2 = side_elem_map[ll]
     normal = face_normals[ll]  
     tmp = np.dot(normal,flux_dir)


     if tmp < - delta :
        side_value[ll] = -DeltaT
     if tmp > delta :
        side_value[ll] = +DeltaT
     

    side_periodic_value = np.zeros((nsides,2))
    periodic_values = {}
    periodic_side_values = {}

    n_el = len(data['elems'])
    B = sp.dok_matrix((n_el,n_el),dtype=np.float64)

    B_with_area_old = sp.dok_matrix((n_el,n_el),dtype=np.float64)
    B_area = np.zeros(n_el,dtype=np.float64)
   
    counter = 0
    pp = ()
    ip = []; jp = []; dp = []; pv = [] 
    if len(data['periodic_sides']) > 0:
     for side in data['pairs']:

      area = data['areas'][side[0]]
      
      side_periodic_value[side[0]][0] = side_value[side[0]]
      side_periodic_value[side[0]][1] = side_value[side[1]]

      i,j = side_elem_map[side[0]]

      periodic_values.update({i:{j:[side_value[side[0]]]}})
      periodic_values.update({j:{i:[side_value[side[1]]]}})
    
      periodic_side_values.update({side[0]:side_value[side[0]]})

      normal = data['face_normals'][side[0]]
      
      voli = volumes[i]
      volj = volumes[j]
 
      B[i,j] = side_value[side[0]]
      B[j,i] = side_value[side[1]]
  

      if abs(side_value[side[0]]) > 0:
       pp += ((data['ij'].index([i,j]),side_value[side[0]]),)
       ip.append(i); jp.append(j); dp.append(side_value[side[0]]); pv.append(normal*area/voli)
       counter +=1

      if abs(side_value[side[1]]) > 0:
       pp += ((data['ij'].index([j,i]),side_value[side[1]]),)
       ip.append(j); jp.append(i); dp.append(side_value[side[1]]); pv.append(-normal*area/volj)
       counter +=1

      if np.linalg.norm(np.cross(data['face_normals'][side[0]],applied_grad)) < 1e-12:
       B_with_area_old[i,j] = abs(side_value[side[0]]*area)
    

    #select flux sides-------------------------------
    flux_sides = [] #where flux goes
    total_area = 0
    for ll in data['periodic_sides']:
     e1,e2 = side_elem_map[ll]
     normal = data['face_normals'][ll]   
     tmp = np.abs(np.dot(normal,flux_dir))
     if tmp > delta : #either negative or positive
       if abs(normal[0]) == 1:
           if dim == 2:  
            area_flux = data['size'][1]
           else: 
            area_flux = data['size'][1]*data['size'][2]
           total_area += data['areas'][ll]

       elif abs(normal[1]) == 1:
           if data['dim'] == 2:  
            area_flux = data['size'][0]
           else:
            area_flux = data['size'][0]*data['size'][2]
           total_area += data['areas'][ll]

       else : #along z
           area_flux = data['size'][0]*data['size'][1]
           total_area += data['areas'][ll]


       flux_sides.append(ll)

    #-------THIS
    if argv.setdefault('contact_area','box') == 'box':
      kappa_factor = data['size'][gradir]/area_flux
    else:  
      kappa_factor = data['size'][gradir]/total_area

    data['kappa_mask']= -np.array(np.sum(B_with_area_old.todense(),axis=0))[0]*kappa_factor*1e-18
    data['periodic_side_values'] = [periodic_side_values[ll]  for ll in data['periodic_sides']]
    data['pp'] = np.array(pp)
    data['applied_gradient'] = np.array(applied_grad)
    data['flux_sides'] = np.array(flux_sides)
    data['kappa_factor'] = kappa_factor



def compute_dists(data):
  

  dists_side = np.zeros((len(data['sides']),3))
  side_elem_map = data['side_elem_map_vec']
  centroids = data['centroids']

  for ll in data['active_sides']:
   if not (ll in data['boundary_sides']):
    elem_1 = side_elem_map[ll][0]
    elem_2 = side_elem_map[ll][1]
    c1 = centroids[elem_1]
    c2 = get_next_elem_centroid(elem_1,ll,data)
    dist = c2 - c1
    dists_side[ll] = dist 

  data['dists'] = np.array(dists_side) 




def compute_interpolation_weigths(data):

  #from here: http://geomalgorithms.com/a05-_intersect-1.html

  net_sides=data['active_sides'][~np.isin(data['active_sides'],data['boundary_sides'])] #Get only the nonboundary sides

  e1 =  data['side_elem_map_vec'][net_sides,0]

  w =  np.zeros((len(data['sides']),3))

  w[net_sides] = data['centroids'][e1] - data['nodes'][data['sides'][net_sides,0]]

  tmp  = np.einsum('ui,ui->u',data['face_normals'],data['dists'])
  tmp2 = np.einsum('ui,ui->u',data['face_normals'],w)

  interp_weigths = np.zeros(len(data['sides']))
  interp_weigths[net_sides]   = 1+tmp2[net_sides]/tmp[net_sides] #this is the relative distance with the centroid of the second element


  #check----------------------------------
  #ll = data['active_sides'][0]
  #(e1,_) = data['side_elem_map_vec'][ll]
  #P2 = data['nodes'][data['sides'][ll][0]]
  #P3 = data['nodes'][data['sides'][ll][1]]
  #P0 = data['centroids'][e1]
  #P1 = get_next_elem_centroid(e1,ll,data)
  #plot([P0[0],P1[0]],[P0[1],P1[1]],'r')
  #plot([P2[0],P3[0]],[P2[1],P3[1]],'b')
  #text(P0[0],P0[1],'P0')
  #text(P1[0],P1[1],'P1')
  #text(P2[0],P2[1],'P2')
  #text(P3[0],P3[1],'P3')
  #print(interp_weigths[ll])
  #show()

  if len(data['boundary_sides']) > 0:
   interp_weigths[data['boundary_sides']] = 1

  data['interp_weigths'] = interp_weigths


def compute_connecting_matrix(data):

   elems         = data['elems']
   elem_side_map = data['elem_side_map_vec']
   side_elem_map = data['side_elem_map_vec']
   elem_volumes  = data['volumes']
   side_areas    = data['areas']
   dim           = data['dim']
   face_normals  = data['face_normals']
   a_sides =       data['active_sides']
   b_sides =       data['boundary_sides']
   c_sides =       data['interface_sides']

   #Active sides-----
   net_sides=a_sides[~np.isin(a_sides,list(b_sides)+list(c_sides))]
   i2 = side_elem_map[net_sides].flatten()
   j2 = np.flip(side_elem_map[net_sides],1).flatten()
   common = np.einsum('u,ui->ui',side_areas[net_sides],face_normals[net_sides,:dim])
   t1 =  common/elem_volumes[side_elem_map[net_sides][:,0],None]
   t2 =  common/elem_volumes[side_elem_map[net_sides][:,1],None]
   k2 = np.zeros((2*t1.shape[0],dim))
   k2[np.arange(t1.shape[0])*2]   =  t1
   k2[np.arange(t1.shape[0])*2+1] = -t2
   k2 = k2.T     
   #----------------

   #ij - it will need to be removed
   ij2 = np.zeros((2*t1.shape[0],2))
   ij2[np.arange(t1.shape[0])*2  ] = side_elem_map[net_sides]
   ij2[np.arange(t1.shape[0])*2+1] = np.flip(side_elem_map[net_sides],1)

   #-------------------------------------

   #Interface sides----
   if len(c_sides) >0 :
    inti2 = side_elem_map[c_sides].flatten()
    intj2 = np.flip(side_elem_map[c_sides],1).flatten()
    tmp = np.flip(side_elem_map[c_sides],1).flatten()
    common = np.einsum('u,ui->ui',side_areas[c_sides],face_normals[c_sides,:dim])
    t1 =  common/elem_volumes[side_elem_map[c_sides][:,0],None]
    t2 =  common/elem_volumes[side_elem_map[c_sides][:,1],None]
    intk2 = np.zeros((2*t1.shape[0],dim))
    intk2[np.arange(t1.shape[0])*2]   =  t1
    intk2[np.arange(t1.shape[0])*2+1] = -t2
    intk2 = intk2.T     
   else:
    inti2 = []   
    intj2 = []   
    intk2 = []   
    
   #boundary sides----
   sb2 = a_sides[np.isin(a_sides,b_sides)]
   eb2 = side_elem_map[sb2][:,0]
   db2 = face_normals[sb2,:dim].T*side_areas[sb2]/elem_volumes[eb2]
   #-------------
   
   data['intk'] = intk2; 
   data['inti'] = inti2;
   data['intj'] = intj2;
   data['i']    = np.array(i2); 
   data['j']    = np.array(j2); 
   data['k']    = np.array(k2);
   data['eb']   = eb2; 
   data['sb']   = sb2; 
   data['db']   = db2; 
   data['ij']   = ij2.tolist(); 

def get_next_elem_centroid(elem,side,data):

  elem_centroids    = data['centroids']
  side_elem_map     = data['side_elem_map_vec']
  side_periodicity  = data['side_periodicity']

  centroid = elem_centroids[elem]

  if not (elem in side_elem_map[side]) : print('error, no neighbor',side,elem)
  for tmp in side_elem_map[side] :
       if not (tmp == elem) :
         elem2 = tmp  
         break

  centroid2 = elem_centroids[elem2]
  ind1 = list(side_elem_map[side]).index(elem)
  ind2 = list(side_elem_map[side]).index(elem2)
  centroid = centroid2 - side_periodicity[side][ind1] + side_periodicity[side][ind2]

  return centroid


def compute_least_square_weigths(data):

   #elems     = data['elems']
   side_elem_map = data['side_elem_map_vec']
   elem_side_map = data['elem_side_map_vec']
   elem_centroids = data['centroids']
   side_centroids = data['side_centroids']
   dists = data['dists']
   dim = data['dim']
   n_elems = len(data['elems'])


   diff_dist = np.zeros((n_elems,max(data['side_per_elem']),dim))

   for ll in data['active_sides'] :
    elems = side_elem_map[ll]
    dist = dists[ll]
    kc1 = elems[0]
    ind1 = list(elem_side_map[kc1]).index(ll)
    if not ll in data['boundary_sides'] :
     kc2 = elems[1]
     ind2 = list(elem_side_map[kc2]).index(ll)
     diff_dist[kc1,ind1] =  dist[:dim]
     diff_dist[kc2,ind2] = -dist[:dim]
    else :
     dist = side_centroids[ll] - elem_centroids[kc1]
     diff_dist[kc1,ind1] = dist[:dim]

   #We solve the pinv for a stack of matrices. We do so for each element group

   index3 = np.where(data['side_per_elem'] == 3)[0]
   G3 = np.linalg.pinv(diff_dist[index3,:3])
   

   index4 = np.where(data['side_per_elem'] == 4)[0]
   G4 = np.linalg.pinv(diff_dist[index4,:4])

   G = np.zeros((n_elems,dim,max(data['side_per_elem'])))  

   G[index3,:,:3] = G3
   G[index4,:,:4] = G4

   data.update({'weigths':G})


def compute_boundary_connection(data):


     elem_centroids = data['centroids']
     side_centroids = data['side_centroids']
     side_elem_map = data['side_elem_map_vec']
     face_normals = data['face_normals']
     

     bconn = []
     for s in data['boundary_sides']:
        ce = elem_centroids[side_elem_map[s][0]]
        cs = side_centroids[s]
        d = cs-ce
        normal = face_normals[s]
        d1 = normal * np.dot(normal,d)
        bconn.append(d1)

     data.update({'bconn':bconn})


def compute_face_normals(data):


   sides = data['sides']
   nodes = data['nodes']
   dim = data['dim']
   side_elem_map = np.array(data['side_elem_map_vec'])
   elem_centroids = np.array(data['centroids'])
   side_centroids = np.array(data['side_centroids'])

   a= time.time()
   v1 =  nodes[sides[:,1]]-nodes[sides[:,0]]

   if dim == 3:
    v2 =  nodes[sides[:,2]]-nodes[sides[:,1]]
   else :
    v2 = np.array([0,0,1]).T
   v = np.cross(v1,v2)

   normal = v.T/np.linalg.norm(v,axis=1)
   
   c = side_centroids - elem_centroids[side_elem_map[:,0]]   
   index = np.where(np.einsum('iu,ui->u',normal,c) < 0)[0]
   normal[:,index] = -normal[:,index]
   normal = normal.T
 
   #---------------------------------------------------------
   data.update({'face_normals':normal})


def import_mesh(**argv):

   data = {}

   elem_region_map = {}
   region_elem_map = {}

   with open('mesh.msh', 'r') as f: lines = f.readlines()

   lines = [l.split()  for l in lines]
   nb = int(lines[4][0])
   current_line = 5
   blabels = {int(lines[current_line+i][1]) : lines[current_line+i][2].replace('"',r'') for i in range(nb)}

   current_line += nb+2
   n_nodes = int(lines[current_line][0])
   current_line += 1
   nodes = np.array([lines[current_line + n][1:4] for n in range(n_nodes)],float)
   current_line += n_nodes+1

   size = [ np.max(nodes[:,i]) - np.min(nodes[:,i])   for i in range(3)]
   
   dim = 2 if size[2] == 0 else 3
   current_line += 1
   n_elem_tot = int(lines[current_line][0])
   current_line += 1

   #type of elements
   bulk_type = {2:[2,3],3:[4]}
   face_type = {2:[1],3:[2]}
   #---------------------------

   bulk_tags = [n for n in range(n_elem_tot) if int(lines[current_line + n][1]) in bulk_type[dim]] 
   face_tags = [n for n in range(n_elem_tot) if int(lines[current_line + n][1]) in face_type[dim]]
   elems = [list(np.array(lines[current_line + n][5:],dtype=int)-1) for n in bulk_tags]
   side_per_elem = np.array([len(e) for e in elems])
   n_elems = len(elems)
   boundary_sides = np.array([ sorted(np.array(lines[current_line + n][5:],dtype=int)) for n in face_tags] ) -1

   #generate sides and maps---
   node_side_map = { i:[] for i in range(len(nodes))}
   elem_side_map = { i:[] for i in range(len(elems))}
   sides = []
   tmp_indices = []
   for k,elem in enumerate(elems):
       tmp = list(elem)
       for i in range(len(elem)):
         tmp.append(tmp.pop(0))
         trial = sorted(tmp[:dim])
         sides.append(trial)
         tmp_indices.append(k)
   sides,inverse = np.unique(sides,axis=0,return_inverse = True)
   for k,s in enumerate(inverse): elem_side_map[tmp_indices[k]].append(s)

   for s,side in enumerate(sides): 
        for t in side:
            node_side_map[t].append(s)
   side_elem_map = { i:[] for i in range(len(sides))}
   for key, value in elem_side_map.items():
     for v in value:   
      side_elem_map[v].append(key)    
   #----------------------------------------------------      
 
   #Build relevant data
   side_areas = compute_side_areas(nodes,sides,dim) 
 
   #Compute volumes
   elem_volumes = compute_elem_volumes(nodes,elems,dim)

   data.update({'elems':elems,'sides':sides,'nodes':nodes,'volumes':elem_volumes})

   side_centroids = np.array([np.mean(nodes[i],axis=0) for i in data['sides']] )
   elem_centroids = np.array([np.mean(nodes[i],axis=0) for i in data['elems']])

   data.update({'side_centroids':side_centroids})
   data.update({'centroids':elem_centroids})
   
   data.update({'side_per_elem':np.array(side_per_elem)})


   #match the boundary sides with the global side.
   physical_boundary = {}
   for n,bs in enumerate(boundary_sides): #match with the boundary side
      side = node_side_map[bs[0]]
      for s in side: 
        if np.allclose(np.array(sides[s]),np.array(bs),rtol=1e-4,atol=1e-4):
            physical_boundary.setdefault(blabels[int(lines[current_line + n][3])],[]).append(s) 
            break

   side_list = {}
   #Apply Periodic Boundary Conditions
   side_list.update({'active':list(range(len(sides)))})
   side_periodicity = np.zeros((len(sides),2,3))
   group_1 = []
   group_2 = []
   #self.pairs = [] #global (all periodic pairs)

   side_list.setdefault('Boundary',[])
   side_list.setdefault('Interface',[])
   periodic_nodes = {}

   if argv.setdefault('delete_gmsh_files',False):
    os.remove(os.getcwd() + '/mesh.msh')
    os.remove(os.getcwd() + '/mesh.geo')

   pairs = []
   for label in list(physical_boundary.keys()):

    if str(label.split('_')[0]) == 'Periodic':
     if not int(label.split('_')[1])%2==0:
      contact_1 = label
      contact_2 = 'Periodic_' + str(int(label.split('_')[1])+1)
      group_1 = physical_boundary[contact_1]
    
      group_2 = physical_boundary[contact_2]
      for s in group_1:
        c = side_centroids[s]
      for s in group_2:
        c = side_centroids[s]

      #----create pairs
      new_pairs = []
      #compute tangential unity vector
      tmp = nodes[sides[group_2[0]][0]] - nodes[sides[group_2[0]][1]]
      t = tmp/np.linalg.norm(tmp)
      n = len(group_1)
      for s1 in group_1:
       d_min = 1e6
       for s in group_2:
        c1 = side_centroids[s1]
        c2 = side_centroids[s]
        d = np.linalg.norm(c2-c1)
        if d < d_min:
         d_min = d
         pp = c1-c2
         s2 = s
       new_pairs.append([s1,s2])
       side_periodicity[s1][1] = pp
       side_periodicity[s2][1] = -pp
      pairs +=new_pairs
      #----------------------------------
      #Amend map
      for s in new_pairs:
       s1 = s[0]
       s2 = s[1]

       #Change side in elem 2--------------------
       elem2 = side_elem_map[s2][0]
       index = elem_side_map[elem2].index(s2)
       elem_side_map[elem2][index] = s1
    
       side_elem_map[s1].append(elem2)
       side_elem_map[s2].append(side_elem_map[s1][0])
       side_list['active'].remove(s2)
       #-----------------------------------------
        
      #Polish sides
      [side_list.setdefault('Periodic',[]).append(i) for i in physical_boundary[contact_1]]
      [side_list.setdefault('Inactive',[]).append(i) for i in physical_boundary[contact_2]]

   if 'Boundary' in physical_boundary.keys(): side_list.update({'Boundary':physical_boundary['Boundary']})

   for side in side_list['Boundary'] :# + self.side_list['Hot'] + self.side_list['Cold']:
    side_elem_map[side].append(side_elem_map[side][0])

   #Put boundaries at the end
   n_non_boundary_side_per_elem = []
   for key,value in elem_side_map.items():
       tmp_1 = []
       tmp_2 = []
       for s in value:
         if s in side_list['Boundary']:
           tmp_1.append(s)
         else:  
           tmp_2.append(s)
       n_non_boundary_side_per_elem.append(len(tmp_2)+len(tmp_1))
       elem_side_map[key] = tmp_2 + tmp_1


   #Fill with -1 the missing connectivity
   ms = max(side_per_elem)
   for n,e in enumerate(elems):
        if len(e) < ms:
            e.append(-1)
            elem_side_map[n].append(-1)
   #----------------------------------------


   data.update({'active_sides':np.array(side_list['active'])})
   data.update({'n_non_boundary_side_per_elem':np.array(n_non_boundary_side_per_elem)})
   data.update({'boundary_sides':np.array(side_list['Boundary'])})
   data.update({'periodic_sides':np.array(side_list['Periodic'])})
   data.update({'side_periodicity':side_periodicity})
   data.update({'areas':np.array(side_areas)})
   data.update({'elem_mat_map':np.zeros(len(elems))})

   data.update({'dim':dim})
   data.update({'dmin':argv['dmin']})
   data.update({'pairs':pairs})
   #aa = np.array([side_elem_map[ll]  for ll in range(len(sides))]) 
   data.update({'elem_side_map_vec': np.array([elem_side_map[ll]  for ll in range(len(elems))]) })
   data.update({'side_elem_map_vec': np.array([side_elem_map[ll]  for ll in range(len(sides))]) })
   data['interface_sides']= []
   direction = argv.setdefault('direction','x')
   if direction == 'x':
    data['direction'] = 0
   elif  direction == 'y': 
    data['direction'] = 1
   else: 
    data['direction'] = 2
   
   data['size'] = size
   data['inactive_sides'] = side_list['Inactive']

   return data
   #------------------


def compute_data(data,**argv):

   compute_face_normals(data)

   compute_boundary_connection(data)
  
   compute_dists(data)
   
   compute_least_square_weigths(data)
   
   compute_connecting_matrix(data)

   compute_interpolation_weigths(data)
   
   compute_boundary_condition_data(data,**argv)

   #Some adjustement--
   data['meta'] = np.asarray([len(data['elems']),data['kappa_factor'],data['dim'],len(data['nodes']),len(data['active_sides']),data['direction'],data['dmin']],np.float64)
   
   del data['direction']
   del data['dim']
   del data['dmin']
   del data['kappa_factor']




def Geometry(**argv):

 if comm.rank == 0 :

   Mesher(argv) 

   if argv.setdefault('only_geo',False):  

    return argv

   else:

    data = import_mesh(**argv)

    compute_data(data,**argv)

    if argv.setdefault('save',True):
     save_data(argv.setdefault('output_filename','geometry'),data)   

    return data



def compute_elem_volumes(nodes,elems,dim):


  n_elems = len(elems)
  elem_volumes = np.zeros(len(elems))
  if dim == 3: #Assuming Tetraedron
   M = np.ones((4,4))
   for i,e in enumerate(elems):
    M[:,:3] = nodes[e,:]
    elem_volumes[i] = 1.0/6.0 * abs(np.linalg.det(M))

  if dim == 2: 
   M = np.ones((3,3))
   for i,e in enumerate(elems):
    if len(e) == 3:     
     M[:,:2] = nodes[e,0:2]
     elem_volumes[i] = abs(0.5*np.linalg.det(M))
    else:

     points = nodes[e]
     x = points[:,0]
     y = points[:,1]
     elem_volumes[i] = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

  return elem_volumes   

comm = MPI.COMM_WORLD


def compute_side_areas(nodes,sides,dim):

  if dim == 2:
   side_areas = [  np.linalg.norm(nodes[s[1]] - nodes[s[0]]) for s in sides ]
  else:     
   p =nodes[sides]
   side_areas = np.linalg.norm(np.cross(p[:,0]-p[:,1],p[:,0]-p[:,2]),keepdims=True,axis=1).T[0]/2

  return side_areas 





        

