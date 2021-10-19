from mpi4py import MPI
import numpy as np
from termcolor import colored, cprint 
from shapely.geometry import Polygon,Point
from shapely.geometry import MultiPolygon,LineString
import shapely
from shapely.ops import unary_union
import os,sys
import functools
import io,zipfile
import math
import pickle
import gzip
import time
import functools

def compute_grad_common(data,geometry):
     """ Compute grad via least-square method """
     jump = True
     #Compute deltas-------------------   
     rr = []
     for i in geometry['side_per_elem']:
       rr.append(i*[0])

     #this is wrong
     for ll in geometry['active_sides'] :
      kc1,kc2 = geometry['side_elem_map_vec'][ll]
      ind1    = list(geometry['elem_side_map_vec'][kc1]).index(ll)
      ind2    = list(geometry['elem_side_map_vec'][kc2]).index(ll)

      delta = 0
      if ll in geometry['periodic_sides']:
         delta = geometry['periodic_side_values'][list(geometry['periodic_sides']).index(ll)]
      else: delta = 0  

      if jump == 0: delta= 0

      rr[kc1][ind1] = [kc2,kc1, delta] 
      rr[kc2][ind2] = [kc1,kc2,-delta]

     diff_data = [[data[j[0]]-data[j[1]]+j[2] for j in f] for f in rr]

     return np.array([np.einsum('js,s->j',geometry['weigths'][k,:,:geometry['n_non_boundary_side_per_elem'][k]],np.array(dt)) for k,dt in enumerate(diff_data)])



def fix_instability(F,B,scale=True):

   n_elems = F.shape[0]
   if scale:
    scale = 1/F.max(axis=0).toarray()[0]
   else: 
    scale = np.ones(n_elems)   
   n = np.random.randint(n_elems)
   scale[n] = 0
   F.data = F.data * scale[F.indices]
   F[n,n] = 1
   B[n] = 0   

   return scale



os.environ['H5PY_DEFAULT_READONLY']='1'

comm = MPI.COMM_WORLD


def extract_variables(solver):

   #Create variables from data--
   variables = {}
   variables['Temperature_BTE']      = {'data':solver['Temperature_BTE'],'units':'K','increment':[-1,0,0]}
   variables['Temperature_Fourier']  = {'data':solver['Temperature_Fourier'],'units':'K','increment':[-1,0,0]}
   variables['Flux_BTE']             = {'data':solver['Flux_BTE'],'units':'W/m/m','increment':[0,0,0]}
   variables['Flux_Fourier']         = {'data':solver['Flux_Fourier'],'units':'W/m/m','increment':[0,0,0]}
   if 'vorticity_BTE' in solver.keys():
    variables['vorticity_BTE']       = {'data':solver['vorticity_BTE']    ,'units':'W/m/m/m','increment':[0,0,0]}
   if 'vorticity_Fourier' in solver.keys():
    variables['vorticity_Fourier']   = {'data':solver['vorticity_Fourier'],'units':'W/m/m/m','increment':[0,0,0]}


   return variables


def compute_vorticity(geometry,J):
   """Compute vorticity (only for 2D cases)"""

   data = None  
   if comm.rank == 0:
    vorticity = np.zeros((len(geometry['elems']),3))
    grad_x = compute_grad_common(J[:,0],geometry)
    grad_y = compute_grad_common(J[:,1],geometry)
    #Defines only for 2D
    vorticity[:,2] = grad_y[:,0]-grad_x[:,1]
    data = {'vorticity':vorticity*1e9} #W/m/m/m
    
   return create_shared_memory_dict(data)
    
  



def expand_variables(data,geometry):

  dim     = int(geometry['meta'][2])
  n_elems = len(geometry['elems'])


  #Here we unroll variables for later use--
  variables = {}
  for key,value in data.items():
  
     if value['data'].ndim == 1: #scalar
       variables[key] = {'data':value['data'],'units':value['units'],'increment':value['increment']}
       n_elems = len(value['data'])
     elif value['data'].ndim == 2 : #vector 
         variables[key + '(x)'] = {'data':value['data'][:,0],'units':value['units'],'increment':value['increment']}
         variables[key + '(y)'] = {'data':value['data'][:,1],'units':value['units'],'increment':value['increment']}
         #if dim == 3: 
         variables[key + '(z)'] = {'data':value['data'][:,2],'units':value['units'],'increment':value['increment']}
         mag = np.array([np.linalg.norm(value) for value in value['data']])
         variables[key + '(mag.)'] = {'data':mag,'units':value['units'],'increment':value['increment']}

  variables['structure'] = {'data':np.zeros(n_elems),'units':'','increment':[0,0,0]}       

  return variables
 







def fast_interpolation(fine,coarse,bound=False,scale='linear') :


 if scale == 'log':
   #xmin    = min([np.min(fine),np.min(coarse)]) 
   #fine   -=xmin
   #coarse -=xmin
   fine    = np.log10(fine)
   coarse  = np.log10(coarse)
 #--------------

 m2 = np.argmax(coarse >= fine[:,np.newaxis],axis=1)
 m1 = m2-1
 a2 = (fine-coarse[m1])/(coarse[m2]-coarse[m1])
 a1 = 1-a2

 if bound == 'periodic':
  Delta = coarse[-1]-coarse[-2]  
  a = np.where(m2==0)[0] #this can be either regions
  m1[a] = len(coarse) -1
  m2[a] = 0
  fine[fine < Delta/2] += 2*np.pi 
  a2[a] = (fine[a] - coarse[-1])/ Delta
  a1 = 1-a2


 if bound == 'extent':

   #Small values
   al = np.where(fine<coarse[0])[0] 
   m2[al] = 1; 
   m1[al] = 0;
   a2[al] = (fine[al]-coarse[0])/ (coarse[1]-coarse[0])
   a1[al] = 1-a2[al]


   #Large values
   ar = np.where(fine>coarse[-1])[0]
   m2[ar] = len(coarse)-1; 
   m1[ar] = len(coarse)-2;
   a2[ar] = (fine[ar]-coarse[-2])/ (coarse[-1]-coarse[-2])
   a1[ar] = 1-a2[ar]


 return a1,a2,m1,m2






def periodic_kernel(x1, x2, p,l,variance):
    return variance*np.exp(-2/l/l * np.sin(np.pi*abs(x1-x2)/p) ** 2)

def gram_matrix(xs,p,l,variance):
    return [[periodic_kernel(x1,x2,p,l,variance) for x2 in xs] for x1 in xs]

def generate_random_interface(p,l,variance,scale):

 xs = np.arange(-p/2, p/2,p/200)
 mean = [0 for x in xs]
 gram = gram_matrix(xs,p,l,variance)
 ys = np.random.multivariate_normal(mean, gram)*scale
 f = interpolate.interp1d(xs, ys,fill_value='extrapolate')

 return f


def make_polygon(Na,A):

 
   dphi = 2.0*math.pi/Na;
   r = math.sqrt(2.0*A/Na/math.sin(2.0 * math.pi/Na))
   poly_clip = []
   for ka in range(Na):
     ph =  dphi/2 + (ka-1) * dphi
     px  = r * math.cos(ph) 
     py  = r * math.sin(ph) 
     poly_clip.append([px,py])


   return poly_clip  



def create_loop(loops,line_list,store):

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


def create_line_list(pp,points,lines,store,lx,ly,step_label = 'h'):

   def line_exists_ordered_old(l,lines):
    for n,line in enumerate(lines) :
     if (line[0] == l[0] and line[1] == l[1]) :
      return n+1
     if (line[0] == l[1] and line[1] == l[0]) :
      return -(n+1)
    return 0


   def already_included_old(all_points,new_point,p_list):

    dd = 1e-12
    I = []
    if len(all_points) > 0:
     I = np.where(np.linalg.norm(np.array(new_point)-np.array(all_points),axis=1)<dd)[0]
    if len(I) > 0: 
       return I[0]
    else: 
      return -1

   p_list = []
   for p in pp:
    tmp = already_included_old(points,p,p_list)
    if tmp == -1:    
      points.append(p)
      p_list.append(len(points)-1)
    else:
      p_list.append(tmp)

   for k,p in enumerate(p_list):
     point = points[-len(p_list)+k] 
     store.write( 'Point('+str(len(points)-len(p_list)+k) +') = {' + str(point[0]/lx) +'*lx,'+ str(point[1]/ly)+'*ly,0,' + step_label + '};\n')

   line_list = []
   for l in range(len(p_list)):
    p1 = p_list[l]
    p2 = p_list[(l+1)%len(p_list)]
    if not p1 == p2:
     tmp = line_exists_ordered_old([p1,p2],lines)

     if tmp == 0 : #craete line
      lines.append([p1,p2])
      store.write( 'Line('+str(len(lines)) +') = {' + str(p1) +','+ str(p2)+'};\n')
      line_list.append(len(lines))
     else:
      line_list.append(tmp)

   
   return line_list


 


def save_data(fh, namedict):

     if comm.rank == 0:
      with gzip.GzipFile(fh + '.npz', 'w') as f:
            pickle.dump(namedict, f,protocol=pickle.HIGHEST_PROTOCOL)

def load(fh):

    return  {f:load_data(f) for f in fh}


def load_data(fh):

    if comm.rank == 0:
     if os.path.isfile(fh + '.npz'):
      with gzip.open(fh + '.npz', 'rb') as f:
          return pickle.load(f)
     print("Can't load " + fh)
     quit()
     return -1 
    comm.Barrier()



def compute_neighbors(mesh,elem):
  
    guess = []

    for side in mesh['elem_side_map_vec'][elem]:
        for elem_2 in mesh['side_elem_map_vec'][side]:
            if not elem_2 in guess:
               guess.append(elem_2)
               
    return guess
                


def find_elem(mesh,p,guess):
 
    #First try in guess--
    for ne in guess:
     elem  = mesh['elems'][ne][0:mesh['side_per_elem'][ne]] 
     nodes = mesh['nodes'][elem][:,0:2] 
     polygon = Polygon(mesh['nodes'][elem][0:mesh['side_per_elem'][ne],0:2])
     if polygon.contains(Point(p[0],p[1])):
      return ne,nodes[:,0],nodes[:,1],True
    #------------------      

    #Find among all elements
    for ne in range(len(mesh['elems'])):

     elem  = mesh['elems'][ne] 
     nodes = mesh['nodes'][elem][:,0:2]

     polygon = Polygon(nodes)
     if polygon.contains(Point(p[0],p[1])):
      return ne,nodes[:,0],nodes[:,1],True


    return -1,-1,-1,False
        


def generate_frame(**argv):


    argv.setdefault('bounding_box',[0,0,0])
    Lx = float(argv.setdefault('lx',argv['bounding_box'][0]))
    Ly = float(argv.setdefault('ly',argv['bounding_box'][1]))
    frame = []
    frame.append([-Lx/2,Ly/2])
    frame.append([Lx/2,Ly/2])
    frame.append([Lx/2,-Ly/2])
    frame.append([-Lx/2,-Ly/2])

    return frame


def translate_shape(s,b,**argv):

  if argv.setdefault('relative',True):
     dx = 1
     dy = 1
  else:   
     dx = argv['lx']
     dy = argv['ly']

  out = []
  for p in s:
    tmp = [p[0] + b[0]*dx,p[1] + b[1]*dy]
    out.append(tmp)

  return out

def repeat_merge_scale(argv):

  polygons = argv['polygons']

  lx = argv['lx']   
  ly = argv['ly']   
  
  if argv.setdefault('relative',True):
    dx = 1;dy=1
  else:  
    dx = lx;dy=ly
  

  pbc = []
  pbc.append([0,0])
  if argv.setdefault('repeat',True):
   pbc.append([dx,0]) 
   pbc.append([-dx,0])
   pbc.append([0,dy])
   pbc.append([0,-dy])
   pbc.append([dx,dy])
   pbc.append([-dx,-dy])
   pbc.append([-dx,dy])
   pbc.append([dx,-dy])

  frame = Polygon([[-dx/2,-dy/2],[-dx/2,dy/2],[dx/2,dy/2],[dx/2,-dy/2]])

   #---------------------
  #Store only the intersecting polygons
  extended_base = []
  final = []
  for pp,poly in enumerate(polygons):

    for kp in range(len(pbc)):
        
     new_base = [argv['base'][pp][i]+pbc[kp][i] for i in range(2)]


     if not new_base in extended_base: #periodicity takes back to original shape

      tmp = [[ p[0] + pbc[kp][0],p[1] + pbc[kp][1] ] for p in poly] 

      p1 = Polygon(tmp)
      if p1.intersects(frame):
       extended_base.append(new_base)

       thin = Polygon(p1).intersection(frame)
       if isinstance(thin, shapely.geometry.multipolygon.MultiPolygon):
        tmp = list(thin)
        for t in tmp:
         final.append(t)
       else:
         final.append(p1)


  #Create bulk surface---get only the exterior to avoid holes
  MP = MultiPolygon(final) 

  conso = unary_union(MP)

  new_poly = []
  if isinstance(conso, shapely.geometry.multipolygon.MultiPolygon):
      for i in conso: 
       new_poly.append(list(i.exterior.coords))
  else: 
       new_poly.append(list(conso.exterior.coords))

  #Find the closest centroids--
  extended_base = np.array(extended_base)
  tmp = np.zeros_like(extended_base)
  for n,p in enumerate(new_poly):
     index = np.argmin(np.linalg.norm(extended_base - np.mean(p,axis=0)[np.newaxis,...],axis=1))
     tmp[n] = extended_base[index]
  extended_base = tmp.copy()
  #---------------------------------

  #cut redundant points

  if argv.setdefault('cut_redundant_point',False):

   new_poly_2 = []
   for poly in new_poly:
    N = len(poly)
    tmp = []
    for n in range(N):
     p1 = poly[n]
     p2 = poly[(n+1)%N]
     if np.linalg.norm(np.array(p1)-np.array(p2)) >1e-4:
      tmp.append(p1)
    new_poly2.append(tmp)
   new_poly = new_poly2.copy()

   #cut redundant points
   discard = 0
   while discard > 0:
      
    discard = 0  
    new_poly_2 = []
    for gg,poly in enumerate(new_poly):
     N = len(poly)
     tmp = []
     for n in range(N):
      p1 = np.array(poly[(n-1+N)%N])
      p = np.array(poly[n])
      p2 = np.array(poly[(n+1)%N])
      di = np.linalg.norm(np.cross(p-p1,p-p2))
      if di >1e-5:
         tmp.append(p)
      else:
         discard += 1 
     new_poly_2.append(tmp)
    new_poly = new_poly_2.copy()


  #----------------------------
  dmin = check_distances(final)
  #dmin = check_distances(new_poly)

  #scale-----------------------
  if argv.setdefault('relative',True):

   #Scale

   #Scale
   polygons = []
   for poly in new_poly:
    tmp = []
    for p in poly:
     g = list(p)
     g[0] *=lx
     g[1] *=lx
     tmp.append(g)
    polygons.append(tmp) 

   #Translate
   tmp = []
   for p,poly in enumerate(polygons):
     base = extended_base[p]
     delta = [0,base[1]*(ly-lx)]
     tmp.append(translate_shape(poly,delta))
   polygons = tmp   

  else:  
    polygons = new_poly.copy()

  argv['polygons'] = polygons
  argv['dmin'] = dmin







def check_distances(new_poly):


  #Merge pores is they are too close
  Pores = [ Polygon(p)  for p in new_poly]
  dmin = 1e4
  move = {}
  for p1 in range(len(Pores)):
   #check if intersect  
   
   for p2 in range(p1+1,len(Pores)):
   
       d = Pores[p1].distance(Pores[p2])
       if d > 0 and d < dmin : 
           dmin = d

   #top
   d = Pores[p1].distance(LineString([(-0.5,0.5),(0.5,0.5)])) 
   if d < dmin and d > 0:
     dmin = d
     #print('upper',p1)  

   #left
   d = Pores[p1].distance(LineString([(0.5,0.5),(0.5,-0.5)])) 
   if d < dmin and d > 0:
     dmin = d
     #print('right',p1)  

   #bottom
   d = Pores[p1].distance(LineString([(-0.5,-0.5),(0.5,-0.5)])) 
   if d < dmin and d > 0:
     dmin = d
     #print('bottom',p1)  
  
   #left
   d = Pores[p1].distance(LineString([(-0.5,-0.5),(-0.5,0.5)])) 
   if d < dmin and d > 0:
     #print('left',p1)  
     dmin = d

   #print(dmin)

  return dmin



def get_linear_indexes(mfp,value,scale,extent):

   if value == 0:
     return -1,-1,-1,-1

   n = len(mfp)

   if scale == 'log':
     mfp = np.log10(mfp.copy())
     value = np.log10(value.copy())

   found = False
   beyond = False
   if extent:
    if value < mfp[0]:
      beyond = True
      found = True
      i=0;j=1;
    elif value > mfp[-1]:
      i=n-2;j=n-1;
      beyond = True
      found = True

   if beyond == False:
    for m in range(n-1):
      if (value <= mfp[m+1]) and (value >= mfp[m]) :
        i = m; j = m + 1 
        found = True
        break
  
   if found == False:
      print('no interpolation found')
   else:  
    aj = (value-mfp[i])/(mfp[j]-mfp[i]) 
    ai = 1-aj
    if scale=='inverse':
     ai *=mfp[i]/value
     aj *=mfp[j]/value
   return i,ai,j,aj  


def load_shared(filename):

    data = None
    if comm.rank == 0:
      data = load_data(filename)
    data =   create_shared_memory_dict(data)  

    return data


def shared_array(value):

    data = {'dummy':value}
    return create_shared_memory_dict(data)['dummy']


def sparse_dense_product(i,j,data,X):
   #  This solves B_ucc' X_uc' -> A_uc

   #  B_ucc' : sparse in cc' and dense in u. Data is its vectorized data COO descrition
   #  X      : dense matrix
   #  '''

     tmp = np.zeros_like(X)
     np.add.at(tmp.T,i,data.T * X.T[j])

     return tmp

def compute_polar(mfp_bulk):
     phi_bulk = np.array([np.arctan2(m[0],m[1]) for m in mfp_bulk])
     phi_bulk[np.where(phi_bulk < 0) ] = 2*np.pi + phi_bulk[np.where(phi_bulk <0)]
     r = np.linalg.norm(mfp_bulk[:,:2],axis=1) #absolute values of the projection
     return r,phi_bulk 


def compute_spherical(mfp_bulk):
 r = np.linalg.norm(mfp_bulk,axis=1) #absolute values of the projection
 phi_bulk = np.array([np.arctan2(m[0],m[1]) for m in mfp_bulk])
 phi_bulk[np.where(phi_bulk < 0) ] = 2*np.pi + phi_bulk[np.where(phi_bulk <0)]
 theta_bulk = np.array([np.arccos((m/r[k])[2]) for k,m in enumerate(mfp_bulk)])

 return r,phi_bulk,theta_bulk


def create_shared_memory_dict(varss):

       dtype = [np.int32,np.int64,np.float32,np.float64]

       dict_output = {}
       if comm.Get_rank() == 0:
          var_meta = {} 
          for var,value in varss.items():
           if callable(value) or type(value) == str or type(value) == int or type(value) == float:
              var_meta[var] = [None,None,None,None,False] 
              continue 
           if type(value) == list: 
              value = np.array(value)   

           #Check types
           if   value.dtype == np.int32:
                data_type = 0
                itemsize = MPI.INT32_T.Get_size()
           elif value.dtype == np.int64:
                data_type = 1
                itemsize = MPI.INT64_T.Get_size()
           elif value.dtype == np.float32:
                data_type = 2
                itemsize = MPI.FLOAT.Get_size() 
           elif value.dtype == np.float64:
                data_type = 3
                itemsize = MPI.DOUBLE.Get_size() 
           else:
              var_meta[var] = [None,None,None,None,False] 
              print('data type for shared memory not supported for ' + var)
              continue 

           size = np.prod(value.shape)
           nbytes = size * itemsize
           var_meta[var] = [value.shape,data_type,itemsize,nbytes,True]
       else: nbytes = 0; var_meta = None
       var_meta = comm.bcast(var_meta,root=0)
       #ALLOCATING MEMORY---------------
       for n,(var,meta) in enumerate(var_meta.items()):
        if meta[-1]:
         win = MPI.Win.Allocate_shared(meta[3],meta[2], comm=comm) 
         buf,itemsize = win.Shared_query(0)
         assert itemsize == meta[2]
         dt = dtype[meta[1]]

         output = np.ndarray(buffer=buf,dtype=dt,shape=meta[0]) 
         if comm.rank == 0: output[:] = varss[var]
         dict_output[var] = output
        else:
            if comm.rank == 0:  
               dict_output[var] = varss[var]   

       del varss
       comm.Barrier()

       return dict_output
   


def store_shared(func,*argv):
    """ Run the function func with positional argument argv on proc 0 and broadcast the results on all processors"""
    if comm.rank == 0:
        output = func(*argv)
    else:
        output = None

    return create_shared_memory_dict(output)     




def duplicate_cells(geometry,variables,repeat,displ):

   dim = int(geometry['meta'][2])
   nodes = np.round(geometry['nodes'],4)
   n_nodes = len(nodes)
   #Correction in the case of user's mistake
   if dim == 2:
     repeat[2]  = 1  
   #--------------------  
   size = geometry['size']

   #Create periodic vector
   P = []
   for px in size[0]*np.arange(repeat[0]):
    for py in size[1]*np.arange(repeat[1]):
     for pz in size[2]*np.arange(repeat[2]):
       P.append([px,py,pz])  
   P = np.asarray(P[1:],np.float32)

   #Compute periodic nodes------------------
   pnodes = []
   for s in list(geometry['periodic_sides'])+ list(geometry['inactive_sides']):
     pnodes += list(geometry['sides'][s])
   pnodes = list(np.unique(np.array(pnodes)))
   #--------------------------------------------

   #Repeat Nodes
   #-----------------------------------------------------
   def repeat_cell(axis,n,d,nodes,replace,displ):

    tmp = nodes.copy()
    for x in np.arange(n-1):
     tmp[:,axis] +=  size[axis] + displ

     nodes = np.vstack((nodes,tmp))
    
    return nodes,replace

   replace = {}  
   for i in range(int(geometry['meta'][2])):
     nodes,replace = repeat_cell(i,repeat[i],size[i],nodes,replace,displ[i])
   #---------------------------------------------------

   #Repeat elements----------
   unit_cell = np.array(geometry['elems']).copy()
  
   elems = geometry['elems'] 
   for i in range(np.prod(repeat)-1):
     elems = np.vstack((elems,unit_cell+(i+1)*(n_nodes)))

   #duplicate variables---
   for n,(key, value) in enumerate(variables.items()):  
     unit_cell = value['data'].copy() 
     for nz in range(repeat[2]):
      for ny in range(repeat[1]):
       for nx in range(repeat[0]):
           if nx + ny + nz > 0:  
            inc = value['increment'][0]*nx + value['increment'][1]*ny + value['increment'][2]*nz
            tmp = unit_cell - inc
            if value['data'].ndim == 1:
             value['data'] = np.hstack((value['data'],tmp))
            else: 
             value['data'] = np.vstack((value['data'],tmp))

     variables[key]['data'] = value['data']

   geometry['elems'] = elems
   geometry['nodes'] = nodes

   #return the new side
   size = [geometry['size'][i] *repeat[i]   for i in range(dim)]

   return size




def get_node_data(variables,geometry):


 var0 = list(variables.keys())[0]


 if not len(variables[var0]['data']) == len(geometry['nodes']):

  dim = int(geometry['meta'][2])
  for key,tmp in variables.items():
   data = tmp['data']
   #NEW-------------
   conn = np.zeros(len(geometry['nodes']))
   if data.ndim == 2:
       node_data = np.zeros((len(geometry['nodes']),3))
   elif data.ndim == 3:   
       node_data = np.zeros((len(geometry['nodes']),3,3))
   else:   
       node_data = np.zeros(len(geometry['nodes']))
    
   #This works only with uniform type of elements 
   elem_flat = np.array(geometry['elems']).flat
   np.add.at(node_data,elem_flat,np.repeat(data,len(geometry['elems'][0]),axis=0))
   np.add.at(conn,elem_flat,np.ones_like(elem_flat))

   if data.ndim == 2:
       np.divide(node_data,conn[:,np.newaxis],out=node_data)
   elif data.ndim == 3:
       np.divide(node_data,conn[:,np.newaxis,np.newaxis],out=node_data)
   else: 
       np.divide(node_data,conn,out=node_data)
   #-----------------------
   variables[key]['data'] = node_data
   

def repeat_nodes_and_data(data_uc,nodes_uc,increment,repeat,size,cells_uc):

   Nx = repeat[0] 
   Ny = repeat[1] 
   Nz = repeat[2] 

   #----------------------------------------------
   n_nodes = len(nodes_uc)

   nodes = []
   data = []
   #add nodes---
   corr = np.zeros(n_nodes*Nx*Ny*Nz,dtype=np.int)
   index = 0
   for nx in range(Nx):
    for ny in range(Ny):
     for nz in range(Nz):
      dd = nx*increment[0] + ny*increment[1] + nz*increment[2]
      index = nx * Ny * Nz * n_nodes +  ny *  Nz * n_nodes + nz * n_nodes
      P = [round(size[0]*nx,3),round(size[1]*ny,3),round(size[2]*nz,3)]
      for n in range(n_nodes):
       node_trial = [round(nodes_uc[n,0],4) + P[0]-(Nx-1)*0.5*size[0],\
                     round(nodes_uc[n,1],4) + P[1]-(Nx-1)*0.5*size[1],\
                     round(nodes_uc[n,2],4) + P[2]-(Nx-1)*0.5*size[2]]

       nodes.append(node_trial)
       data.append(data_uc[n]-dd)
       corr[index+n] = len(nodes)-1

   cells = []
   for nx in range(Nx):
    for ny in range(Ny):
     for nz in range(Nz):
      P = [size[0]*nx,size[1]*ny,size[2]*nz]
      index = nx * Ny * Nz * n_nodes +  ny *  Nz * n_nodes + nz * n_nodes
      #add cells
      for gg in cells_uc:
       tmp = []
       for n in gg:
        tmp.append(corr[n+index])
       cells.append(tmp)

   return np.array(nodes),np.array(cells),np.array(data)
        

def create_shared_memory(varss):
       for var in varss:
         #--------------------------------------
         if comm.Get_rank() == 0: 
          tmp = eval('self.' + var)
          if tmp.dtype == np.int64:
              data_type = 0
              itemsize = MPI.INT.Get_size() 
          elif tmp.dtype == np.float64:
              data_type = 1
              itemsize = MPI.DOUBLE.Get_size() 
          else:
              print('data type for shared memory not supported')
              quit()
          size = np.prod(tmp.shape)
          nbytes = size * itemsize
          meta = [tmp.shape,data_type,itemsize]
         else: nbytes = 0; meta = None
         meta = comm.bcast(meta,root=0)

         #ALLOCATING MEMORY---------------
         win = MPI.Win.Allocate_shared(nbytes,meta[2], comm=comm) 
         buf,itemsize = win.Shared_query(0)
         assert itemsize == meta[2]
         dt = 'i' if meta[1] == 0 else 'd'
         output = np.ndarray(buffer=buf,dtype=dt,shape=meta[0]) 

         if comm.rank == 0:
             output[:] = tmp  

         exec('self.' + var + '=output')


