from mpi4py import MPI
import numpy as np
from termcolor import colored, cprint 
from shapely.geometry import Polygon,Point
from shapely.geometry import MultiPolygon,LineString
import shapely
from shapely.ops import cascaded_union
import os,sys
import functools
import io,zipfile
import math
import pickle
import gzip
import time
import matplotlib.tri as mtri





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

def compute_kappa(W,b):


 L   = lambda x: np.dot(W,x)
 M   = lambda x: x/np.sqrt(np.diag(W))

 def callback(x):
     return np.dot(x,b)

 x0 = b.copy()/np.diag(W)
 x  = cg(L,b,M=M,x0=x0,callback=callback,verbose = False)

 return callback(x)


os.environ['H5PY_DEFAULT_READONLY']='1'

comm = MPI.COMM_WORLD

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
     return -1 




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
  a = time.time()
  final = []
  for pp,poly in enumerate(polygons):
    for kp in range(len(pbc)):
        
     tmp = [[ p[0] + pbc[kp][0],p[1] + pbc[kp][1] ] for p in poly] 

     p1 = Polygon(tmp)
     if p1.intersects(frame):
      thin = Polygon(p1).intersection(frame)
      if isinstance(thin, shapely.geometry.multipolygon.MultiPolygon):
       tmp = list(thin)
       for t in tmp:
        final.append(t)
      else:
        final.append(p1)


  #Create bulk surface---get only the exterior to avoid holes
  MP = MultiPolygon(final) 

  conso = cascaded_union(MP)

  new_poly = []
  if isinstance(conso, shapely.geometry.multipolygon.MultiPolygon):
      for i in conso: 
       new_poly.append(list(i.exterior.coords))
  else: 
       new_poly.append(list(conso.exterior.coords))
  

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
   polygons = []
   for poly in new_poly:
    tmp = []
    for p in poly:
     g = list(p)
     g[0] *=lx
     g[1] *=ly
     tmp.append(g)
    polygons.append(tmp) 
  else:  
    polygons = new_poly.copy()

  argv['polygons'] = polygons
  argv['dmin'] = dmin






def compute_line_data(mesh,solver,**argv):

 data = solver['variables'][argv['variable']]['data']
 
 guess = []
 old_mat = 1
 p_old = [-1e3,-1e3]
 crossing = []
 pair = []
 p1 = np.array(argv['p1'])
 p2 = np.array(argv['p2'])
 N = argv['N']

 L = []
 tmp_L = []
 output = []
 tmp_output = []
 already_passed = False
 dx = []
 for kk in range(N):

   p = p1 + (p2 - p1)*kk/(N-1)

   if len(dx) == 0:
    dx.append(0)
   else: 
    dx.append(dx[-1] + np.linalg.norm(p-p_old))
    

   elem,x,y,found = find_elem(mesh,p,guess)

   if found:

    already_passed = False
    new_elem = elem

    nodes = mesh['elems'][elem]

    if len(nodes) == 3:
     d = mtri.LinearTriInterpolator(mtri.Triangulation(x, y,[[0,1,2]]),data[nodes])(p[0],p[1]).data
    else:
     d = interpolate.interp2d(x, y, data[nodes], kind='linear')(p[0],p[1])[0]

    tmp_output.append(float(d))

    tmp_L.append(dx[-1])

   else:
       if not already_passed :
        L.append(tmp_L)   
        output.append(tmp_output)
        tmp_output = []
        tmp_L = []
        already_passed = True

   guess = compute_neighbors(mesh,elem)  

   old_elem = new_elem
   p_old = p.copy()


 L.append(tmp_L)   
 output.append(tmp_output)


 return L,output



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



def interpolate(vector,value,bounds='extent',period = None):

   n = len(vector)

   if value >= vector[0] and value <= vector[-1]:
     
    for m in range(n-1):
      if (value <= vector[m+1]) and (value >= vector[m]) :
        i = m; j = m + 1 
        break
    aj = (value-vector[i])/(vector[j]-vector[i]) 
    ai = 1-aj
    return i,ai,j,aj  

   else:

    if bounds == 'extent': 
     if value < vector[0]:
       i=0;j=1;
     elif value > vector[-1]:
       i=n-2;j=n-1;
     aj = (value-vector[i])/(vector[j]-vector[i]) 
     ai = 1-aj
     return i,ai,j,aj 

    elif bounds == 'periodic':     
     i=n-1;j=0;
     if value < vector[0]:
       aj = (value + period -vector[i])/(vector[j] + period -vector[i])
     elif value > vector[-1]:
       aj = (value - vector[-1])/(vector[0] + period - vector[-1]) 

     ai = 1-aj
     return i,ai,j,aj 




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


def shared_array(value):

    data = {'dummy':value}
    return create_shared_memory_dict(data)['dummy']


def get_kappa_map_from_mat(**argv):

        mat_map = argv['geometry']['elem_mat_map']
        kappa =   argv['material']['kappa']

        kappa = np.array(kappa)
        if kappa.ndim == 3:
         return np.array([ list(kappa[i])  for i in mat_map])
        else:
         return np.array([ list(kappa)  for i in mat_map])
            


def create_shared_memory_dict(varss):

       dict_output = {}
       if comm.Get_rank() == 0:
          var_meta = {} 
          for var,value in varss.items():

           if type(value) == list: value = np.array(value)   
           if value.dtype == np.int64:
              data_type = 0
              itemsize = MPI.INT.Get_size()
           elif value.dtype == np.float64:
              data_type = 1
              itemsize = MPI.DOUBLE.Get_size() 
           else:
              print('data type for shared memory not supported for ' + var)
              quit()

           size = np.prod(value.shape)
           nbytes = size * itemsize
           var_meta[var] = [value.shape,data_type,itemsize,nbytes]

       else: nbytes = 0; var_meta = None
       var_meta = comm.bcast(var_meta,root=0)

       #ALLOCATING MEMORY---------------
       for n,(var,meta) in enumerate(var_meta.items()):
       
        win = MPI.Win.Allocate_shared(meta[3],meta[2], comm=comm) 
        buf,itemsize = win.Shared_query(0)
        assert itemsize == meta[2]
        dt = 'i' if meta[1] == 0 else 'd'
        output = np.ndarray(buffer=buf,dtype=dt,shape=meta[0]) 
        if comm.rank == 0: output[:] = varss[var]
        dict_output[var] = output

       del varss
       comm.Barrier()

       return dict_output


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


