import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pyvtk import *
import numpy as np
import os
from .utils import *
from .viewer import *
import matplotlib
from matplotlib.pylab import *
from .kappa_mode import *
from scipy.spatial import distance
import numpy.ma as ma


from mpi4py import MPI
comm = MPI.COMM_WORLD
class Plot(object):

 def __init__(self,**argv):

  if comm.rank == 0:

   #import data-------------------------------
   self.mesh     = argv['geometry'] if 'geometry' in argv.keys() else load_data('geometry')
   self.solver     = argv['solver'] if 'geometry' in argv.keys() else load_data('solver')
   self.material = argv['material'] if 'material' in argv.keys() else load_data('material')

   self.dim = int(self.mesh['meta'][2])

   if argv.setdefault('user',False):
    argv['model'](self.material,self.solver,self.mesh)   
   else:    
    model = argv['model'].split('/')[0]
    if model == 'structure':
     self.duplicate_cells(**argv)
     self.plot_structure(**argv)

    elif model == 'maps':
     self.duplicate_cells(**argv)
     self.plot_maps(**argv)

    elif model == 'matplotlib':
     self.plot_matplotlib(**argv)

    elif model == 'kappa_mode':
        plot_mode_kappa(**argv)

    elif model == 'vtu':
     #self.duplicate_cells(**argv)
     #self.write_cell_vtu(**argv)  
     self.write_vtu(**argv)   

 def duplicate_cells(self,**argv):

   bb = time.time()   
   nodes = np.round(self.mesh['nodes'],4)
   n_nodes = len(nodes)
   repeat = argv.setdefault('repeat',[1,1,1])
   size = self.mesh['size']

   #Create periodic vector
   P = []
   for px in size[0]*np.arange(repeat[0]):
    for py in size[1]*np.arange(repeat[1]):
     for pz in size[2]*np.arange(repeat[2]):
       P.append([px,py,pz])  
   P = np.asarray(P[1:],np.float32)

   #Compute periodic nodes------------------
   pnodes = []
   for s in list(self.mesh['periodic_sides'])+ list(self.mesh['inactive_sides']):
     pnodes += list(self.mesh['sides'][s])
   pnodes = list(np.unique(np.array(pnodes)))
   #--------------------------------------------

   #Repeat Nodes
   #-----------------------------------------------------
   def repeat_cell(axis,n,d,nodes,replace):

    tmp = nodes.copy()
    for x in np.arange(n-1):
     tmp[:,axis] +=  size[axis]

     #Mask arrays are used in order not to consider nodes that have been already assigned to their periodic ones.
     #intersecting = np.nonzero(distance.cdist(tmp,ma.array(nodes, \
     #                          mask = replace.keys()))<1e-4)
     #for new,old in zip(*intersecting): replace[new] = old

     nodes = np.vstack((nodes,tmp))
    
    return nodes,replace

   replace = {}  
   for i in range(int(self.mesh['dim'])):
     nodes,replace = repeat_cell(i,repeat[i],size[i],nodes,replace)
   #---------------------------------------------------

   #Repeat elements----------
   unit_cell = self.mesh['elems'].copy()
   
   elems = self.mesh['elems'] 
   for i in range(np.prod(repeat)-1):
     elems = np.vstack((elems,unit_cell+(i+1)*(n_nodes)))
   #------------------------

   #duplicate variables---
   for n,(key, value) in enumerate(self.solver['variables'].items()):
      
     unit_cell = value['data'].copy()  
     for nz in range(repeat[0]):
      for ny in range(repeat[1]):
       for nx in range(repeat[2]):
           if nx + ny + nz > 0:   
            inc = value['increment'][0]*nx + value['increment'][1]*ny + value['increment'][2]*nz
            tmp = unit_cell - inc
            if value['data'].ndim == 1:
             value['data'] = np.hstack((value['data'],tmp))
            else: 
             value['data'] = np.vstack((value['data'],tmp))

     self.solver['variables'][key]['data'] = value['data']


   self.mesh['elems'] = elems
   self.mesh['nodes'] = nodes

 
 def duplicate_cells_old(self,**argv):

   #Compute periodic_nodes
   nodes = np.round(self.mesh['nodes'],4)
   pnodes = []
   for s in list(self.mesh['periodic_sides'])+ list(self.mesh['inactive_sides']):
     pnodes += list(self.mesh['sides'][s])
   pnodes = list(np.unique(np.array(pnodes)))

   #repeat elements
   bb = time.time()
   #---------------------------
   size = self.mesh['size']
   n_nodes = len(nodes)
   repeat = argv.setdefault('repeat',[1,1,1])
   Nx = repeat[0] 
   Ny = repeat[1] 
   Nz = repeat[2] 
   new_nodes = list(nodes.copy())
   corr = np.zeros(n_nodes*Nx*Ny*Nz,dtype=int)
   corr[:n_nodes] = range(n_nodes)
   for nx in range(Nx):
    for ny in range(Ny):
     for nz in range(Nz):
      P = [size[0]*nx,size[1]*ny,size[2]*nz]
      index = nx * Ny * Nz * n_nodes +  ny *  Nz * n_nodes + nz * n_nodes
      print(index)
      if index > 0:
       for n in range(n_nodes):
        node_trial = [nodes[n,0] + P[0],\
                      nodes[n,1] + P[1],\
                      nodes[n,2] + P[2]]
 
        if n in pnodes:
         a = np.where(np.linalg.norm(np.array(new_nodes)[pnodes]-node_trial,axis=1)<1e-3)[0]
         if len(a) > 0:
           corr[index+n] = pnodes[a[0]]
         else:    
           new_nodes.append(node_trial)
           pnodes.append(len(new_nodes)-1)
           corr[index+n] = len(new_nodes)-1
        else:  
         new_nodes.append(node_trial)
         corr[index+n] = len(new_nodes)-1

   new_nodes = np.array(new_nodes)

   #Duplicate Elements
   cells = []
   mapp = {}
   for nx in range(Nx):
    for ny in range(Ny):
     for nz in range(Nz):
      index = nx * Ny * Nz * n_nodes +  ny *  Nz * n_nodes + nz * n_nodes
      for k,gg in enumerate(self.mesh['elems']):
        tmp = []
        for n in gg:
         tmp.append(corr[n+index])
        cells.append(tmp)
        mapp.setdefault(k,[]).append(len(cells)-1)

   #Duplicate sides---------------------------------
   self.surface_sides = []
   self.surface_nodes = []
   self.indices = []
   for nx in range(Nx):
    for ny in range(Ny):
     for nz in range(Nz):
      index = nx * Ny * Nz * n_nodes +  ny *  Nz * n_nodes + nz * n_nodes
      for s in list(self.mesh['boundary_sides']) + list(self.mesh['periodic_sides']) + list(self.mesh['inactive_sides']):
        nodes = self.mesh['sides'][s]  
        tmp = []
        for n in nodes:
         self.surface_nodes.append(new_nodes[corr[n+index]])   
         self.indices.append(corr[n+index])
         tmp.append(len(self.surface_nodes)-1)
        self.surface_sides.append(tmp)

   self.surface_sides = np.array(self.surface_sides)     
   self.surface_nodes = np.array(self.surface_nodes)     
   self.indices = np.array(self.indices)     
   #-------------------------------------------------

   #duplicate variables---
   for n,(key, value) in enumerate(self.solver['variables'].items()):
     if value['data'].ndim == 1:
      tmp = np.zeros(len(self.mesh['elems'])*np.prod(argv['repeat']))
     else: 
      tmp = np.zeros((len(self.mesh['elems'])*np.prod(argv['repeat']),self.dim))

     #NEW
     g = 0
     for nx in range(Nx):
      for ny in range(Ny):
       for nz in range(Nz):
        inc = value['increment'][0]*nx + value['increment'][1]*ny + value['increment'][2]*nz
        for k,gg in enumerate(self.mesh['elems']):
          tmp[g] = value['data'][k] - inc
          g +=1

     #OLD
     #for k,v in mapp.items():
     #  for vv in v: tmp[vv] = value['data'][k]  

     self.solver['variables'][key]['data'] = tmp

   self.mesh['nodes'] = new_nodes
   self.mesh['elems'] = cells



 def write_cell_vtu(self,**argv):

   output = []
   strc = 'CellData('
   for n,(key, value) in enumerate(self.solver['variables'].items()):

     #if  value['data'].shape[0] == 1:  value['data'] =  value['data'][0] #band-aid solution

     #if value['data'].ndim == 1:
     # tmp = np.zeros(len(self.mesh['elems'])*np.prod(argv['repeat']))
     #else: 
     # tmp = np.zeros((len(self.mesh['elems'])*np.prod(argv['repeat']),3))
     #for k,v in mapp.items():
     #  for vv in v:
     #    tmp[vv] = value['data'][k]  

     #output.append(tmp)

     output.append(value['data'])
     name = value['name'] + '[' + value['units'] + ']'
     if value['data'].ndim == 1:
       strc += r'''Scalars(output[''' + str(n) + r'''],name =' ''' + name +  r''' ')'''
    
     if value['data'].ndim == 2:
       strc += r'''Vectors(output[''' + str(n) + r'''],name =' ''' + name +  r''' ')'''

     if n == len(self.solver['variables'])-1:
      strc += ')'
     else:
      strc += ','

   data = eval(strc)
   if self.mesh['dim'] == 3:
    vtk = VtkData(UnstructuredGrid(self.mesh['nodes'],tetra=self.mesh['elems']),data)
   else :
    vtk = VtkData(UnstructuredGrid(self.mesh['nodes'][:,:2],triangle=self.mesh['elems']),data)

   vtk.tofile('output','ascii')


 def write_vtu(self,**argv):
   self.duplicate_cells(**argv)
   for key in self.solver['variables'].keys():
       self.solver['variables'][key]['data'] = self._get_node_data(self.solver['variables'][key]['data'])

   output = []
   strc = 'PointData('
   for n,(key, value) in enumerate(self.solver['variables'].items()):


     if  value['data'].shape[0] == 1:  value['data'] =  value['data'][0] #band-aid solution

     name = value['name'] + '[' + value['units'] + ']'
     if value['data'].ndim == 1:
       strc += r'''Scalars(output[''' + str(n) + r'''],name =' ''' + name +  r''' ')'''
       output.append(value['data'])
    
     if value['data'].ndim == 2:

       if self.mesh['dim'] == 2:
        t = np.zeros_like(value['data'])
        value['data'] = np.concatenate((value['data'],t),axis=1)[:,:3]
       output.append(value['data'])

       strc += r'''Vectors(output[''' + str(n) + r'''],name =' ''' + name +  r''' ')'''

     if n == len(self.solver['variables'])-1:
      strc += ')'
     else:
      strc += ','

   data = eval(strc)

   if self.mesh['dim'] == 3:
    vtk = VtkData(UnstructuredGrid(self.mesh['nodes'],tetra=self.mesh['elems']),data)
   else :
    nodes = self.mesh['nodes']
    vtk = VtkData(UnstructuredGrid(nodes,triangle=self.mesh['elems']),data)
   
   vtk.tofile('output','ascii')


 def _get_node_data(self,data,indices=[]):

   #NEW-------------
   conn = np.zeros(len(self.mesh['nodes']))
   if data.ndim > 1:
       node_data = np.zeros((len(self.mesh['nodes']),int(self.dim)))
   else:   
       node_data = np.zeros(len(self.mesh['nodes']))
    
   #This works only with uniform type of elements 
   elem_flat = self.mesh['elems'].flat
   np.add.at(node_data,elem_flat,np.repeat(data,len(self.mesh['elems'][0]),axis=0))
   np.add.at(conn,elem_flat,np.ones_like(elem_flat))
   if data.ndim > 1:
       np.divide(node_data,conn[:,np.newaxis],out=node_data)
   else: 
       np.divide(node_data,conn,out=node_data)
   #-----------------------

   if len(indices) > 0: 
    node_data = node_data[indices]

   return node_data


 def get_surface_nodes(self):

     triangles = []
     nodes = []
     for l in  list(self.mesh['boundary_sides']) + \
               list(self.mesh['periodic_sides']) + \
               list(self.mesh['inactive_sides']) :
          tmp = []
          for i in self.mesh['sides'][l]:
              #if i in nodes:  
              # k = nodes.index(i)   
              #else: 
               nodes.append(i)
               k = len(nodes)-1
               tmp.append(k)
          triangles.append(tmp)


     return np.array(triangles),nodes

 
 def plot_structure(self,**argv):

    data = {0:{'name':'Structure','units':'','data':np.zeros(len(self.mesh['nodes'] if self.dim == 2 else self.indices))}}
   

    plot_results(data,np.array(self.mesh['nodes']) if self.dim == 2 else self.surface_nodes,\
                                        np.array(self.mesh['elems']) if self.dim == 2 else self.surface_sides,**argv)

 def plot_maps(self,**argv):

   ind = self.mesh['meta'][-1]
   argv.update({'bulk':self.material['kappa'][int(ind),int(ind)],'fourier':self.solver['kappa_fourier']})
   if len(self.solver['variables'])> 2:
    argv.update({'bte':self.solver['kappa'][-1]})



   for key in self.solver['variables'].keys():
        self.solver['variables'][key]['data'] = self._get_node_data(self.solver['variables'][key]['data'],\
                indices= self.indices if self.dim == 3 else range(len(self.mesh['nodes'])))
   

   self.solver['variables'][-1] = {'name':'Structure','units':'','data':np.zeros(len(self.mesh['nodes'] if self.dim == 2 else self.indices))}

   if argv.setdefault('save_plot',False):
    dd.io.save('plot.h5',{'variables':self.solver['variables'],'nodes':self.mesh['nodes'] if self.dim == 2 else self.surface_nodes,\
                                                               'elems':self.mesh['elems'] if self.dim == 2 else self.surface_sides})

   self.fig = plot_results(self.solver['variables'],np.array(self.mesh['nodes']) if self.dim == 2 else self.surface_nodes,\
                                          np.array(self.mesh['elems']) if self.dim == 2 else self.surface_sides,**argv)

