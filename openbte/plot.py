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
   self.solver   = argv['solver']   if 'geometry' in argv.keys() else load_data('solver')
   self.material = argv['material'] if 'material' in argv.keys() else load_data('material')

   self.dim = int(self.mesh['meta'][2])

   if argv.setdefault('user',False):
    argv['user_model'].plot(self.material,self.solver,self.mesh)   
   else:    
    model = argv['model'].split('/')[0]
    #if model == 'structure':
    # self.duplicate_cells(**argv)
    # self.plot_structure(**argv)

    if model == 'maps':
     self.get_node_data()
     if self.dim == 3:
      self.get_surface_nodes()
     self.duplicate_cells(**argv)

     self.plot_maps(**argv)

    elif model == 'matplotlib':
     self.plot_matplotlib(**argv)

    elif model == 'kappa_mode':
        plot_mode_kappa(**argv)

    elif model == 'vtu':
     self.get_node_data()
     self.duplicate_cells(**argv)
     self.write_vtu(**argv)  

 def duplicate_cells(self,**argv):

   bb = time.time()   
   nodes = np.round(self.mesh['nodes'],4)
   n_nodes = len(nodes)
   repeat = argv.setdefault('repeat',[1,1,1])
   #Correction in the case of user's mistake
   if self.dim == 2:
     repeat[2]  = 1  
   #--------------------  
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
   for i in range(int(self.mesh['meta'][2])):
     nodes,replace = repeat_cell(i,repeat[i],size[i],nodes,replace)
   #---------------------------------------------------

   #Repeat elements----------
   unit_cell = np.array(self.mesh['elems']).copy()
  
   elems = self.mesh['elems'] 
   for i in range(np.prod(repeat)-1):
     elems = np.vstack((elems,unit_cell+(i+1)*(n_nodes)))

   #------------------------

   #duplicate variables---
   for n,(key, value) in enumerate(self.solver['variables'].items()):  
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

     self.solver['variables'][key]['data'] = value['data']


   self.mesh['elems'] = elems
   self.mesh['nodes'] = nodes

 

 def write_cell_vtu(self,**argv):

   output = []
   strc = 'CellData('
   for n,(key, value) in enumerate(self.solver['variables'].items()):

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

   output = []
   strc = 'PointData('
   for n,(key, value) in enumerate(self.solver['variables'].items()):

     if  value['data'].shape[0] == 1:  value['data'] =  value['data'][0] #band-aid solution

     name = value['name'] + '[' + value['units'] + ']'
     if value['data'].ndim == 1:
       strc += r'''Scalars(output[''' + str(n) + r'''],name =' ''' + name +  r''' ')'''
       output.append(value['data'])
    
     if value['data'].ndim == 2:

       if self.dim == 2:
        t = np.zeros_like(value['data'])
        value['data'] = np.concatenate((value['data'],t),axis=1)[:,:3]
       output.append(value['data'])

       strc += r'''Vectors(output[''' + str(n) + r'''],name =' ''' + name +  r''' ')'''

     if n == len(self.solver['variables'])-1:
      strc += ')'
     else:
      strc += ','

   data = eval(strc)

   if self.dim == 3:
   
    vtk = VtkData(UnstructuredGrid(self.mesh['nodes'],tetra=self.mesh['elems']),data)
   else :
    vtk = VtkData(UnstructuredGrid(self.mesh['nodes'],triangle=self.mesh['elems']),data)
   
   vtk.tofile('output','ascii')


 def get_node_data(self):

  for key in self.solver['variables'].keys():
   data = self.solver['variables'][key]['data']

   #NEW-------------
   conn = np.zeros(len(self.mesh['nodes']))
   if data.ndim > 1:
       node_data = np.zeros((len(self.mesh['nodes']),int(self.dim)))
   else:   
       node_data = np.zeros(len(self.mesh['nodes']))
    
   #This works only with uniform type of elements 
   elem_flat = np.array(self.mesh['elems']).flat
   np.add.at(node_data,elem_flat,np.repeat(data,len(self.mesh['elems'][0]),axis=0))
   np.add.at(conn,elem_flat,np.ones_like(elem_flat))

   if data.ndim > 1:
       np.divide(node_data,conn[:,np.newaxis],out=node_data)
   else: 
       np.divide(node_data,conn,out=node_data)
   #-----------------------
   self.solver['variables'][key]['data'] = node_data

  ##Add structure
  #self.solver['variables'][-1] = {'name':'Structure','units':'','data':np.zeros(len(self.mesh['nodes']))}



 def get_surface_nodes(self):

     sides = list(self.mesh['boundary_sides']) + \
                     list(self.mesh['periodic_sides']) + \
                     list(self.mesh['inactive_sides'])

     nodes = self.mesh['sides'][sides].flat


     triangles = np.arange(len(nodes)).reshape((len(sides),3))
     
     for key in self.solver['variables'].keys():
        self.solver['variables'][key]['data'] = self.solver['variables'][key]['data'][nodes]


     self.mesh['nodes'] = self.mesh['nodes'][nodes]
     self.mesh['elems'] = np.array(triangles)


 
 #def plot_structure(self,**argv):

 #   data = {0:{'name':'Structure','units':'','data':np.zeros(len(self.mesh['nodes'] if self.dim == 2 else self.indices))}}
   
 #   self.fig = plot_results(data,**argv)
    #plot_results(data,np.array(self.mesh['nodes']) if self.dim == 2 else self.surface_nodes,\
    #                                    np.array(self.mesh['elems']) if self.dim == 2 else self.surface_sides,**argv)


 def plot_maps(self,**argv):

   ind = self.mesh['meta'][-1]
   argv.update({'bulk':self.material['kappa'][int(ind),int(ind)],'fourier':self.solver['kappa_fourier']})
   if len(self.solver['variables'])> 2:
    argv.update({'bte':self.solver['kappa'][-1]})

   data = {'variables': self.unrolled_variables()  ,'nodes':self.mesh['nodes'],'elems':self.mesh['elems']}

   if argv.setdefault('write_data',False):

    bb = str(round(argv['bte'],2))+' W/m/K' if 'bte' in argv.keys() else '--'
    meta = 'Bulk: ' + str(round(argv['bulk'],2)) +' W/m/K <br>Fourier: ' +  str(round(argv['fourier'],2)) + ' W/m/K <br>BTE:' + bb
    data['meta'] = meta
    save_data(argv.setdefault('output_filename','sample'),data)


   if argv.setdefault('show',True):
    self.fig = plot_results(data,**argv)


 def unrolled_variables(self):

   #Enchance variables----
   variables = {}
   for key,value in self.solver['variables'].items():
     if value['data'].ndim == 1: #scalar
         variables[value['name']] = {'data':value['data'],'units':value['units']}
     elif value['data'].ndim == 2 : #vector 
         variables[value['name'] + '(x)'] = {'data':value['data'][:,0],'units':value['units']}
         variables[value['name'] + '(y)'] = {'data':value['data'][:,1],'units':value['units']}
         if self.dim == 3: 
          variables[value['name'] + '(z)'] = {'data':value['data'][:,2],'units':value['units']}

         mag = [np.linalg.norm(value) for value in value['data']]
         variables[value['name'] + '(mag.)'] = {'data':mag,'units':value['units']}
   return variables      
    








