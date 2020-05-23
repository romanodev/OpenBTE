import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pyvtk import *
import numpy as np
import deepdish as dd
import os
from .utils import *
import deepdish as dd
from .viewer import *

from mpi4py import MPI
comm = MPI.COMM_WORLD

class Plot(object):

 def __init__(self,**argv):

  if comm.rank == 0:

   #import data-------------------------------
   if 'geometry' in argv.keys():
    self.mesh = argv['geometry'].data
   else: 
    if os.path.isfile('geometry.h5') :   
     self.mesh = dd.io.load('geometry.h5')
   
   if 'solver' in argv.keys():
    self.solver = argv['solver'].state
   else: 
       if os.path.isfile('solver.h5') :
        self.solver = dd.io.load('solver.h5')

   #if 'material' in argv.keys():
   # self.solver = argv['material']
   #else: 
   #    if os.path.isfile('material.h5') :
   #     self.material = load_dictionary('material.h5')
   #--------------------------------------------

   model = argv['model'].split('/')[0]

   if model == 'structure':
    self.plot_structure(**argv)

   elif model == 'maps':
    self.plot_maps(**argv)

   elif model == 'vtk':
    self.write_cell_vtk(**argv)   

 def write_cell_vtk(self,**argv):


   output = []
   strc = 'CellData('
   for n,(key, value) in enumerate(self.solver['variables'].items()):

     if  value['data'].shape[0] == 1:  value['data'] =  value['data'][0] #band-aid solution
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
    vtk = VtkData(UnstructuredGrid(self.mesh['nodes'],triangle=self.mesh['elems']),data)

   vtk.tofile('output.vtk','ascii')



 def write_vtk(self,**argv):

   for key in self.solver['variables'].keys(): 
       self.solver['variables'][key]['data'] = self._get_node_data(self.solver['variables'][key]['data'])

   output = []
   strc = 'PointData('
   for n,(key, value) in enumerate(self.solver['variables'].items()):
      
     if  value['data'].shape[0] == 1:  value['data'] =  value['data'][0] #band-aid solution
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
    vtk = VtkData(UnstructuredGrid(self.mesh['nodes'],triangle=self.mesh['elems']),data)

   vtk.tofile('output.vtk','ascii')


 def _get_node_data(self,data,indices=None):


   if data.ndim > 1:
       node_data = np.zeros((len(self.mesh['nodes']),3))
   else:   
       node_data = np.zeros(len(self.mesh['nodes']))

   for k,e in enumerate(self.mesh['elems']):
     for n in e:
      node_data[n] += data[k]/self.mesh['conn'][n]

   node_data = node_data[indices]

   return node_data


 def get_surface_nodes(self):


     triangles = []
     nodes = []
     for l in list(self.mesh['boundary_sides']) + \
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

   if self.mesh['dim'] == 2:   
    data = {0:{'name':'Structure','units':'','data':np.zeros(len(self.mesh['nodes']))}}
    plot_results(data,self.mesh['nodes'],np.array(self.mesh['elems']))
   else: 
    if self.mesh['dim'] == 3:
     triangles,indices = self.get_surface_nodes()   
     data = {0:{'name':'Structure','units':'','data':np.zeros(len(indices))}}
     plot_results(data,self.mesh['nodes'][indices],triangles)
     

 def plot_maps(self,**argv):


   if self.mesh['dim'] == 3:
     elems,indices = self.get_surface_nodes() 
   else:  
     elems = np.array(self.mesh['elems'])
     indices = np.arange(len(self.mesh['nodes']))

   for key in self.solver['variables'].keys(): 
       self.solver['variables'][key]['data'] = self._get_node_data(self.solver['variables'][key]['data'],indices=indices)

   self.solver['variables'][-1] = {'name':'Structure','units':'','data':np.zeros(len(indices))}

   plot_results(self.solver['variables'],self.mesh['nodes'][indices],elems,**argv)


