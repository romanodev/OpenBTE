import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pyvtk import *
import numpy as np
import os
from .utils import *
from .viewer import *
from .kappa_mode import *
from .suppression import *
from scipy.spatial import distance
import numpy.ma as ma
from mpi4py import MPI
comm = MPI.COMM_WORLD


def expand_variables(solver,dim):

  #Here we unroll variables for later use--
  variables = {}
  for key,value in solver['variables'].items():
  
     if value['data'].ndim == 1: #scalar
       variables[key] = {'data':value['data'],'units':value['units'],'increment':value['increment']}
       n_elems = len(value['data'])
     elif value['data'].ndim == 2 : #vector 
         variables[key + '(x)'] = {'data':value['data'][:,0],'units':value['units'],'increment':value['increment']}
         variables[key + '(y)'] = {'data':value['data'][:,1],'units':value['units'],'increment':value['increment']}
         if dim == 3: 
             variables[key + '(z)'] = {'data':value['data'][:,2],'units':value['units'],'increment':value['increment']}
         mag = np.array([np.linalg.norm(value) for value in value['data']])
         variables[key + '(mag.)'] = {'data':mag,'units':value['units'],'increment':value['increment']}

  variables['structure'] = {'data':np.zeros(n_elems),'units':'','increment':[0,0,0]}       
  solver['variables'] = variables



def add_preamble(geometry,store):

   store.write('# vtk DataFile Version 2.0\n')
   store.write('OpenBTE Data\n')
   store.write('ASCII\n')
   store.write('DATASET UNSTRUCTURED_GRID\n')
   store.write('POINTS ' + str(len(geometry['nodes'])) +   ' double\n')

   #write points--
   for n in range(len(geometry['nodes'])):
       for i in range(3): 
         store.write(str(geometry['nodes'][n,i])+' ')
       store.write('\n')  

   #write elems--
   dim = int(geometry['meta'][2])
   if dim == 2: 
      m = 3 
      ct = '5'
   elif dim == 3:
      m = 4 
      ct = '10'
   n = m+1

   n_elems = len(geometry['elems'])
   store.write('CELLS ' + str(n_elems) + ' ' + str(n*n_elems) + ' ' +  '\n')
   for k in range(len(geometry['elems'])):

       store.write(str(m) + ' ')  
       for i in range(n-1): 
         store.write(str(geometry['elems'][k][i])+' ')
       store.write('\n')  
           
   store.write('CELL_TYPES ' + str(n_elems) + '\n')
   for i in range(n_elems): 
     store.write(ct + ' ')
   store.write('\n')  



def write_vtu(solver,geometry,dof):

   #Write
   store = open('output.vtk', 'w+')

   add_preamble(geometry,store)
   
   n_elems = len(geometry['elems'])
   n_nodes = len(geometry['nodes'])
   dim = int(geometry['meta'][2])

   #write data
   if dof == 'cell':
    store.write('CELL_DATA ' + str(n_elems) + '\n')
   else: 
    store.write('POINT_DATA ' + str(n_nodes) + '\n')

   for n,(key, value) in enumerate(solver['variables'].items()):

     name = key + '[' + value['units'] + ']'
     if value['data'].ndim == 1: #scalar
         strc = np.array2string(value['data'],max_line_width=1e6)
         store.write('SCALARS ' + name + ' double\n')
         store.write('LOOKUP_TABLE default\n')
         for i in value['data']:
          store.write(str(i)+' ')
         store.write('\n') 

     elif value['data'].ndim == 2: #tensor
         store.write('VECTORS ' + name + ' double\n')
         for i in value['data']:
           strc = np.array2string(i,max_line_width=1e6)
           if dim == 2:
            store.write(strc[1:-1]+' 0.0 \n')
           else: 
            store.write(strc[1:-1]+'\n')

     elif value['data'].ndim == 3: #tensor
         store.write('TENSORS ' + name + ' double\n')
         for i in value['data']:
          for j in i:
           strc = np.array2string(j,max_line_width=1e6)
           store.write(strc[1:-1]+'\n')
          store.write('\n') 

   store.close()



def get_surface_nodes(solver,geometry):

     sides = list(geometry['boundary_sides']) + \
                     list(geometry['periodic_sides']) + \
                     list(geometry['inactive_sides'])

     nodes = geometry['sides'][sides].flat


     triangles = np.arange(len(nodes)).reshape((len(sides),3))
     
     for key in solver['variables'].keys():
        solver['variables'][key]['data'] = solver['variables'][key]['data'][nodes]


     geometry['nodes'] = geometry['nodes'][nodes]
     geometry['elems'] = np.array(triangles)




def duplicate_cells(geometry,solver,repeat):

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
   def repeat_cell(axis,n,d,nodes,replace):

    tmp = nodes.copy()
    for x in np.arange(n-1):
     tmp[:,axis] +=  size[axis]

     nodes = np.vstack((nodes,tmp))
    
    return nodes,replace

   replace = {}  
   for i in range(int(geometry['meta'][2])):
     nodes,replace = repeat_cell(i,repeat[i],size[i],nodes,replace)
   #---------------------------------------------------

   #Repeat elements----------
   unit_cell = np.array(geometry['elems']).copy()
  
   elems = geometry['elems'] 
   for i in range(np.prod(repeat)-1):
     elems = np.vstack((elems,unit_cell+(i+1)*(n_nodes)))

   #------------------------

   #duplicate variables---
   for n,(key, value) in enumerate(solver['variables'].items()):  
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

     solver['variables'][key]['data'] = value['data']

   geometry['elems'] = elems
   geometry['nodes'] = nodes

   #return the new side
   size = [geometry['size'][i] *repeat[i]   for i in range(dim)]

   return size




def get_node_data(solver,geometry):

  dim = int(geometry['meta'][2])
  for key in solver['variables'].keys():
   data = solver['variables'][key]['data']
   #NEW-------------
   conn = np.zeros(len(geometry['nodes']))
   if data.ndim == 2:
       node_data = np.zeros((len(geometry['nodes']),int(dim)))
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
   solver['variables'][key]['data'] = node_data


def Plot(**argv):


  if comm.rank == 0:

   #import data-------------------------------
   geometry     = argv['geometry'] if 'geometry' in argv.keys() else load_data('geometry')
   solver       = argv['solver']   if 'geometry' in argv.keys() else load_data('solver')
   material     = argv['material'] if 'material' in argv.keys() else load_data('material')
   dim = int(geometry['meta'][2])
   model = argv['model']

   if model == 'maps':

     expand_variables(solver,dim) #this is needed to get the component-wise data for plotly

     get_node_data(solver,geometry)
     repeat = argv.setdefault('repeat',[1,1,1])

     if dim == 3:
      get_surface_nodes(solver,geometry)

     size = duplicate_cells(geometry,solver,repeat)

     if argv.setdefault('show',True):
        plot_results(solver,geometry,**argv)

     output ={'nodes'    :geometry['nodes'],\
             'elems'    :geometry['elems'],\
             'variables':solver['variables'],'bulk':solver['kappa_bulk'],'fourier':solver['kappa_fourier'],'size':size,'direction':['x','y','z'][np.argmax(geometry['applied_gradient'])]}

     if 'kappa_bte' in solver.keys():
         output.update({'bte':solver['kappa_bte'][-1]}) 

     if argv.setdefault('save_sample',False):
       save_data('sample',output)

     return output

   elif model == 'line':

     get_node_data(solver,geometry)

     duplicate_cells(geometry,solver,argv.setdefault('repeat',[1,1,1]))

     return compute_line_data(geometry,solver,**argv)
     

   elif model == 'vtu':

       if argv.setdefault('dof','nodes'):  
        get_node_data(solver,geometry)

       duplicate_cells(geometry,solver,argv.setdefault('repeat',[1,1,1]))

       write_vtu(solver,geometry,argv['dof'])  
       #write_vtu(solver,geometry)  

   #elif model == 'vtu_cell':

   #  duplicate_cells(geometry,solver,argv.setdefault('repeat',[1,1,1]))
     
   #  write_vtu_cell(solver,geometry)  


   elif model == 'suppression':

      return write_suppression(**argv)

   elif model == 'kappa_mode':

      write_mode_kappa(**argv)






