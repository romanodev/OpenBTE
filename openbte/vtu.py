import openbte.utils as utils
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

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
       if len(geometry['elems'][0]) == 3: 
        m = 3 
        ct = '5'
       else: 
        m = 4 
        ct = '9'

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

   
def vtu(material,solver,geometry,options_vtu):

 if comm.rank == 0:

   #Parse options--
   dof      = options_vtu.setdefault('dof','nodes')
   repeat   = options_vtu.setdefault('repeat',[1,1,1])
   filename = options_vtu.setdefault('filename','output')
   mask     = options_vtu.setdefault('mask',[])
   displ    = options_vtu.setdefault('displ',[0,0,0])
   #---------------

   variables = utils.extract_variables(solver)
   #mask
   if len(mask) > 0:
    geometry['elems'] = np.array(geometry['elems'])[mask]
    for n,(key, value) in enumerate(variables.items()):
       value['data'] = value['data'][mask] 
   #---      

   if dof=='nodes':  
      utils.get_node_data(variables,geometry)

   if np.prod(repeat) > 1:
      utils.duplicate_cells(geometry,variables,repeat,displ)

   store = open(filename + '.vtk', 'w+')

   add_preamble(geometry,store)
   
   n_elems = len(geometry['elems'])
   n_nodes = len(geometry['nodes'])
   dim = int(geometry['meta'][2])

   #write data
   if dof == 'cell':
    store.write('CELL_DATA ' + str(n_elems) + '\n')
   else: 
    store.write('POINT_DATA ' + str(n_nodes) + '\n')

   for n,(key, value) in enumerate(variables.items()):

     name = key + '[' + value['units'] + ']'
     if value['data'].ndim == 1: #scalar
         strc = np.array2string(value['data'],max_line_width=1e6)
         store.write('SCALARS ' + name + ' double\n')
         store.write('LOOKUP_TABLE default\n')
         for i in value['data']:
          store.write(str(i)+' ')
         store.write('\n') 

     elif value['data'].ndim == 2: #vector
         store.write('VECTORS ' + name + ' double\n')
         for i in value['data']:
           strc = np.array2string(i,max_line_width=1e6)
           if dim == 2:
            store.write(strc[1:-1] +'\n')
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
