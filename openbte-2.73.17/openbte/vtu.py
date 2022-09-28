
def add_preamble(geometry,store):

   store +='# vtk DataFile Version 2.0\n'
   store +='OpenBTE Data\n'
   store +='ASCII\n'
   store +='DATASET UNSTRUCTURED_GRID\n'
   store +='POINTS ' + str(len(geometry['nodes'])) +   ' double\n'
   #write points--
   for n in range(len(geometry['nodes'])):
       for i in range(3): 
         store +=str(geometry['nodes'][n,i])+' '
       store +='\n'

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
   store +='CELLS ' + str(n_elems) + ' ' + str(n*n_elems) + ' ' +  '\n'
   for k in range(len(geometry['elems'])):

       store +=str(m) + ' '
       for i in range(n-1): 
         store +=str(geometry['elems'][k][i])+' '
       store +='\n'
           
   store +='CELL_TYPES ' + str(n_elems) + '\n'
   for i in range(n_elems): 
     store +=ct + ' '
   store +='\n'

   return store

   
def vtu(material,solver,geometry,options_vtu):

 import openbte.utils as utils
 import numpy as np
 from mpi4py import MPI
 comm = MPI.COMM_WORLD

 store = None
 if comm.rank == 0:

   #Parse options--
   dof      = options_vtu.setdefault('dof','nodes')
   repeat   = options_vtu.setdefault('repeat',[1,1,1])
   filename = options_vtu.setdefault('filename','output')
   mask     = options_vtu.setdefault('mask',[])
   displ    = options_vtu.setdefault('displ',[0,0,0])
   #---------------

   variables = utils.extract_variables(solver,geometry)
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

   store = ''

   store = add_preamble(geometry,store)
  
   n_elems = len(geometry['elems'])
   n_nodes = len(geometry['nodes'])
   dim = int(geometry['meta'][2])

   #write data
   if dof == 'cell':
    store +='CELL_DATA ' + str(n_elems) + '\n'
   else: 
    store +='POINT_DATA ' + str(n_nodes) + '\n'

   for n,(key, value) in enumerate(variables.items()):

     name = key + '[' + value['units'] + ']'
     if value['data'].ndim == 1: #scalar
         strc = np.array2string(value['data'],max_line_width=1e6)
         store +='SCALARS ' + name + ' double\n'
         store +='LOOKUP_TABLE default\n'
         for i in value['data']:
          store +=str(i)+' '
         store +='\n'

     elif value['data'].ndim == 2: #vector
         store +='VECTORS ' + name + ' double\n'
         for i in value['data']:
           strc = np.array2string(i,max_line_width=1e6)
           if dim == 2:
            store +=strc[1:-1] +'\n'
           else: 
            store +=strc[1:-1]+'\n'

     elif value['data'].ndim == 3: #tensor
         store +='TENSORS ' + name + ' double\n'
         for i in value['data']:
          for j in i:
           strc = np.array2string(j,max_line_width=1e6)
           store +=strc[1:-1]+'\n'
          store +='\n'

   if options_vtu.setdefault('write',True):   
    with open('output.vtk','w') as f:
       f.write(store)

 store = comm.bcast(store)

 return store    


