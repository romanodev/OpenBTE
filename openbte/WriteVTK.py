from mpi4py import MPI
from pyvtk import *
import numpy as np
from matplotlib.tri import Triangulation

class WriteVtk(object):

 def __init__(self,argv):

  self.mesh = argv['Geometry']
  self.Nx = argv.setdefault('repeat_x',1)
  self.Ny = argv.setdefault('repeat_y',1)
  self.Nz = argv.setdefault('repeat_z',1)
  self.data = []
  self.label = []


 def add_variable(self,variable,**argv):

  self.data.append(variable)
  self.label.append(argv['label'])

 def cell_to_node(self,data):


   #n_nodes = len(self.mesh.nodes)

   #print(len(self.mesh.node_list['Interface']))
   delta = 2e-6*np.ones(3)
   #delta[2] = 0.0

   add_nodes = True
   if add_nodes:

    for node in self.mesh.node_list['Interface']:

     self.mesh.nodes = np.append(self.mesh.nodes,[self.mesh.nodes[node]],axis=0)
     node_elem_map_1 = []
     node_elem_map_2 = []
     for elem in self.mesh.node_elem_map[node]:
      if self.mesh.get_region_from_elem(elem) == 'Matrix':
        node_elem_map_1.append(elem)
      else:
        node_elem_map_2.append(elem)
        index = self.mesh.elems[elem].index(node)
        self.mesh.elems[elem][index]= len(self.mesh.nodes)-1


     self.mesh.node_elem_map[node] = node_elem_map_1
     self.mesh.node_elem_map.update({len(self.mesh.nodes)-1:node_elem_map_2})

   #for node  in self.mesh.node_list['Interface']:
#     print(len(self.mesh.node_elem_map[node]))

   #quit()
   n_nodes = len(self.mesh.nodes)
    #-----------------------
    #quit()
   n_col = len(np.shape(data))
   if n_col == 2: n_col = 3
   node_data = np.zeros((n_nodes,n_col))

   conn = np.zeros(n_nodes)
   for n in self.mesh.node_elem_map.keys():
    for elem in self.mesh.node_elem_map[n]:
     node_data[n] += data[elem]
     conn[n] +=1

   for n in range(n_nodes):
    node_data[n] /= conn[n]

   return node_data

 def repeat_nodes_and_data(self,data_uc,increment):

   periodic_nodes = np.array(self.mesh.periodic_nodes)
   periodic_1 = [i[0] for i in periodic_nodes]
   periodic_2 = [i[1] for i in periodic_nodes]

   #----------------------------------------------
   nodes_uc = self.mesh.nodes

   n_nodes = len(nodes_uc)

   nodes = []
   data = []
   #add nodes---
   corr = np.zeros(n_nodes*self.Nx*self.Ny*self.Nz,dtype=np.int)
   index = 0
   for nx in range(self.Nx):
    for ny in range(self.Ny):
     for nz in range(self.Nz):
      dd = nx*increment[0] + ny*increment[1] + nz*increment[2]
      index = nx * self.Ny * self.Nz * n_nodes +  ny *  self.Nz * n_nodes + nz * n_nodes
      P = [round(self.mesh.size[0]*nx,3),round(self.mesh.size[1]*ny,3),round(self.mesh.size[2]*nz,3)]
      for n in range(n_nodes):
       #index += 1
       node_trial = [round(nodes_uc[n,0],4) + P[0]-(self.Nx-1)*0.5*self.mesh.size[0],\
                     round(nodes_uc[n,1],4) + P[1]-(self.Nx-1)*0.5*self.mesh.size[1],\
                     round(nodes_uc[n,2],4) + P[2]-(self.Nx-1)*0.5*self.mesh.size[2]]

       nodes.append(node_trial)
       data.append(data_uc[n]-dd)
       corr[index+n] = len(nodes)-1


   cells_uc = self.mesh.elems

   cells = []
   for nx in range(self.Nx):
    for ny in range(self.Ny):
     for nz in range(self.Nz):
      P = [self.mesh.size[0]*nx,self.mesh.size[1]*ny,self.mesh.size[2]*nz]
      index = nx * self.Ny * self.Nz * n_nodes +  ny *  self.Nz * n_nodes + nz * n_nodes
      #add cells
      for gg in cells_uc:
       tmp = []
       for n in gg:
        tmp.append(corr[n+index])
       cells.append(tmp)



   return np.array(nodes),np.array(cells),np.array(data)

 def get_node_data(self,variable):

 
    node_data = self.cell_to_node(variable)
    is_scalar = len(list(np.shape(variable))) == 1
    increment = [0,0,0]
    tmp = [[1,0,0],[0,1,0],[0,0,1]]
    if is_scalar:
     increment = tmp[self.mesh.direction]
    nodes,cells,data = self.repeat_nodes_and_data(node_data,increment)

    #return Triangulation(np.array(self.mesh.nodes)[:,0],np.array(self.nodes)[:,1], triangles=self.mesh.elems, mask=None),node_data,self.mesh.nodes



    return Triangulation(nodes[:,0],nodes[:,1], triangles=cells, mask=None),data,nodes



 def write_vtk(self,filename='output.vtk'):

  if MPI.COMM_WORLD.Get_rank() == 0:
   stored_data = {}


   output = []
   strc = 'PointData('
   for n,variable in enumerate(self.data) :

    is_scalar = len(list(np.shape(variable))) == 1

    node_data = self.cell_to_node(variable)
    #stored_data.update({self.label[n]:{variable:node_data[:,0],'label':self.label[n]}})

    #Get increment for plotting-------------
    increment = [0,0,0]
    tmp = [[1,0,0],[0,1,0],[0,0,1]]
    if is_scalar:
     increment = tmp[self.mesh.direction]
    #---------------------------------------
    ss = '''self.repeat_nodes_and_data(node_data,increment)'''
    nodes,cells,tmp = eval(ss)
    output.append(tmp)
    if is_scalar:
      strc += r'''Scalars(output[''' + str(n) + r'''],name =' ''' + self.label[n] +  r''' ')'''
    else:
      strc += r'''Vectors(output[''' + str(n) + r'''],name =' ''' + self.label[n] +  r''' ')'''

    if n == len(self.data)-1:
     strc += ')'
    else:
     strc += ','
 
   data=eval(strc)

   if self.mesh.dim == 3:
    vtk = VtkData(UnstructuredGrid(nodes,tetra=cells),data)
   else :
    if len(cells[0]) == 3:
     vtk = VtkData(UnstructuredGrid(nodes,triangle=cells),data)
    else:
     vtk = VtkData(PolyData(points = nodes,polygons = cells),data)
   vtk.tofile(filename,'ascii')
