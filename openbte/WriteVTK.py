from mpi4py import MPI
from pyvtk import *
import numpy as np
from matplotlib.tri import Triangulation

class WriteVtk(object):

 def __init__(self,**argv):

  self.argv = argv
  self.mesh = argv['Geometry']
  self.matrix_inclusion_map = argv.setdefault('matrix_inclusion_map',None)
  #-----------------------------
  self.solver = argv['Solver']
 
  self.Nx = argv.setdefault('repeat_x',1)
  self.Ny = argv.setdefault('repeat_y',1)
  self.Nz = argv.setdefault('repeat_z',1)
  self.data = []
  self.label = []


 def add_variable(self,variable,**argv):

  self.data.append(variable)
  self.label.append(argv['label'])



 def cell_to_node(self,**argv):

   data = argv['variable']
   delta = 2e-6*np.ones(3)

   if argv.setdefault('add_interfacial_nodes',False):
    data_int = argv['data_int']

    #-------------------------------
    data_int_inclusion = {}
    data_int_matrix = {}
    #matrix_inclusion_map = {}
    for node in self.mesh.node_list['Interface']:

     #self.mesh.nodes = np.append(self.mesh.nodes,[self.mesh.nodes[node]],axis=0)
     for ll in self.mesh.node_side_map[node]:
       if ll in self.mesh.side_list['Interface']: 
         (i,j) = self.mesh.side_elem_map[ll]
         if self.mesh.elem_region_map[i] == 'Matrix':
            im = 0 
            ii = 1
         else  :
            im = 1 
            ii = 0

         if node in data_int_matrix.keys():
          data_int_matrix[node] += data_int[ll][im]/2
         else: 
          data_int_matrix[node] = data_int[ll][im]/2

         node2 = self.matrix_inclusion_map[node]
         if node2 in data_int_inclusion.keys():
          data_int_inclusion[node2] += data_int[ll][ii]/2
         else: 
          data_int_inclusion[node2] = data_int[ll][ii]/2

    data_int_node = {'matrix':data_int_matrix,'inclusion':data_int_inclusion}

   #add new
   n_nodes = len(self.mesh.nodes)
    #-----------------------
   n_col = len(np.shape(data))
   if n_col == 3:
    node_data = np.zeros((n_nodes,3,3))
   else:
    if n_col == 2: n_col = 3 #vector
    node_data = np.zeros((n_nodes,n_col))

   conn = np.zeros(n_nodes)
   for n in self.mesh.node_elem_map.keys():
    for elem in self.mesh.node_elem_map[n]:
     node_data[n] += data[self.mesh.g2l[elem]]
     conn[n] +=1
     C = self.mesh.compute_elem_centroid(elem)

     if len(self.mesh.elems[0])==4 and n_col == 1:
      if C[self.mesh.direction] - self.mesh.nodes[n][self.mesh.direction] > self.mesh.size[self.mesh.direction]/2:
        node_data[n] +=1
     
      if C[self.mesh.direction] - self.mesh.nodes[n][self.mesh.direction] < -self.mesh.size[self.mesh.direction]/2:
        node_data[n] -=1

   for n in range(n_nodes):
    node_data[n] /= conn[n]
    

   if argv.setdefault('add_interfacial_nodes',False):

    for node in self.mesh.node_list['Interface']:
     node_data[node] = data_int_node['matrix'][node]
     node_data[self.matrix_inclusion_map[node]] =  data_int_node['inclusion'][self.matrix_inclusion_map[node]]

    #-----------------------------------------------------------------------------------

   return node_data

 def repeat_nodes_and_data(self,data_uc,increment):

   #periodic_nodes = set()
   #for side in self.mesh.side_list['Periodic']:
   # periodic_nodes.add(self.mesh.sides[side][0])
   # periodic_nodes.add(self.mesh.sides[side][1])
   #periodic_nodes = list(periodic_nodes)

   #periodic_nodes = np.array(self.mesh.periodic_nodes)
   periodic_1 = []
   periodic_2 = []
   for i in self.mesh.periodic_nodes.keys():
    periodic_1.append(i)
    periodic_1.append(self.mesh.periodic_nodes[i])
   #periodic_1 = [self.mesh.periodic_nodes[i][0] for i in self.mesh.periodic_nodes.keys()]
   #periodic_2 = [self.mesh.periodic_nodes[i][1] for i in self.mesh.periodic_nodes.keys()]

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

 def get_node_data(self,**argv):

    variable = argv['variable'] 
     
    node_data = self.cell_to_node(**argv)

    is_scalar = len(list(np.shape(variable))) == 1
    increment = [0,0,0]
    tmp = [[1,0,0],[0,1,0],[0,0,1]]
    if is_scalar:
     increment = tmp[self.mesh.direction]


    nodes,cells,data = self.repeat_nodes_and_data(node_data,increment)

    #return Triangulation(np.array(self.mesh.nodes)[:,0],np.array(self.nodes)[:,1], triangles=self.mesh.elems, mask=None),node_data,self.mesh.nodes


    return Triangulation(nodes[:,0],nodes[:,1], triangles=cells, mask=None),data,nodes


 def write_vtk(self):

  filename = self.argv.setdefault('filename_vtk','output.vtk')
  if MPI.COMM_WORLD.Get_rank() == 0:
   stored_data = {}

   output = []
   strc = 'PointData('
   for n,name in enumerate(self.data) :
    variable = self.solver[name]
    is_scalar = len(list(np.shape(variable))) == 1
    is_vector = len(list(np.shape(variable))) == 2
    is_tensor = len(list(np.shape(variable))) == 3

    if name+ '_int' in self.solver.keys():
     data_int = self.solver[name+ '_int']
     self.argv.update({'data_int':data_int})


    self.argv.update({'variable':self.solver[name]})
    node_data = self.cell_to_node(**self.argv)
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
    if is_vector:
      strc += r'''Vectors(output[''' + str(n) + r'''],name =' ''' + self.label[n] +  r''' ')'''
    if is_tensor:
      strc += r'''Tensors(output[''' + str(n) + r'''],name =' ''' + self.label[n] +  r''' ')'''

    if n == len(self.data)-1:
     strc += ')'
    else:
     strc += ','

  
   data = eval(strc)


   if self.mesh.dim == 3:
    vtk = VtkData(UnstructuredGrid(nodes,tetra=cells),data)
   else :
    if len(cells[0]) == 3:
     vtk = VtkData(UnstructuredGrid(nodes,triangle=cells),data)
    else:
     vtk = VtkData(PolyData(points = nodes,polygons = cells),data)

   vtk.tofile(filename,'ascii')
