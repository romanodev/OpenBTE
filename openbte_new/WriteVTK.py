from mpi4py import MPI
from pyvtk import *
import numpy as np

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

   n_nodes = len(self.mesh.nodes)
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

 def repeat_nodes_and_data(self,Nx,Ny,Nz,data_uc,increment):

   periodic_nodes = np.array(self.mesh.periodic_nodes)
   periodic_1 = [i[0] for i in periodic_nodes]
   periodic_2 = [i[1] for i in periodic_nodes]

   #----------------------------------------------
   nodes_uc = self.mesh.nodes
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
      P = [round(self.mesh.size[0]*nx,3),round(self.mesh.size[1]*ny,3),round(self.mesh.size[2]*nz,3)]
      for n in range(n_nodes):
       #index += 1
       node_trial = [round(nodes_uc[n,0],4) + P[0],\
                     round(nodes_uc[n,1],4) + P[1],\
                     round(nodes_uc[n,2],4) + P[2]]

       if not node_trial in nodes:
        nodes.append(node_trial)
        data.append(data_uc[n]-dd)
        corr[index+n] = len(nodes)-1
       else:
        corr[index+n] = nodes.index(node_trial)

  
   cells_uc = self.mesh.elems
   cells = []
   for nx in range(Nx):
    for ny in range(Ny):
     for nz in range(Nz):
      P = [self.mesh.size[0]*nx,self.mesh.size[1]*ny,self.mesh.size[2]*nz]
      index = nx * Ny * Nz * n_nodes +  ny *  Nz * n_nodes + nz * n_nodes
      #add cells
      for gg in cells_uc:
       tmp = []
       for n in gg:
        tmp.append(corr[n+index])
       cells.append(tmp)


   return nodes,cells,data
 
 def write_vtk(self,filename='output.vtk'):

  if MPI.COMM_WORLD.Get_rank() == 0:
   stored_data = {}

   strc = 'data = PointData('
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
    ss = r'''nodes,cells,output_''' + str(n)+r'''= self.repeat_nodes_and_data(self.Nx,self.Ny,self.Nz,node_data,increment)'''
    exec(ss)
    #nodes,cells,output = self.repeat_nodes_and_data(self.Nx,self.Ny,self.Nz,node_data,increment)
    if is_scalar:
      strc += r'''Scalars(output_''' + str(n) + r''',name =' ''' + self.label[n] +  r''' ')'''
    else:
      strc += r'''Vectors(output_''' + str(n) + r''',name =' ''' + self.label[n] +  r''' ')'''

    if n == len(self.data)-1:
     strc += ')'
    else:   
     strc += ','

   exec(strc)
   if self.mesh.dim == 3:
    vtk = VtkData(UnstructuredGrid(nodes,tetra=cells),data)
   else :
    if len(cells[0]) == 3:
     vtk = VtkData(UnstructuredGrid(nodes,triangle=cells),data)
    else:
     vtk = VtkData(PolyData(points = nodes,polygons = cells),data)
   vtk.tofile(filename,'ascii')
