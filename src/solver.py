from mpi4py import MPI
from fourier import Fourier
from bte import BTE
from pyvtk import *
import numpy as np
import deepdish as dd

class Solver(object):

 def __init__(self,**argv):
  
  self.mesh = argv['geometry']
  self.state = {}
  #Print kappa Bulk
  if MPI.COMM_WORLD.Get_rank() == 0:

   #Initialization-------------
   print(' ')
   self.print_logo()
   print(' ')
   print('Bulk Thermal Conductivity:  ' + str(round(argv['material'].state['kappa_bulk_tot'],4)) + ' W/m/K')

  #We solve Fourier in any case
  fourier = Fourier(argv)
  self.state.update({'fourier_temperature':fourier.temperature})
  self.state.update({'fourier_flux':fourier.flux})
  self.state.update({'fourier_kappa':fourier.kappa})
  argv.update(self.state)
  #Solve BTE
  if argv['model'] == 'bte':
   bte = BTE(argv)
   self.state.update(bte.state)
  #if MPI.COMM_WORLD.Get_rank() == 0:
   #print('Ratio:  ' + str(round(self.state['fourier_kappa']/self.state['bte_kappa'],4)))
   #print(' ')



  self.write_vtk(argv)
  if MPI.COMM_WORLD.Get_rank() == 0:
   dd.io.save('solver.hdf5', self.state)
  
#kb,temp = bte.compute_function(x0,fourier_temp = temp)
   
 def get_increment(self,argv):

   gradient = argv.setdefault('gradient','x')

   increment = [0,0,0] 
   if gradient == 'x': increment = [1,0,0]
   if gradient == 'y': increment = [0,1,0]
   if gradient == 'z': increment = [0,0,1]
   if argv['variable'] == 'bte_flux':
    increment = [0,0,0]
   if argv['variable'] == 'fourier_flux':
    increment = [0,0,0]

   return increment


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
 
 def write_vtk(self,argv):

  if MPI.COMM_WORLD.Get_rank() == 0:
   #Write Fourier Temperature
   Nx = argv.setdefault('repeat_x',3)
   Ny = argv.setdefault('repeat_y',3)
   Nz = argv.setdefault('repeat_z',1)

   #Fourier Temperature----
   node_data = self.cell_to_node(self.state['fourier_temperature'])
   argv.update({'variable':'fourier_temperature'})
   increment = self.get_increment(argv)
   nodes,cells,fourier_temp = self.repeat_nodes_and_data(Nx,Ny,Nz,node_data,increment)


   #Fourier Flux----
   node_data = self.cell_to_node(np.array(self.state['fourier_flux']))
   argv.update({'variable':'fourier_flux'})
   increment = self.get_increment(argv)
   nodes,cells,fourier_flux = self.repeat_nodes_and_data(Nx,Ny,Nz,node_data,increment)


   #BTE Temperature----
   node_data = self.cell_to_node(self.state['bte_temperature'])
   argv.update({'variable':'bte_temperature'})
   increment = self.get_increment(argv)
   nodes,cells,bte_temp = self.repeat_nodes_and_data(Nx,Ny,Nz,node_data,increment)

   #BTE Flux----
   node_data = self.cell_to_node(np.array(self.state['bte_flux']))
   argv.update({'variable':'bte_flux'})
   increment = self.get_increment(argv)
   nodes,cells,bte_flux = self.repeat_nodes_and_data(Nx,Ny,Nz,node_data,increment)


   #Save data------
   data =  PointData(Scalars(fourier_temp,name = 'Temperature (Fourier)'),\
                       Scalars(bte_temp,name = 'Temperature (BTE)'),\
                       Vectors(fourier_flux,name='Flux (Fourier)'),\
                       Vectors(bte_flux,name='Flux (BTE)'))
   vtk = VtkData(UnstructuredGrid(nodes,triangle=cells),data)
   vtk.tofile('output.vtk','ascii')


   #--------------




 def print_logo(self):

   print(r'''  ___                   ____ _____ _____ ''' )
   print(r''' / _ \ _ __   ___ _ __ | __ )_   _| ____|''')
   print(r'''| | | | '_ \ / _ \ '_ \|  _ \ | | |  _|  ''')
   print(r'''| |_| | |_) |  __/ | | | |_) || | | |___ ''')
   print(r''' \___/| .__/ \___|_| |_|____/ |_| |_____|''')
   print(r'''      |_|                                ''')
   print('')
   print('Giuseppe Romano [romanog@mit.edu]')

