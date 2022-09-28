import openbte.mesh as mesh
import openbte.mesher as mesher
import openbte.utils as utils
from mpi4py import MPI

comm = MPI.COMM_WORLD

def geometry(options_geometry)->'geometry':

 options_geometry.setdefault('model','lattice') 
 options_geometry.setdefault('lx',0)
 options_geometry.setdefault('ly',options_geometry['lx']) 

 #Just for back-compatibility
 direction = options_geometry.setdefault('direction','x') 
 if direction == 'x': options_geometry['applied_gradient'] = [1,0]
 if direction == 'y': options_geometry['applied_gradient'] = [0,1]
 #---------
 
 
 data = None
 if comm.rank == 0:     
   
  options_geometry['dmin'] = 0  
  options_geometry['overlap'] = 0  

  if not options_geometry['model'] == 'gmsh':
     mesher.Mesher(options_geometry)

  data = mesh.import_mesh(**options_geometry)

  mesh.compute_data(data,**options_geometry)
  
 return utils.create_shared_memory_dict(data)

def Geometry(**options_geometry): 

    data =  geometry(options_geometry) #quick hack

    if options_geometry.setdefault('save',True) and comm.rank == 0:
     utils.save(options_geometry.setdefault('output_filename','geometry'),data)   

    return data

