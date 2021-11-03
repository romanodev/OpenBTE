def geometry(options_geometry)->'geometry':

 import openbte.mesh as mesh
 import openbte.mesher as mesher
 import openbte.utils as utils
 from mpi4py import MPI
 comm = MPI.COMM_WORLD

 options_geometry.setdefault('model','lattice')  
 options_geometry.setdefault('ly',options_geometry['lx']) 
 
 data = None
 if comm.rank == 0:     
   
  options_geometry['dmin'] = 0  

  mesher.Mesher(options_geometry)

  data = mesh.import_mesh(**options_geometry)

  mesh.compute_data(data,**options_geometry)


 return utils.create_shared_memory_dict(data)
  

def Geometry(**options_geometry): 

    data =  geometry(options_geometry) #quick hack

    if options_geometry.setdefault('save',True) and comm.rank == 0:
     save_data(options_geometry.setdefault('output_filename','geometry'),data)    

    return data

