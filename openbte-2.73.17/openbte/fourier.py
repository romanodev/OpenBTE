from __future__ import absolute_import
import numpy as np
from scipy.sparse.linalg import splu
from termcolor import colored, cprint 
from .utils import *
from mpi4py import MPI
import scipy.sparse as sp
import time
from scipy.sparse.linalg import lgmres
import sys
import scipy
#from cachetools import cached,LRUCache
#from cachetools.keys import hashkey
import openbte.utils as utils

comm = MPI.COMM_WORLD

#cache_assemble = LRUCache(maxsize=1000)
#cache_compute_grad_data = LRUCache(maxsize=10000)
#cache_get_SU = LRUCache(maxsize=10000)
#cache_get_decomposed_directions = LRUCache(maxsize=10000)


def clear_fourier_cache():

    #if comm.rank == 0:
    cache_assemble.clear()
    cache_compute_grad_data.clear()
    cache_get_SU.clear()
    cache_get_decomposed_directions.clear()

def fourier_info(data):
          print('                        FOURIER                 ',flush=True)   
          print(colored(' -----------------------------------------------------------','green'),flush=True)
          print(colored('  Iterations:                              ','green') + str(int(data[2])),flush=True)
          print(colored('  Relative error:                          ','green') + '%.1E' % (data[1]),flush=True)
          print(colored('  Fourier Thermal Conductivity [W/m/K]:    ','green') + str(round(data[0],3)),flush=True)
          print(colored(' -----------------------------------------------------------','green'),flush=True)
          print(" ")



#@cached(cache=cache_get_SU, key=lambda bundle, k: hashkey(k))
#def get_SU(bundle,k):

#     A = k*bundle[0] + sp.diags(bundle[1],format='csc')
    
#     return splu(A)

#def get_key(ll,kappa):

#   return (ll,tuple(map(tuple,kappa)))

#def unpack(data):

#    return data[0],np.array(data[1])


#def get_kappa_map_from_mat(material,geometry):

#        mat_map =geometry['elem_mat_map']
#        dim = int(geometry['meta'][2])

#        kappa = material['kappa']

#        kappa = np.array(kappa)
#        if kappa.ndim == 3:
#            return np.array([ list(kappa[i][:dim,:dim])  for i in mat_map])
#        else:
#            return np.array([list(kappa)]*len(geometry['elems']))


def solve_fourier(material,geometry,kappa,**argv):
  """ Solve Fourier with a single set of kappa """

  data = None  
  if comm.rank == 0:

   geometry['kappa'] = kappa

   SU,B,scale = assemble(material,geometry,**argv)


   #options = {'scale':scale}
   argv['scale'] = scale
   meta,temp,grad =  solve_convergence(geometry,SU,B,**argv)

   flux = -np.einsum('cij,cj->ci',geometry['kappa'],grad)*1e9
 
   dim     = int(geometry['meta'][2])
   n_elems = len(geometry['elems'])
   if dim == 2:flux = np.append(flux, np.zeros((n_elems,1)), axis=1)

   if argv.setdefault('verbose',True):
     fourier_info(meta)
   
   data =  {'Temperature_Fourier':temp,'kappa_fourier':np.array([meta[0]]),'Flux_Fourier':flux,'Heat_Generation':geometry['generation']}

  return utils.create_shared_memory_dict(data)



def get_kappa(mesh,i,j,ll):

   kappa = mesh['kappa']

   if i == j:
    return np.array(kappa[i])
   
   dim = int(mesh['meta'][2])
   normal = mesh['face_normals'][ll,0:dim]

   kappa_i = np.array(kappa[i])
   kappa_j = np.array(kappa[j])

   ki = np.dot(normal,np.dot(kappa_i,normal))
   kj = np.dot(normal,np.dot(kappa_j,normal))
   w  = mesh['interp_weigths'][ll]

   #if ll in mesh['interface_sides']:
     #dist   = mesh['dists'][ll,0:dim]*1e-9
     #h = 0.882e9
     #B = ki*kj/h/np.linalg.norm(dist)
   #  B = 0
   #else:  
   B = 0

   kappa_loc = kj*kappa_i/(ki*(1-w) + kj*w + B)

   return kappa_loc


def get_kappa_tensor(mesh,ll,kappa):

    if np.isscalar(kappa):
       rot = kappa*np.eye(3) 
    elif  kappa.ndim == 3:
       i,j = mesh['side_elem_map_vec'][ll]
       rot = get_kappa(mesh,i,j,ll,kappa)
    else:   
       rot = kappa  

    return rot   


#cached(cache=cache_get_decomposed_directions, key=lambda mesh,kappa_ll:hashkey(kappa_ll))
def get_decomposed_directions(mesh,ll,kappa):
     
    #ll,kappa = unpack(kappa_ll)

    kappa_map = get_kappa_tensor(mesh,ll,kappa)

    dim = int(mesh['meta'][2])
    normal = mesh['face_normals'][ll,0:dim]
    dist   = mesh['dists'][ll,0:dim]
    rot = kappa_map[:dim,:dim]

    if ll in (list(mesh['fixed_temperature_sides'])):
       v_orth = np.dot(normal,np.dot(rot,normal))/np.dot(normal,dist)
       #v_non_orth = np.dot(rot,normal) - dist*v_orth
       v_non_orth = np.zeros(dim)
    else:   
     v_orth = np.dot(normal,np.dot(rot,normal))/np.dot(normal,dist)
     v_non_orth = np.dot(rot,normal) - dist*v_orth
    
    return v_orth,v_non_orth[:dim]

def compute_laplacian(temp,**argv):

  grad_T = compute_grad(temp,**argv).T

  laplacian = np.zeros_like(temp)
  for i,grad in enumerate(grad_T):
     laplacian +=   compute_grad(grad,**argv).T[i]

  return laplacian




def compute_secondary_flux(geometry,temp,**argv):

   kappa = geometry['kappa'] 

   gradT = compute_grad_common(temp,geometry)

   dim     = int(geometry['meta'][2])
   n_elems = len(geometry['elems'])

   C = np.zeros(n_elems)
   for ll in geometry['active_sides']:
      area = geometry['areas'][ll] 
      (i,j) = geometry['side_elem_map_vec'][ll]
      kappa_loc = get_kappa(geometry,i,j,ll)
      (v_orth,v_non_orth) = get_decomposed_directions(geometry,ll,kappa_loc)
      if not i==j:
       w =  geometry['interp_weigths'][ll]
       F_ave = (w*gradT[i] + (1.0-w)*gradT[j])
       tmp = np.dot(F_ave,v_non_orth)*area
       C[i] += tmp
       C[j] -= tmp
      #else:
      #  if ll in list(mesh['fixed_temperature_sides']):
      #   F_ave = gradT[i] 
      #   tmp = np.dot(F_ave,v_non_orth)*area
      #   C[i] += tmp

   return C/geometry['volumes'],gradT



def compute_effective_thermal_conductivity(geometry,temp,**argv):


   kappa = geometry['kappa'] 
   kappa_eff = 0
   area_tot = 0
   g = 0
   for ll in geometry['flux_sides']:
    (i,j) = geometry['side_elem_map_vec'][ll]
    area = geometry['areas'][ll]

    kappa_loc = get_kappa(geometry,i,j,ll)
    (v_orth,v_non_orth) = get_decomposed_directions(geometry,ll,kappa_loc)

    T1 = temp[i] * geometry['cp'][i]
    T2 = temp[j] * geometry['cm'][j] + geometry['perturbation'][j]

    DeltaT = T1 - T2

    kappa_eff += v_orth *  DeltaT 

   return kappa_eff*geometry['kappa_factor'][0]






#def solve_convergence(argv,kappa,B,SU,log=False,scale=[]):
def solve_convergence(geometry,SU,B,**argv):

    scale = argv.setdefault('scale',1)
    argv.setdefault('max_fourier_error',1e-13) 
    argv.setdefault('max_fourier_iter',1) 

    C  = np.zeros_like(B)

    #kappa_map = get_kappa_map(mesh,kappa)

    n_elems = len(geometry['elems'])
    dim     = int(geometry['meta'][2])
    grad    = np.zeros((n_elems,dim))

    n_iter = 0;kappa_old = 0;error = 1;C_old = np.zeros(n_elems)
    while error > argv['max_fourier_error'] and \
                  n_iter < argv['max_fourier_iter'] :

        RHS = B + C #+ generation
        if len(scale) > 0: RHS *=scale

        temp = SU.solve(RHS)

        argv['grad'] = grad
        kappa_eff = compute_effective_thermal_conductivity(geometry,temp,**argv)
        kappa_old = kappa_eff
        n_iter +=1
        #TODO: check error on residual
        C,grad = compute_secondary_flux(geometry,temp,**argv)

        error = (np.linalg.norm(C-C_old))/np.linalg.norm(C)
        C_old = C.copy()

    #temp = temp - (max(temp)+min(temp))/2.0
    return [kappa_eff,error,n_iter],temp,grad 


def get_kappa_map(mesh,kappa):


    n_elems = len(mesh['elems'])
    dim = int(mesh['meta'][2])

    if np.isscalar(kappa):
        if kappa == -1: 
           kappa = mesh['elem_kappa_map']
        else: 
           print('error')

    elif isinstance(kappa,tuple):
       kappa = np.array(kappa) 
       if kappa.ndim == 2: 
          kappa = np.tile(kappa,(n_elems,1,1))
    elif isinstance(kappa,np.ndarray):
        if kappa.ndim == 2: 
          kappa = np.tile(kappa,(n_elems,1,1))

    kappa = kappa[:,:dim,:dim]
    return kappa


#def assemble(mesh,material,kappa):
def assemble(material,geometry,**argv):


    kappa = geometry['kappa']
    if 'boundary_conductance' in material.keys():
        h =  material['boundary_conductance'][0]*1e-9
    else:
        h = None
    #h = None
    #--------------------

    n_elems = len(geometry['elems'])
    dim = int(geometry['meta'][2])

    iff = [];jff = [];dff = []

    B = np.zeros(n_elems)
    for ll in geometry['active_sides']:

      area  = geometry['areas'][ll] 
      (i,j) = geometry['side_elem_map_vec'][ll]
      vi    = geometry['volumes'][i]
      vj    = geometry['volumes'][j]
      kappa_loc = get_kappa(geometry,i,j,ll)
      if not i == j:
     
       #(v_orth,_) = get_decomposed_directions(geometry,get_key(ll,kappa_loc))
       (v_orth,_) = get_decomposed_directions(geometry,ll,kappa_loc)

       iff.append(i)
       jff.append(i)
       dff.append(v_orth/vi*area)
       iff.append(i)
       jff.append(j)
       dff.append(-v_orth/vi*area)
       iff.append(j)
       jff.append(j)
       dff.append(v_orth/vj*area)
       iff.append(j)
       jff.append(i)
       dff.append(-v_orth/vj*area)
       if ll in geometry['periodic_sides']:    
        kk = list(geometry['periodic_sides']).index(ll)   
        vi = geometry['volumes'][i]
        B[i] += geometry['periodic_side_values'][kk]*v_orth/vi*area
        B[j] -= geometry['periodic_side_values'][kk]*v_orth/vj*area


    for value,ll in zip(geometry['fixed_temperature'],geometry['fixed_temperature_sides']):    
          area = geometry['areas'][ll] 
          (i,j) = geometry['side_elem_map_vec'][ll]
          vi = geometry['volumes'][i]
          #(v_orth,_) = get_decomposed_directions(mesh,get_key(ll,kappa_loc))
          (v_orth,_) = get_decomposed_directions(geometry,ll,kappa_loc)
          if h == None:  #Dirichlet B.C.
             a = v_orth
          else: 
             a = v_orth*h/(h + v_orth) 

          iff.append(i)
          jff.append(i)
          dff.append(a/vi*area)

          if not argv.setdefault('DSA',False):
           B[i] += value*a/vi*area

    F  = sp.csc_matrix((np.array(dff),(np.array(iff),np.array(jff))),shape = (n_elems,n_elems))

    # if argv.setdefault('advection',False) :
    #    F += sp.eye(n_elems,format='csc')
    #-------
    
    if argv.setdefault('DSA',False) :
       #Synthetic Diffusion Approximation--
       # We set to zero all the external perturbations
       B = argv['temp_error']
    #--------


    #Fix point
    scale = fix_instability(F,B,scale=False,floating=len(geometry['fixed_temperature_sides'])==0)

    return splu(F),B,scale






