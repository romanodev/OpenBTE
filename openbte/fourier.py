from __future__ import absolute_import
import numpy as np
from scipy.sparse.linalg import splu
from termcolor import colored, cprint 
from .utils import *
from mpi4py import MPI
import scipy.sparse as sp
import time
from scipy.sparse.linalg import lgmres
#import scikits.umfpack as um
import sys
import scipy
from cachetools import cached,LRUCache
from cachetools.keys import hashkey

comm = MPI.COMM_WORLD

cache_assemble = LRUCache(maxsize=1000)
cache_compute_grad_data = LRUCache(maxsize=10000)
cache_get_SU = LRUCache(maxsize=10000)
cache_get_decomposed_directions = LRUCache(maxsize=10000)


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



@cached(cache=cache_get_SU, key=lambda bundle, k: hashkey(k))
def get_SU(bundle,k):

     A = k*bundle[0] + sp.diags(bundle[1],format='csc')
    
     return splu(A)

def get_key(ll,kappa):


   return (ll,tuple(map(tuple,kappa)))

def unpack(data):

    return data[0],np.array(data[1])


def get_kappa_map_from_mat(**argv):

        mat_map = argv['geometry']['elem_mat_map']
        dim = int(argv['geometry']['meta'][2])

        kappa = argv['material']['kappa']

        kappa = np.array(kappa)
        if kappa.ndim == 3:
            return np.array([ list(kappa[i][:dim,:dim])  for i in mat_map])
        else:
            return np.array([list(kappa)]*len(argv['geometry']['elems']))


def solve_fourier_single(argv):
   """ Solve Fourier with a single set of kappa """


   #data = None

   #if comm.rank == 0:

   mesh = argv['geometry']

   if 'kappa_map_external' in argv.keys(): #read external kappa map
     mesh['elem_kappa_map'] = argv['kappa_map_external']
   else: 
     mesh['elem_kappa_map'] = get_kappa_map_from_mat(**argv)
     #['kappa_map_external'] = mesh['elem_kappa_map']

    
   kappa = -1 #it means that it will take the map from mesh

   F,B,scale = assemble(mesh,kappa)

   meta,temp,grad =  solve_convergence(argv,kappa,B,splu(F),scale=scale)

   kappa_map = get_kappa_map(mesh,kappa)

   flux = -np.einsum('cij,cj->ci',kappa_map,grad)*1e9

   #add an extra dimension if needed

   dim = int(mesh['meta'][2])
   n_elems = len(mesh['elems'])
   if dim == 2:flux = np.append(flux, np.zeros((n_elems,1)), axis=1)


   if argv.setdefault('verbose',False):
     fourier_info(meta)
   
   data = {'meta':meta,'flux':flux,'temperature':temp,'grad':grad}


   clear_fourier_cache()

   return data



def get_kappa(mesh,i,j,ll,kappa):

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


@cached(cache=cache_get_decomposed_directions, key=lambda mesh,kappa_ll:hashkey(kappa_ll))
def get_decomposed_directions(mesh,kappa_ll):
     
    ll,kappa = unpack(kappa_ll)

    kappa_map = get_kappa_tensor(mesh,ll,kappa)

    dim = int(mesh['meta'][2])
    normal = mesh['face_normals'][ll,0:dim]
    dist   = mesh['dists'][ll,0:dim]

    rot = kappa_map[:dim,:dim]
    v_orth = np.dot(normal,np.dot(rot,normal))/np.dot(normal,dist)
    v_non_orth = np.dot(rot,normal) - dist*v_orth
    
    return v_orth,v_non_orth[:dim]


def compute_laplacian(temp,**argv):

  grad_T = compute_grad(temp,**argv).T

  laplacian = np.zeros_like(temp)
  for i,grad in enumerate(grad_T):
     laplacian +=   compute_grad(grad,**argv).T[i]

  return laplacian



@cached(cache=cache_compute_grad_data,key=lambda mesh,dummy:dummy)
def compute_grad_data(mesh,jump):

     #Compute deltas-------------------   
     rr = []
     for i in mesh['side_per_elem']:
       rr.append(i*[0])

     #this is wrong
     for ll in mesh['active_sides'] :
      kc1,kc2 = mesh['side_elem_map_vec'][ll]
      ind1    = list(mesh['elem_side_map_vec'][kc1]).index(ll)
      ind2    = list(mesh['elem_side_map_vec'][kc2]).index(ll)

      delta = 0
      if ll in mesh['periodic_sides']:
         delta = mesh['periodic_side_values'][list(mesh['periodic_sides']).index(ll)]
      else: delta = 0  
      if jump == 0: delta= 0

      rr[kc1][ind1] = [kc2,kc1, delta] 
      rr[kc2][ind2] = [kc1,kc2,-delta]
 
     return rr


def compute_grad(temp,**argv):

   mesh = argv['geometry'] 

   argv.setdefault('jump',True)
   rr = compute_grad_data(mesh,argv['jump'])

   diff_temp = [[temp[j[0]]-temp[j[1]]+j[2] for j in f] for f in rr]
   
   #Add boundary
   #if 'TB' in argv.keys():
   # TB  = argv['TB']
   # for n,(eb,sb) in enumerate(zip(*(mesh['eb'],mesh['sb']))):
   #   ind               = list(mesh['elem_side_map_vec'][eb]).index(sb)
   #   diff_temp[eb][ind] = temp[eb]-TB[n]

   gradT = np.array([np.einsum('js,s->j',mesh['weigths'][k,:,:mesh['n_non_boundary_side_per_elem'][k]],np.array(dt)) for k,dt in enumerate(diff_temp)])


   return gradT




def compute_secondary_flux(temp,kappa_map,**argv):

   
   #Compute gradient
   gradT = compute_grad(temp,**argv)
   #--------------

   mesh = argv['geometry'] 
   dim = int(mesh['meta'][2])
   n_elems = len(mesh['elems'])

   #-----------SAVE some time--------------------------------------
   v_non_orth = {}
   for ll in mesh['active_sides']:
      (i,j) = mesh['side_elem_map_vec'][ll]
      if not i==j:

       kappa_loc = get_kappa(mesh,i,j,ll,kappa_map)
       (v_orth,v_non_orth[ll]) = get_decomposed_directions(mesh,get_key(ll,kappa_loc))

       area   = mesh['areas'][ll]   
       normal = mesh['face_normals'][ll,0:dim]
       dist   = mesh['dists'][ll,0:dim]
   #-----------------------------------------------------------------

   C = np.zeros(n_elems)
   for ll in mesh['active_sides']:
      area = mesh['areas'][ll] 
      (i,j) = mesh['side_elem_map_vec'][ll]
      if not i==j:
       w =  mesh['interp_weigths'][ll]
       F_ave = (w*gradT[i] + (1.0-w)*gradT[j])
       #tmp = np.dot(F_ave,cache['v_non_orth'][ll])*area
       tmp = np.dot(F_ave,v_non_orth[ll])*area
       C[i] += tmp
       C[j] -= tmp

   return C/mesh['volumes'],gradT




def compute_diffusive_thermal_conductivity(temp,kappa_map,**argv):
    
   mesh = argv['geometry']  

   dim = int(mesh['meta'][2])
   kappa_eff = 0
   k_non_orth = 0
   area_tot = 0
   for ll in mesh['flux_sides']:
    (i,j) = mesh['side_elem_map_vec'][ll]
    area = mesh['areas'][ll]

    kappa_loc = get_kappa(mesh,i,j,ll,kappa_map)
    (v_orth,v_non_orth) = get_decomposed_directions(mesh,get_key(ll,kappa_loc))

    deltaT = temp[i] - temp[j] - 1
    kappa_eff -= v_orth *  deltaT * area

    if 'grad' in argv.keys():
     gradT = argv['grad']  
     w  =  mesh['interp_weigths'][ll]
     grad_ave = w*gradT[i] + (1.0-w)*gradT[j]
     k_non_orth += np.dot(grad_ave,v_non_orth)/2 * area
    area_tot +=area

   
   return (kappa_eff+k_non_orth)*mesh['meta'][1]


def solve_convergence(argv,kappa,B,SU,log=False,scale=[]):

    argv.setdefault('max_fourier_error',1e-6) 
    argv.setdefault('max_fourier_iter',10) 

    C  = np.zeros_like(B)

    mesh = argv['geometry']  

    kappa_map = get_kappa_map(mesh,kappa)

    n_elems = len(mesh['elems'])
    dim = int(mesh['meta'][2])
    grad = np.zeros((n_elems,dim))

    
    n_iter = 0;kappa_old = 0;error = 1  
    while error > argv['max_fourier_error'] and \
                  n_iter < argv['max_fourier_iter'] :

        RHS = B + C
    
        if len(scale) > 0: RHS *=scale

        temp = SU.solve(RHS)
        temp = temp - (max(temp)+min(temp))/2.0

        kappa_eff = compute_diffusive_thermal_conductivity(temp,kappa_map,**argv)

        error = abs((kappa_eff - kappa_old)/kappa_eff)
        kappa_old = kappa_eff
        n_iter +=1
        C,grad = compute_secondary_flux(temp,kappa_map,**argv)


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


@cached(cache=cache_assemble, key=lambda mesh, kappa: hashkey(kappa))
def assemble(mesh,kappa):

    kappa_map = get_kappa_map(mesh,kappa)

    n_elems = len(mesh['elems'])
    dim = int(mesh['meta'][2])

    iff = [];jff = [];dff = []

    B = np.zeros(n_elems)
    for ll in mesh['active_sides']:

      area = mesh['areas'][ll] 
      (i,j) = mesh['side_elem_map_vec'][ll]
      vi = mesh['volumes'][i]
      vj = mesh['volumes'][j]
      if not i == j:
     
       kappa_loc = get_kappa(mesh,i,j,ll,kappa_map)
      
       (v_orth,dummy) = get_decomposed_directions(mesh,get_key(ll,kappa_loc))

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
       if ll in mesh['periodic_sides']:    
        kk = list(mesh['periodic_sides']).index(ll)   
        B[i] += mesh['periodic_side_values'][kk]*v_orth/vi*area
        B[j] -= mesh['periodic_side_values'][kk]*v_orth/vj*area

    F = sp.csc_matrix((np.array(dff),(np.array(iff),np.array(jff))),shape = (n_elems,n_elems))

    scale = fix_instability(F,B,scale=False)

    return F,B,scale






