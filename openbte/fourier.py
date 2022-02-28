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


def solve_fourier_single(**argv):
   """ Solve Fourier with a single set of kappa """

   mesh = argv['geometry']

   if 'kappa_map_external' in argv.keys(): #read external kappa map
     mesh['elem_kappa_map'] = argv['kappa_map_external']
   else: 
     mesh['elem_kappa_map'] = get_kappa_map_from_mat(**argv)

    
   kappa = -1 #it means that it will take the map from mesh

   F,B,scale = assemble(mesh,argv['material'],kappa)

   meta,temp,grad =  solve_convergence(argv,kappa,B,splu(F),scale=scale,generation=mesh['generation']+argv.setdefault('additional_heat_source',np.zeros_like(mesh['generation'])))

   kappa_map = get_kappa_map(mesh,kappa)

   flux = -np.einsum('cij,cj->ci',kappa_map,grad)*1e9

   #add an extra dimension if needed

   dim = int(mesh['meta'][2])
   n_elems = len(mesh['elems'])
   if dim == 2:flux = np.append(flux, np.zeros((n_elems,1)), axis=1)

   if argv.setdefault('verbose',True):
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

    if ll in (list(mesh['cold_sides']) + list(mesh['hot_sides'])):
       v_orth = np.dot(normal,np.dot(rot,normal))/np.dot(normal,dist)
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




def compute_secondary_flux(temp,kappa_map,**argv):

   mesh = argv['geometry']


   gradT = compute_grad_common(temp,mesh)

   dim = int(mesh['meta'][2])
   n_elems = len(mesh['elems'])

   C = np.zeros(n_elems)
   for ll in mesh['active_sides']:
      area = mesh['areas'][ll] 
      (i,j) = mesh['side_elem_map_vec'][ll]
      kappa_loc = get_kappa(mesh,i,j,ll,kappa_map)
      (v_orth,v_non_orth) = get_decomposed_directions(mesh,get_key(ll,kappa_loc))
      if not i==j:
       w =  mesh['interp_weigths'][ll]
       F_ave = (w*gradT[i] + (1.0-w)*gradT[j])
       tmp = np.dot(F_ave,v_non_orth)*area
       C[i] += tmp
       C[j] -= tmp
      #else:
      #  if ll in (list(mesh['cold_sides'])+list(mesh['hot_sides'])):
      #    F_ave = gradT[i] 
      #    tmp = np.dot(F_ave,v_non_orth)*area
      #    C[i] += tmp

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

    if ll in mesh['hot_sides']: #Isothermal
     kappa_eff += np.einsum('ij,j,i->',kappa_loc,argv['grad'][i],mesh['face_normals'][ll][:dim])*area #this is because here flux is in
    else:
     deltaT = temp[i] - temp[j] - 1 #Periodic
     kappa_eff -= v_orth *  deltaT * area


   return (kappa_eff+k_non_orth)*mesh['meta'][1]


def solve_convergence(argv,kappa,B,SU,log=False,scale=[],generation=None):

    argv.setdefault('max_fourier_error',1e-10) 
    argv.setdefault('max_fourier_iter',20) 

    C  = np.zeros_like(B)

    mesh = argv['geometry']  

    kappa_map = get_kappa_map(mesh,kappa)

    n_elems = len(mesh['elems'])
    dim = int(mesh['meta'][2])
    grad = np.zeros((n_elems,dim))

    n_iter = 0;kappa_old = 0;error = 1;C_old = np.zeros(n_elems)
    while error > argv['max_fourier_error'] and \
                  n_iter < argv['max_fourier_iter'] :

        RHS = B + C + generation
    
        if len(scale) > 0: RHS *=scale

        temp = SU.solve(RHS)
        #temp = temp - (max(temp)+min(temp))/2.0

        argv['grad'] = grad
        kappa_eff = compute_diffusive_thermal_conductivity(temp,kappa_map,**argv)
        kappa_old = kappa_eff
        n_iter +=1
        #TODO: check error on residual
        C,grad = compute_secondary_flux(temp,kappa_map,**argv)

        error = (np.linalg.norm(C-C_old))/np.linalg.norm(C)
        C_old = C.copy()

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


#@cached(cache=cache_assemble, key=lambda mesh, kappa: hashkey(kappa))
def assemble(mesh,material,kappa):

    kappa_map = get_kappa_map(mesh,kappa)

    #Boundary conductance
    if 'boundary_conductance' in material.keys():
        h =  material['boundary_conductance'][0]*1e-9
    else:
        h = None
    #--------------------

    n_elems = len(mesh['elems'])
    dim = int(mesh['meta'][2])

    iff = [];jff = [];dff = []

    B = np.zeros(n_elems)
    for ll in mesh['active_sides']:

      area = mesh['areas'][ll] 
      (i,j) = mesh['side_elem_map_vec'][ll]
      vi = mesh['volumes'][i]
      vj = mesh['volumes'][j]
      kappa_loc = get_kappa(mesh,i,j,ll,kappa_map)
      if not i == j:
     
       (v_orth,_) = get_decomposed_directions(mesh,get_key(ll,kappa_loc))

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
        vi = mesh['volumes'][i]
        B[i] += mesh['periodic_side_values'][kk]*v_orth/vi*area
        B[j] -= mesh['periodic_side_values'][kk]*v_orth/vj*area


    for value,ll in zip(mesh['fixed_temperature'],mesh['fixed_temperature_sides']):    
          area = mesh['areas'][ll] 
          (i,j) = mesh['side_elem_map_vec'][ll]
          (v_orth,_) = get_decomposed_directions(mesh,get_key(ll,kappa_loc))
          if h == None:  #Dirichlet B.C.
             a = v_orth
          else:  
             a = v_orth*h/(h + v_orth) 
          iff.append(i)
          jff.append(i)
          dff.append(a/vi*area)
          B[i] += value*a/vi*area



    F = sp.csc_matrix((np.array(dff),(np.array(iff),np.array(jff))),shape = (n_elems,n_elems))

    scale = fix_instability(F,B,scale=False,floating=(len(mesh['cold_sides'])+len(mesh['hot_sides']))==0)

    return F,B,scale






