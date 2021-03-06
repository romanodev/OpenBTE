from __future__ import absolute_import
import numpy as np
from scipy.sparse.linalg import splu
from termcolor import colored, cprint 
from .utils import *
from mpi4py import MPI
import scipy.sparse as sp
import time
from scipy.sparse.linalg import lgmres
import scikits.umfpack as um
import sys
#from shapely.geometry import LineString
import scipy
from cachetools import cached
from cachetools.keys import hashkey


comm = MPI.COMM_WORLD

@cached(cache={}, key=lambda F, k: hashkey(k))
def get_SU(F,k):
     return splu(k*F + sp.eye(F.shape[0]))

def get_key(ll,kappa):


   return (ll,tuple(map(tuple,kappa)))

def unpack(data):

    return data[0],np.array(data[1])


def fix(F,B):

   n_elems = F.shape[0]
   scale = 1/F.max(axis=0).toarray()[0]
   n = np.random.randint(n_elems)
   scale[n] = 0
   F.data = F.data * scale[F.indices]
   F[n,n] = 1
   B[n] = 0   

   return scale


def solve_fourier_single(kappa,**argv):

   mesh = argv['mesh']
   dim = int(mesh['meta'][2])
   n_elems = int(mesh['meta'][0])

   kappa = -1 #it means that it will take the map from mesh

   F,B = assemble(mesh,kappa)

   if argv.setdefault('fix',True): 
       scale = fix(F,B)
   else:
       scale = []

   meta,temp,grad =  solve_convergence(argv,kappa,B,splu(F),scale=scale)

   kappa_map = get_kappa_map(mesh,kappa)

   flux = -np.einsum('cij,cj->ci',kappa_map,grad)
   return {'flux_fourier':flux,'temperature_fourier':temp,'meta':np.array(meta),'grad':grad}



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

   kappa_loc = kj*kappa_i/(ki*(1-w) + kj*w)

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


@cached(cache={}, key=lambda mesh,kappa_ll:hashkey(kappa_ll))
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



@cached(cache={},key=lambda mesh,dummy:dummy)
def compute_grad_data(mesh,dummy):

     #Compute deltas-------------------   
     rr = []
     for i in mesh['side_per_elem']:
       rr.append(i*[0])

     #this is wrong

     for ll in mesh['active_sides'] :
      kc1,kc2 = mesh['side_elem_map_vec'][ll]
      ind1 = list(mesh['elem_side_map_vec'][kc1]).index(ll)
      ind2 = list(mesh['elem_side_map_vec'][kc2]).index(ll)
      if ll in mesh['periodic_sides']:
         delta = mesh['periodic_side_values'][list(mesh['periodic_sides']).index(ll)]
      else: delta = 0  
      rr[kc1][ind1] = [kc2,kc1,delta] 
      rr[kc2][ind2] = [kc1,kc2,-delta] 
 
     return rr


def compute_grad(temp,**argv):

   mesh = argv['mesh'] 
   
   rr = compute_grad_data(mesh,1)

   diff_temp = [[temp[j[0]]-temp[j[1]]+j[2] for j in f] for f in rr]
   
   gradT = np.array([np.einsum('js,s->j',mesh['weigths'][k,:,:mesh['n_non_boundary_side_per_elem'][k]],np.array(dt)) for k,dt in enumerate(diff_temp)])
   return gradT




def compute_secondary_flux(temp,kappa_map,**argv):

   
   #Compute gradient
   gradT = compute_grad(temp,**argv)
   #--------------

   mesh = argv['mesh'] 
   dim = int(mesh['meta'][2])
   n_elems = len(argv['mesh']['elems'])

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
    


   mesh = argv['mesh']  

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

    C  = np.zeros_like(B)

    mesh = argv['mesh']  

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
        #print(min(temp),max(temp))

        kappa_eff = compute_diffusive_thermal_conductivity(temp,kappa_map,**argv)
        #quit()
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


@cached(cache={}, key=lambda mesh, kappa: hashkey(kappa))
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

    return F,B




def solve_fourier(kappa,DeltaT,argv):

    argv['DeltaT'] = DeltaT

    n_elems = len(argv['mesh']['elems'])

    tf = shared_array(np.zeros((argv['n_serial'],argv['n_elems'])) if comm.rank == 0 else None)
    tfg = shared_array(np.zeros((argv['n_serial'],argv['n_elems'],argv['dim'])) if comm.rank == 0 else None)

    if comm.rank == 0:
     dim = int(argv['mesh']['meta'][2])


     F,B = assemble(argv['mesh'],tuple(map(tuple,np.eye(dim))))

     ratio_old = 0
     ratio = np.zeros_like(kappa)
     for m,k in enumerate(kappa):
         
         BB = k*B+argv['DeltaT']
         meta,tf[m,:],tfg[m,:,:] = solve_convergence(argv\
                                                        ,k*np.eye(dim),BB,\
                                                         get_SU(F,k))
         kappaf = meta[0]
         ratio[m] = kappaf/k

         if m > int(len(kappa)/4):
           error = abs((ratio[m]-ratio[m-1]))/ratio[m]
           if error < 1e-2:
             tf[m:,:]  = tf[m-1]; tfg[m:,:,:] = tfg[m-1]
             ratio[m:] = ratio[m]
             break

     #Regularize---
     if argv.setdefault('experimental_multiscale',False):
      kappa_map = get_kappa_map(argv['mesh'],np.eye(dim))
      ratio_0 = compute_diffusive_thermal_conductivity(DeltaT,kappa_map,**argv)
      grad_DeltaT = compute_grad(DeltaT,**argv)
    
      #Regularize gradient
      for m1 in range(len(kappa))[::-1]:
        error = (ratio[m1] - ratio_0)/ratio[m1]
        if (ratio[m1] - ratio_0)/ratio[m1] < 1e-1 and abs(ratio[m1]-ratio[m1-1])/ratio[m1] < 1e-1:
            tfg[:m1+1,:,:] = grad_DeltaT
            tf[:m1+1,:] = DeltaT
            break

      #Regularize average temperature
      for m1 in range(len(kappa))[::-1]:
        error = np.linalg.norm(tf[m1] - DeltaT)/np.linalg.norm(DeltaT)
        if error < 1e-1 :
            tf[:m1+1,:] = DeltaT
            break
     #-----------------

    comm.Barrier()
    return tf,tfg



