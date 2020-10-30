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
from shapely.geometry import LineString
import scipy

comm = MPI.COMM_WORLD


def solve_fourier_single(kappa,**argv):

   mesh = argv['mesh']
   dim = int(mesh['meta'][2])
   n_elems = int(mesh['meta'][0])

   F,B = assemble(mesh,kappa)
   
   SU = splu(F.tocsc())
   meta,temp,grad =  solve_convergence(argv,kappa,B,SU)
   flux = -np.einsum('cij,cj->ci',kappa,grad)
 
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

  
def get_decomposed_directions(mesh,ll,kappa):

    if np.isscalar(kappa):
       rot = [] 
    elif  kappa.ndim == 3:
       i,j = mesh['side_elem_map_vec'][ll]
       rot = get_kappa(mesh,i,j,ll,kappa)
    else:   
       rot = kappa  

    dim = int(mesh['meta'][2])
    normal = mesh['face_normals'][ll,0:dim]
    dist   = mesh['dists'][ll,0:dim]

    if len(rot) == 0: #to speed up a bit
      v_orth = 1/np.dot(normal,dist)
      v_non_orth = normal - dist*v_orth
    else:  
     rot = rot[:dim,:dim]
     v_orth = np.dot(normal,np.dot(rot,normal))/np.dot(normal,dist)
     v_non_orth = np.dot(rot,normal) - dist*v_orth

    
    return v_orth,v_non_orth[:dim]


def compute_grad(temp,**argv):

   mesh = argv['mesh'] 
   
   #Compute grad
   if not 'rr' in argv['cache']:

     #Compute deltas-------------------   
     rr = [i*[0]   for k,i in enumerate(mesh['n_side_per_elem']) ]

     for ll in mesh['active_sides'] :
      kc1,kc2 = argv['mesh']['side_elem_map_vec'][ll]
      ind1 = list(mesh['elem_side_map_vec'][kc1]).index(ll)
      ind2 = list(mesh['elem_side_map_vec'][kc2]).index(ll)
      if ll in mesh['periodic_sides']:
         delta = mesh['periodic_side_values'][list(mesh['periodic_sides']).index(ll)]
      else: delta = 0   
      rr[kc1][ind1] = [kc2,kc1,delta] 
      rr[kc2][ind2] = [kc1,kc2,-delta] 
 
     argv['cache']['rr'] = rr
   else:
     rr = argv['cache']['rr']  



   diff_temp = [[temp[j[0]]-temp[j[1]]+j[2] for j in f] for f in rr]
   gradT = np.array([np.einsum('js,s->j',mesh['weigths'][k,:,:mesh['n_non_boundary_side_per_elem'][k]],np.array(dt)) for k,dt in enumerate(diff_temp)])
   return gradT




def compute_secondary_flux(temp,kappa,**argv):

   
   #Compute gradient
   gradT = compute_grad(temp,**argv)
   #--------------

   mesh = argv['mesh'] 
   cache = argv['cache'] 
   dim = int(mesh['meta'][2])
   n_elems = len(argv['mesh']['elems'])

   #-----------SAVE some time--------------------------------------
   if not 'v_non_orth' in cache.keys():
    v_non_orth = {}
    for ll in mesh['active_sides']:
      (i,j) = mesh['side_elem_map_vec'][ll]
      if not i==j:

       (v_orth,v_non_orth[ll]) = get_decomposed_directions(mesh,ll,kappa)

       area   = mesh['areas'][ll]   
       normal = mesh['face_normals'][ll,0:dim]
       dist   = mesh['dists'][ll,0:dim]
    cache['v_non_orth'] = v_non_orth
   #-----------------------------------------------------------------

   C = np.zeros(n_elems)
   for ll in mesh['active_sides']:
      area = mesh['areas'][ll] 
      (i,j) = mesh['side_elem_map_vec'][ll]
      if not i==j:
       w =  mesh['interp_weigths'][ll]
       F_ave = (w*gradT[i] + (1.0-w)*gradT[j])
       tmp = np.dot(F_ave,cache['v_non_orth'][ll])*area
       C[i] += tmp
       C[j] -= tmp

   return C/mesh['volumes'],gradT




def compute_diffusive_thermal_conductivity(temp,kappa,**argv):


   mesh = argv['mesh']  
   dim = int(mesh['meta'][2])
   kappa_eff = 0
   k_non_orth = 0
   for ll in mesh['flux_sides']:

    (i,j) = mesh['side_elem_map_vec'][ll]
    area = mesh['areas'][ll]

    (v_orth,v_non_orth) = get_decomposed_directions(mesh,ll,kappa)

    deltaT = temp[i] - temp[j] - 1
    kappa_eff -= v_orth *  deltaT * area

    if 'grad' in argv.keys():
     gradT = argv['grad']  
     w  =  mesh['interp_weigths'][ll]
     grad_ave = w*gradT[i] + (1.0-w)*gradT[j]
     k_non_orth += np.dot(grad_ave,v_non_orth)/2 * area

   #print(kappa_eff,k_non_orth,kappa_eff+k_non_orth)

   return kappa_eff+k_non_orth*mesh['meta'][1]


def solve_convergence(argv,kappa,B,SU,log=False):

    C  = np.zeros_like(B)

    mesh = argv['mesh']   
    n_elems = len(mesh['elems'])
    dim = int(mesh['meta'][2])
    grad = np.zeros((n_elems,dim))

    n_iter = 0;kappa_old = 0;error = 1  
    while error > argv['max_fourier_error'] and \
                  n_iter < argv['max_fourier_iter'] :

        #argv['grad'] = grad
        RHS = B + C
        temp = SU.solve(RHS)
        temp = temp - (max(temp)+min(temp))/2.0
        kappa_eff = compute_diffusive_thermal_conductivity(temp,kappa,**argv)
        error = abs((kappa_eff - kappa_old)/kappa_eff)
        kappa_old = kappa_eff
        n_iter +=1
        C,grad = compute_secondary_flux(temp,kappa,**argv)

    return [kappa_eff,error,n_iter],temp,grad 


def assemble(mesh,kappa):

    iff = [];jff = [];dff = []

    n_elems = len(mesh['elems'])
    B = np.zeros(n_elems)
    for ll in mesh['active_sides']:

      area = mesh['areas'][ll] 
      (i,j) = mesh['side_elem_map_vec'][ll]
      vi = mesh['volumes'][i]
      vj = mesh['volumes'][j]
      if not i == j:

       (v_orth,dummy) = get_decomposed_directions(mesh,ll,kappa)

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

     if len(argv['cache']):
      mesh = argv['mesh']
      cache = {}
      F,B = assemble(mesh,1)
      cache['RHS_FOURIER'] =B   
      cache['F'] = F
      cache['SU'] = {}
      argv.update({'cache':cache})
    
     old_kappa = 0
     for m,k in enumerate(kappa):

         B  = k*argv['cache']['RHS_FOURIER'].copy() + argv.setdefault('DeltaT',np.zeros(n_elems)) 
         SU = argv['cache']['SU'].setdefault(m,splu(k*argv['cache']['F'] + sp.eye(n_elems)))
         log = True if m == 0 else False
         meta,tf[m,:],tfg[m,:,:] = solve_convergence(argv,k,B,SU,log=False)
         kappaf = meta[0]
         error = abs((kappaf-old_kappa))/kappaf
         if m > int(len(kappa)/4) and error < 1e-2:
          tf[m:,:]  = tf[m-1]; tfg[m:,:,:] = tfg[m-1] 
          break 
         else:  
          old_kappa = kappaf

    comm.Barrier()
    return tf,tfg



