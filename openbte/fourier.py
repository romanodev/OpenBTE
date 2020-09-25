from __future__ import absolute_import
import numpy as np
from scipy.sparse.linalg import splu
from termcolor import colored, cprint 
from .utils import *
from mpi4py import MPI
import scipy.sparse as sp
import time
import scikits.umfpack as um
from scipy.sparse.linalg import lgmres
import scikits.umfpack as um
import sys
from shapely.geometry import LineString
import scipy
from .solve_mfp import *

comm = MPI.COMM_WORLD


def compute_grad(temp,argv):

    #if not 'rr' in argv['cache']:

    rr = [i*[0]   for k,i in enumerate(argv['mesh']['n_side_per_elem']) ]

    for ll in argv['mesh']['active_sides'] :   
      kc1,kc2 = argv['mesh']['side_elem_map_vec'][ll]
      #if not kc1 == kc2 :
      #if 1 == 1:    
      ind1 = list(argv['mesh']['elem_side_map_vec'][kc1]).index(ll)
      ind2 = list(argv['mesh']['elem_side_map_vec'][kc2]).index(ll)
      temp_1 = temp[kc1]
      temp_2 = temp[kc2]

      if ll in argv['mesh']['periodic_sides']:
        delta= argv['mesh']['periodic_side_values'][list(argv['mesh']['periodic_sides']).index(ll)]
      else: 
       delta = 0   
 
      #if ll in argv['mesh']['boundary_sides']:
      #   delta = 1

      rr[kc1][ind1] = [kc2,kc1,delta] 
      rr[kc2][ind2] = [kc1,kc2,-delta]

    #argv['cache']['rr'] = rr

    #for r in rr: print(r)
    #diff_temp = [[temp[j[0]]-temp[j[1]]+j[2] for j in f] for f in rr]
    #gradT = np.array([np.einsum('js,s->j',argv['mesh']['weigths'][k,:,:argv['mesh']['n_non_boundary_side_per_elem'][k]],np.array(dt)) for k,dt in enumerate(diff_temp)])
    diff_temp = [[temp[j[0]]-temp[j[1]]+j[2] for j in f] for f in rr]
    gradT = np.array([np.einsum('js,s->j',argv['mesh']['weigths'][k,:,:argv['mesh']['n_non_boundary_side_per_elem'][k]],np.array(dt)) for k,dt in enumerate(diff_temp)])

    return gradT


def compute_secondary_flux(argv,temp,kappa):

   if not 'rr' in argv['cache']:

    rr = [i*[0]   for k,i in enumerate(argv['mesh']['n_side_per_elem']) ]

    for ll in argv['mesh']['active_sides'] :
     kc1,kc2 = argv['mesh']['side_elem_map_vec'][ll]
     ind1 = list(argv['mesh']['elem_side_map_vec'][kc1]).index(ll)
     ind2 = list(argv['mesh']['elem_side_map_vec'][kc2]).index(ll)
     temp_1 = temp[kc1]
     temp_2 = temp[kc2]

     if ll in argv['mesh']['periodic_sides']:
        delta= argv['mesh']['periodic_side_values'][list(argv['mesh']['periodic_sides']).index(ll)]
     else: 
        delta = 0   

 
     rr[kc1][ind1] = [kc2,kc1,delta] 
     rr[kc2][ind2] = [kc1,kc2,-delta] 

    argv['cache']['rr'] = rr


   diff_temp = [[temp[j[0]]-temp[j[1]]+j[2] for j in f] for f in argv['cache']['rr']]
   gradT = np.array([np.einsum('js,s->j',argv['mesh']['weigths'][k,:,:argv['mesh']['n_non_boundary_side_per_elem'][k]],np.array(dt)) for k,dt in enumerate(diff_temp)])
   #-----------------------------------------------------

   #-----------SAVE some time--------------------------------------
   if not 'v_non_orth' in argv['cache'].keys():
    v_non_orth = {}
    for ll in argv['mesh']['active_sides']:
      (i,j) = argv['mesh']['side_elem_map_vec'][ll]
      if not i==j:
       area = argv['mesh']['areas'][ll]   
       normal = argv['mesh']['face_normals'][ll,0:argv['dim']]
       dist   = argv['mesh']['dists'][ll,0:argv['dim']]
       v_orth = 1/np.dot(normal,dist)
       v_non_orth[ll] = ((normal - dist*v_orth)*area)[:argv['dim']]
    argv['cache']['v_non_orth'] = v_non_orth
   #-----------------------------------------------------------------

   C = np.zeros(argv['n_elems'])
   for ll in argv['mesh']['active_sides']:
      (i,j) = argv['mesh']['side_elem_map_vec'][ll]
      if not i==j:
       w = argv['mesh']['interp_weigths'][ll]
       F_ave = (w*gradT[i] + (1.0-w)*gradT[j])*kappa
       tmp = np.dot(F_ave,argv['cache']['v_non_orth'][ll])
       C[i] += tmp
       C[j] -= tmp

   return C/argv['mesh']['volumes'],gradT




def compute_diffusive_thermal_conductivity(argv,temp,kappa):

   kappa_eff = 0
   for ll in argv['mesh']['flux_sides']:

    (i,j) = argv['mesh']['side_elem_map_vec'][ll]

    normal = argv['mesh']['face_normals'][ll]
    dist   = argv['mesh']['dists'][ll]
    v_orth = 1/np.dot(normal,dist)
    deltaT = temp[i] - (temp[j] + 1)

    kappa_eff -= kappa * v_orth *  deltaT * argv['mesh']['areas'][ll]
    w  = argv['mesh']['interp_weigths'][ll]

   return kappa_eff*argv['kappa_factor']



def fourier_scalar(m,kappa,DeltaT,C,argv):

    C *=kappa 
    n_elems = len(argv['mesh']['elems'])
    dim = int(argv['mesh']['meta'][2])

    n_iter = 0
    kappa_old = 0
    error = 1  

    #tmp = np.zeros(n_elems)
    #for n,e in enumerate(argv['mesh']['eb']):
    #   area = argv['mesh']['areas'][argv['mesh']['eb'][n]]
    #   volume = argv['mesh']['volumes'][e]
       #tmp[e] += argv['TB'][n]*area/volume

    B = kappa*argv['cache']['RHS_FOURIER'].copy() + DeltaT #+ argv['thermal_conductance'][m]*argv['TB']*area/volume

    if m in argv['cache']['SU'].keys():
        SU = argv['cache']['SU'][m]
    else:
        F = kappa*argv['cache']['F'] + sp.eye(n_elems) #+ argv['thermal_conductance'][m]* kappa*argv['cache']['F_B']
        argv['cache']['SU'][m] = splu(F)
    SU = argv['cache']['SU'][m]

   
    while error > argv['max_fourier_error'] and \
                  n_iter < argv['max_fourier_iter'] :

        RHS = B + C
        temp = SU.solve(RHS)
        temp = temp - (max(temp)+min(temp))/2.0
        kappa_eff = compute_diffusive_thermal_conductivity(argv,temp,1)
        error = abs((kappa_eff - kappa_old)/kappa_eff)
        kappa_old = kappa_eff
        n_iter +=1
        C,grad = compute_secondary_flux(argv,temp,kappa)

    flux = -grad*kappa
    return kappa_eff,temp,grad,C/kappa


def assemble_fourier(argv):

  if not 'cache' in argv.keys():
    mesh = argv['mesh']

    cache = {}
    
    iff = []
    jff = []
    dff = []


    iffb = []
    jffb = []
    dffb = []

    n_elems = len(mesh['elems'])
    B = np.zeros(n_elems)
    Bb = np.zeros(n_elems)
    for ll in mesh['active_sides']:

      area = mesh['areas'][ll] 
      (i,j) = mesh['side_elem_map_vec'][ll]
      vi = mesh['volumes'][i]
      vj = mesh['volumes'][j]
      if not i == j:

       normal = mesh['face_normals'][ll]
       dist   = mesh['dists'][ll]
       v_orth = 1/np.dot(normal,dist)

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
      else:  
       iffb.append(i)
       jffb.append(i)
       dffb.append(1/vi*area)
       Bb[i] += 1/vi*area

        



    cache['RHS_FOURIER'] =B   
    cache['RHS_FOURIER_B'] =Bb
    cache['F'] = sp.csc_matrix((np.array(dff),(np.array(iff),np.array(jff))),shape = (n_elems,n_elems))
    cache['F_B'] = sp.csc_matrix((np.array(dffb),(np.array(iffb),np.array(jffb))),shape = (n_elems,n_elems))
    cache['SU'] = {}
    argv.update({'cache':cache})
    


def solve_fourier(kappa,DeltaT,argv):


    tf = shared_array(np.zeros((argv['n_serial'],argv['n_elems'])) if comm.rank == 0 else None)
    tfg = shared_array(np.zeros((argv['n_serial'],argv['n_elems'],argv['dim'])) if comm.rank == 0 else None)

    if comm.rank == 0:

     assemble_fourier(argv)

     old_kappa = 0
     C = np.zeros(argv['n_elems'])
     for m,k in enumerate(kappa):
         kappaf,tf[m,:],tfg[m,:,:],C = fourier_scalar(m,k,DeltaT,C,argv)
         if m > int(len(kappa)/4) and abs((kappaf-old_kappa))/kappaf < 1e-2:
          tf[m:,:]  = tf[m-1]; tfg[m:,:,:] = tfg[m-1] 
          break 
         else:  
          old_kappa = kappaf

    comm.Barrier()
    return tf,tfg
