from openbte.objects import Mesh,Material,SolverResults,BoundaryConditions,List,EffectiveThermalConductivity
from openbte.heat_source     import HeatSource
import scipy.sparse as sp
from multiprocessing import Process,current_process
from multiprocessing import Array,Value
from multiprocessing import Barrier
from functools import partial
from multiprocessing import set_start_method
from multiprocessing import Lock
import openbte.utils as utils
import os
from cachetools import cached,LRUCache
import numpy as np
import time
from nptyping import NDArray, Shape, Float, Int

set_start_method('fork',force=True)

def BTE_RTA(geo                               : Mesh,\
            mat                               : Material,\
            bcs                               : BoundaryConditions,\
            fourier                           : SolverResults = None,\
            effective_thermal_conductivity    : EffectiveThermalConductivity = None,\
            heat_source                       : HeatSource = None,\
            direct                            : bool  = True,\
            bte_max_iter                      : Int   = 100,\
            bte_rtol                          : Float = 1e-5)->SolverResults:

    #parse material data
    mfps = mat.mfps
    t_coeff = mat.t_coeff

    if fourier == None:
        DeltaT = np.zeros(geo.n_elems)
    else:    
        DeltaT = fourier.variables['Temperature_Fourier']['data']
    #----------------------------


    #This preprossing is to include the ji indices from the ij ones (stored in Geometry). Also, we only consider internal sides
    i       =  geo.side_elem_map[:,0] 
    j       =  geo.side_elem_map[:,1]
    dk1     =  geo.normal_areas[geo.internal]/geo.elem_volumes[i[geo.internal]][:,np.newaxis]
    dk2     = -geo.normal_areas[geo.internal]/geo.elem_volumes[j[geo.internal]][:,np.newaxis]
    k_mat   =  np.vstack((dk1,dk2))
    i_mat   =  np.hstack((i[geo.internal],j[geo.internal]))
    j_mat   =  np.hstack((j[geo.internal],i[geo.internal]))
    #------------------------------------------

    #Internal properties-----------------------------------
    G        = np.einsum('qj,nj->qn',mat.vmfp[:,:geo.dim],k_mat,optimize=True)
    Gp       = G.clip(min=0)
    Gm = G.clip(max=0)
    D        = np.zeros((mat.n_angles,geo.n_elems))
    np.add.at(D.T,i_mat,Gp.T)
 
    #Boundary--------------------------------------------
    Gbm2 = 0 #Dummy placeholder
    GG = 0 #Dummy placeholder
    n_elems   = geo.n_elems
    if len(bcs.diffuse) > 0:
     s_diffuse = geo.side_physical_regions[bcs.diffuse] #Side of the diffuse boundary
     i_diffuse  = i[s_diffuse] 
     db      = geo.normal_areas[s_diffuse]/geo.elem_volumes[i_diffuse][:,np.newaxis]
     Gb      = np.einsum('mqj,sj->mqs',mat.sigma[:,:,:geo.dim],db,optimize=True)
     Gbp2    = Gb.clip(min=0)
     tmp     = Gb.clip(max=0).sum(axis=0).sum(axis=0) #Diffuse scattering
     R       = np.divide(1, tmp, out=np.zeros_like(tmp), where=tmp!=0)
     tmp     = np.einsum('rj,nj->rn',mat.vmfp[:,:geo.dim],db,optimize=True)  
     Gbp2    = tmp.clip(min=0); 
     Gbm2 = tmp.clip(max=0)
     np.add.at(D.T,i_diffuse,Gbp2.T)
     Gbp = np.einsum('mqj,sj->mqs',mat.sigma[:,:,:geo.dim],db,optimize=True).clip(min=0)
     GG = np.einsum('mqs,s->mqs',Gbp,R)
    else: 
     i_diffuse = np.empty(0)
    n_diffuse = len(i_diffuse)

    #---Thermalizing boundary----------------------
    RHS_ISO  = np.zeros((mat.n_angles,geo.n_elems))
    for key,contact in bcs.mixed.items():
        sides = geo.side_physical_regions[key]
        tmp   = geo.normal_areas[sides]/geo.elem_volumes[i[sides]][:,np.newaxis]
        tmp   = np.einsum('rj,nj->rn',mat.vmfp,tmp,optimize=True)  
        np.add.at(RHS_ISO.T,i[sides],-(tmp.clip(max=0)*contact['value']).T)
        np.add.at(D.T,i[sides],tmp.clip(min=0).T)

    #---Periodic boundary----------------------
    P  = np.zeros((mat.n_angles,geo.n_elems))
    for region,jump in bcs.periodic.items():

        sides =  geo.periodic_sides[region]
        tmp   =  geo.normal_areas[sides]/geo.elem_volumes[i[sides]][:,np.newaxis]
        tmp   =  np.einsum('rj,nj->rn',mat.vmfp,tmp,optimize=True)  
        np.add.at(P.T,i[sides],-(tmp.clip(max=0)*jump).T)

        tmp   =  -geo.normal_areas[sides]/geo.elem_volumes[j[sides]][:,np.newaxis]
        tmp   =  np.einsum('rj,nj->rn',mat.vmfp,tmp,optimize=True)  
        np.add.at(P.T,j[sides],(tmp.clip(max=0)*jump).T)

    #---Perturbation due to nanowire
    if bcs.is_nanowire:
     P_nano = -mat.vmfp[:,2]
    else: 
     P_nano = np.zeros(mat.n_angles)   
 

    #Total Perturbation--
    perturbation = P + RHS_ISO + P_nano[:,np.newaxis]
    #--------------------

    #SuperLu sover-------------------
    im = np.concatenate((i_mat,list(np.arange(geo.n_elems))))
    jm = np.concatenate((j_mat,list(np.arange(geo.n_elems))))
    
    #Total heat source
    if heat_source == None:
        heat_source = np.zeros(geo.n_elems)
    else:    
        heat_source = heat_source.variables['Heat_Source']['data']

    if not effective_thermal_conductivity == None:
     #Preparation effective kappa
     #contact,normalization_factor = kwargs['kappa_eff_options']
     contact = effective_thermal_conductivity.contact
     normalization = effective_thermal_conductivity.normalization

     if bcs.is_nanowire:
      alpha = 0
      e2 = np.empty(0)
      e1 = np.arange(geo.n_elems)
      sigma_normal = np.einsum('ml,c->mlc',mat.sigma[:,:,2],geo.elem_volumes)
      is_nanowire = 1
     else:  
      is_nanowire = 0
      bc   = bcs.get_bc_by_name(contact)
      if bc == 'Mixed': 
       sides = geo.side_physical_regions[contact]
       e1    = geo.side_elem_map[sides][:,0]
       e2    = np.empty(0)
       alpha = bcs.mixed[contact]['value']
      else:  
       sides = geo.periodic_sides[contact]
       e1 = geo.side_elem_map[sides][:,0]
       e2 = geo.side_elem_map[sides][:,1]
       alpha = bcs.periodic[contact]
     sigma_normal = np.einsum('mli,ci->mlc',mat.sigma,geo.normal_areas[sides])
     side_areas = np.linalg.norm(geo.normal_areas[sides],axis=1)
     #--------------------------------
    K = np.zeros((mat.n_mfp,mat.n_angles))
    S = np.zeros_like(K)

    if len(i_diffuse) > 0:
     TB     = DeltaT[i_diffuse]
    else: 
     TB =   np.empty(0)   
    J = np.zeros((geo.n_elems,mat.vmfp.shape[1]))

    #Options
    #-----------------
    def core(d,tasks)->None:

        inds = tasks[0]

        #Init cache
        cache_compute_lu = LRUCache(maxsize=1e3)
 
        _,n_angles = mat.t_coeff.shape

        @cached(cache=cache_compute_lu)
        def compute_lu(n,m):
          d_mat  = np.concatenate((mfps[m]*Gm[n],mfps[m]*D[n]+np.ones(n_elems)))
          A      = sp.csc_matrix((d_mat,(im,jm)),shape=(n_elems,n_elems),dtype=np.float64)
          return  sp.linalg.splu(A)

        def solve_iterative(n,m,B,X0): #Faster for larger structures
          d_mat  = np.concatenate((mfps[m]*Gm[n],mfps[m]*[n]+np.ones(n_elems)))
          A      = sp.csc_matrix((d_mat,(im,jm)),shape=(n_elems,n_elems),dtype=np.float64)
          return sp.linalg.lgmres(A,B,x0=X0)[0]


        norm_T_old = np.linalg.norm(DeltaT)
        error = 1
        n_iter = 0 
        while error > bte_rtol and n_iter < bte_max_iter:
         n_iter +=1 

         if n_diffuse > 0:
           boundary   = -np.einsum('c,nc->nc',d['TB'],Gbm2) if n_diffuse > 0 else  np.zeros((n_angles,0))

         partial_TB = np.zeros(n_diffuse)
         partial_T  = np.zeros_like(DeltaT)
         partial_J  = np.zeros_like(J)
         partial_K  = np.zeros_like(K)
         partial_S  = np.zeros_like(S)
         
         T = np.zeros(n_elems)
         for n in inds:  
           for m,mfp in enumerate(mfps):
                 
                 B = d['DeltaT'] + mfp * perturbation[n] + heat_source * mat.heat_source_coeff

                 if n_diffuse > 0:
                  np.add.at(B,i_diffuse,mfp*boundary[n])
             
                 #solve
                 if direct:
                  T  =  compute_lu(n,m).solve(B)
                 else:
                  T  = solve_iterative(n,m,B,T)


                 #Contribute to DeltaT
                 partial_T += t_coeff[m,n]*T

                 #Contribute to TB
                 if n_diffuse > 0:
                  np.add.at(partial_TB,np.arange(n_diffuse),-T[i_diffuse]*GG[m,n])

                 #Contribute to Flux
                 partial_J += np.einsum('c,j->cj',T,mat.sigma[m,n])

                 if not effective_thermal_conductivity == None:
                  #Compute kappa_mode
                  if is_nanowire:
                   partial_K[m,n]  =  np.dot(T[e1],sigma_normal[m,n])*normalization
                  else:
                   partial_K[m,n]  =  np.dot(T[e1],sigma_normal[m,n].clip(min=0))*normalization
                   partial_S[m,n]  =  np.dot(T[e1],side_areas)*normalization

                  if len(e2) > 0:
                      partial_K[m,n] += np.dot(T[e2],sigma_normal[m,n].clip(max=0))*normalization

                  partial_K[m,n] += alpha*np.sum(sigma_normal[m,n].clip(max=0))*normalization

         d.reduce(    [['DeltaT',partial_T],\
                       ['TB',    partial_TB],\
                       ['J',     partial_J],\
                       ['S',     partial_S],\
                       ['K',     partial_K]])

         #Compute error--
         norm_T = np.linalg.norm(d['DeltaT'])
         error  = abs(norm_T-norm_T_old)/norm_T
         norm_T_old = norm_T
         
         d.print("Kappa (BTE): {:E}, Iterations {:n}, Error {:E}".format(np.sum(d['K']),n_iter,error))
         

    sh = utils.run(target=core,n_tasks = (mat.n_angles,),shared_variables = {'TB':TB,\
                                                                             'J':J,\
                                                                             'DeltaT':DeltaT,\
                                                                             'S':S,\
                                                                             'K':K})


    kappa_eff = np.sum(sh['K'])

    variables = {'Temperature_BTE':{'data':sh['DeltaT'],'units':'K'},'Flux_BTE':{'data':sh['J'],'units':'W/m/m'}}

    #variables.update({'Vorticity_BTE':{'data':geo.vorticity(bcs,sh['J']),'units':'W/m/m/m'}})

    return SolverResults(kappa_eff = kappa_eff,variables=variables,aux = {'suppression':sh['S']})
                   



