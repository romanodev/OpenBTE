from openbte.objects import Material, Mesh,SolverResults,BoundaryConditions,List,f64,EffectiveThermalConductivity
from openbte.heat_source     import HeatSource
import numpy as np
from scipy.sparse import linalg 
import scipy.sparse as sp
import openbte.utils as utils
import matplotlib.pylab as plt
from cachetools import cached,LRUCache
from nptyping import NDArray, Shape, Float, Int



cache_assembly = LRUCache(maxsize=1e6)

def Fourier(geometry                       : Mesh,\
            thermal_conductivity           : NDArray[Shape["[dim,dim]"], Float],\
            boundary_conditions            : BoundaryConditions,\
            effective_thermal_conductivity : EffectiveThermalConductivity = None,\
            heat_source                    : NDArray = [],\
            global_temperature             : NDArray[Shape["[n_volumes]"], Float] = 0,\
            modified_fourier               : bool  = False,\
            fourier_max_iter               : Int   = 40,\
            fourier_rtol                   : Float = 1e-8,\
            verbose                        : bool  = True)->SolverResults:
    r"""Solve the steady-state heat conduction equation:

  .. math::
   -\nabla \cdot \kappa \nabla T(\mathbf{r}) = H(\mathbf{r})

  :param geometry: geometry
  :param thermal_conductivity: bulk thermal conductivity :math:`\kappa`. dim is the dimension of the system. Units are :math:`Wm^{-1}K^{-1}` is dim = 2 and :math:`WK^{-1}` if dim = 2. 
  :param heat_source: volumetric heat source :math:`H(\mathbf{r})`. n_volumes is the number of volumes.  Units are :math:`Wm^{-3}` is dim = 3 and :math:`m^{-2}` if dim = 2.
  :param boundary_conditions: Boundary conditions
  :param effective_thermal_conductivity: Effective thermal conductivity
  :param global_temperature: Global temperature to be used in multiscale simulations (experimental)
  :param modified_fourier: Modified Fourier solver (experimental)
  :param fourier_max_iter: max number of iterations for loop termination
  :param fourier_rtol: relative error condition for loop termination
  :param verbose: whether to write output on screen
    """

    #This specifies the volume to be pinned to zero (for stability)
    N = 10

    #@cached(cache=cache_assembly)
    def assembly(kappa,modified_fourier):

     v_orth_v,v_nonorth_v =  geometry.get_decomposed_directions(kappa)

     d_vec = [];i_vec = [];j_vec = []
     for s in geometry.internal:
       c = geometry.side_centroids[s]
       #---------------------------- 
       i,j  = geometry.side_elem_map[s] 

       vi         = geometry.elem_volumes[i]
       vj         = geometry.elem_volumes[j]
       v_orth     = v_orth_v[s]

       i_vec.append(i); j_vec.append(i); d_vec.append( v_orth/vi)
       i_vec.append(i); j_vec.append(j); d_vec.append(-v_orth/vi)
       i_vec.append(j); j_vec.append(j); d_vec.append( v_orth/vj)
       i_vec.append(j); j_vec.append(i); d_vec.append(-v_orth/vj)


     #Boundary condition
     B = np.zeros(geometry.n_elems)

     #Dirichlet
     for key,contact in boundary_conditions.mixed.items():

        #Get all the sides given a key
        sides = geometry.side_physical_regions[key]

        for s in sides: 
         i          = geometry.side_elem_map[s][0]   
         vi         = geometry.elem_volumes[i]

         h          = contact['boundary_conductance']
         v_orth = v_orth_v[s]

         #Boundary condutance
         tmp =  v_orth * h/(h + v_orth/geometry.side_areas[s])

         i_vec.append(i); j_vec.append(i); d_vec.append( tmp/vi)
         B[i] = contact['value'] * tmp/vi

     #Periodic
     for region,jump in boundary_conditions.periodic.items():

        area = 0 
        for s in geometry.periodic_sides[region]:
         area += geometry.side_areas[s]   
         i,j        = geometry.side_elem_map[s]
         vi         = geometry.elem_volumes[i]
         vj         = geometry.elem_volumes[j]
         B[i]      += jump * v_orth_v[s]/vi
         B[j]      -= jump * v_orth_v[s]/vj

     #Add heat source
     if not len(heat_source) == 0 :
      if not modified_fourier:
       #B += heat_source.variables['Heat_Source']['data']*heat_source.aux['heat_source_coeff'] #Heat source is in K (like as in the BTE)
       #else:
       B += heat_source*1e-18 #from W/m^3 tp W/nm^2/m [kappa is still in W/m/K]
      else: 
       B += heat_source #K

     F   = sp.csc_matrix((d_vec,(i_vec,j_vec)),shape = (geometry.n_elems,geometry.n_elems))

     #modified Fourier
     if modified_fourier:
      F += sp.eye(geometry.n_elems,format='csc')
     else:   
      if not effective_thermal_conductivity == None:
       #Fix a point for stability
       scale = np.ones(geometry.n_elems); scale[N] = 0
       F.data = F.data * scale[F.indices]
       F[N,N] = 1
       #-----------------------------

     lu = linalg.splu(F)

     #Prepare data for nonorthogonal contribution
     inds  = geometry.side_elem_map[geometry.internal]
     w     = geometry.interp_weights[geometry.internal]
     vi    = geometry.elem_volumes[inds[:,0]]
     vj    = geometry.elem_volumes[inds[:,1]]
     v_nonorth = v_nonorth_v[geometry.internal]
     #------------------------------------

     return lu,B,w,inds,v_nonorth,vi,vj

    thermal_conductivity = thermal_conductivity[:geometry.dim,:geometry.dim]

    lu,B,w,inds,v_nonorth,vi,vj = assembly(thermal_conductivity,modified_fourier)

   
    if modified_fourier:
       B += global_temperature
    else:   
       B[N] = 0
    #-----------------------

    C = np.zeros_like(B); norm_C_old = 0
    error = 1
    n_iter = 0
    kappa_eff = 0
    while error > fourier_rtol and n_iter < fourier_max_iter:

        n_iter +=1
        T        = lu.solve(B+C)
        #Normalized T 
        if (not effective_thermal_conductivity == None) and (not modified_fourier == True):
         T -= (max(T) + min(T))*0.5
        
        gradient = geometry.gradient(boundary_conditions,T)
       
        #Nonorthogonal contribution (only on internal sides)
        C = np.zeros_like(B) 
        F_ave = np.einsum('k,ki->ki',w,gradient[inds[:,0]]) + np.einsum('k,ki->ki',1-w,gradient[inds[:,1]])
        tmp =  np.einsum('ki,ki->k',F_ave,v_nonorth)
        np.add.at(C,inds[:,0], tmp/vi)
        np.add.at(C,inds[:,1],-tmp/vj)
        #--------------

        norm_C = np.linalg.norm(C)
        
        error  = 0 if norm_C == 0 else abs(norm_C-norm_C_old)/norm_C
        norm_C_old = norm_C

        if not effective_thermal_conductivity == None:
            kappa_eff = effective_thermal_conductivity.get_diffusive(T,geometry,boundary_conditions,thermal_conductivity)
     
        if verbose:
         strc = "Kappa (Fourier): {:E}, Iterations {:n}, Error {:E}".format(kappa_eff,n_iter,error) 
         print(strc)
    else:   
       kappa_eff = 0

    J = -np.einsum('ij,cj->ci',thermal_conductivity,gradient)

    variables = {'Temperature_Fourier':{'data':T,'units':'K'},'Flux_Fourier':{'data':J,'units':'W/m/m'}}
    return SolverResults(kappa_eff = kappa_eff,variables=variables)


