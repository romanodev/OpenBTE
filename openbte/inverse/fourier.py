import numpy as np
import scipy.sparse as sp
from typing import Callable
import openbte.inverse.matinverse as mi
import time
import jax
from jax import numpy as jnp
from functools import partial
import scipy.sparse.linalg as spla
from functools import lru_cache
import scipy
import openbte.utils as utils

@jax.jit
def get_aux(rho,eta_fourier,i_mat,j_mat,ind_extremes):
    
     k0 = 1e-12;k1 = 1.0
     kappa_map       = k0 + rho*(k1-k0)
     kappa_ij     = 2*kappa_map[i_mat] * kappa_map[j_mat]/(kappa_map[i_mat] + kappa_map[j_mat])

     kappad_ij    = (k1-k0)*0.5*jnp.power(kappa_ij/kappa_map[i_mat],2)
     kappad_ji    = (k1-k0)*0.5*jnp.power(kappa_ij/kappa_map[j_mat],2)

     kappad_ij    = eta_fourier*jnp.power(kappa_ij,eta_fourier-1)*kappad_ij
     kappad_ji    = eta_fourier*jnp.power(kappa_ij,eta_fourier-1)*kappad_ji

     kappa_ij     = jnp.power(kappa_ij,eta_fourier)

     #assembly
     v1 = jnp.zeros_like(rho).at[i_mat].add(kappa_ij)
     data = jnp.hstack((-kappa_ij,v1))
 
     N   = len(rho)
     dim = len(ind_extremes)

     i = jnp.hstack((i_mat,jnp.arange(N)))
     j = jnp.hstack((j_mat,jnp.arange(N)))

     P_vec   = jnp.zeros((dim,N))

     #compute vectorial perturbation
     for i in range(dim):
      ii  = ind_extremes[i,1]   
      P_vec   = P_vec.at[(i,i_mat[ii])].add(  kappa_ij[ii])
      P_vec   = P_vec.at[(i,j_mat[ii])].add( -kappa_ij[ii])

     return data,i,j,kappa_ij,kappad_ij,kappad_ji,P_vec


partial(jax.jit,static_argnums=(2,))
def compute_kappa_and_gradient(T,rho_dep,N):

      [kappa_ij,P,i_mat,j_mat,ii,kappad_ij,kappad_ji] = rho_dep
      #Compute kappa---
      kappa = jnp.sum(kappa_ij[ii]) - jnp.dot(T,P)
      #Compute gradient
      gradient = jnp.zeros(N).at[i_mat].add(kappad_ij*jnp.power(T[i_mat]-T[j_mat],2))
      gradient = gradient.at[i_mat[ii]].add(kappad_ij[ii]*(1-2*(T[i_mat[ii]]-T[j_mat[ii]])))
      gradient = gradient.at[j_mat[ii]].add(kappad_ji[ii]*(1+2*(T[j_mat[ii]]-T[i_mat[ii]])))

      return kappa,gradient

@jax.jit
def L(x,aux):
  
    kappa_ij,i_mat,j_mat = aux

    return  jnp.zeros_like(x).at[i_mat].add(kappa_ij*(x[i_mat]-x[j_mat]))


def fourier(**options)->Callable:
    """Fourier Solver"""

    direct = True
    directions =options['directions']
    n_dir,dim = np.array(directions).shape
    grid        = options['grid']
    if dim == 3:
     N           = int(grid**3)
     factor      = 1/grid
    else: 
     N           = int(grid**2)
     factor      = 1

    eta_fourier = options.setdefault('eta',1)
    aux         = mi.get_grid(grid,dim)
    i_mat       = aux['i']
    j_mat       = aux['j']
    ind_extremes = aux['ind_extremes'] 

    #For direct
    if direct:
     row = jnp.hstack((i_mat,jnp.arange(N))) 
     col = jnp.hstack((j_mat,jnp.arange(N))) 

    x0        = np.zeros((n_dir,N))

    def func(rho):
     
     data,i,j,kappa_ij,kappad_ij,kappad_ji,P_vec = get_aux(rho,eta_fourier,i_mat,j_mat,ind_extremes)

     aux = kappa_ij,i_mat,j_mat

     #Direct
     if direct:
      tmp = jnp.zeros_like(rho).at[i_mat].add(kappa_ij)
      d   = jnp.hstack((-kappa_ij,tmp))
      A   = scipy.sparse.csc_matrix((d, (row, col)), shape=(N, N))
      A[12,:] = 0;A[12,12] = 1
      lu = sp.linalg.splu(A)
     #---------------------

     #---------------------------------------
     kappa     = np.zeros(n_dir)
     jacobian  = np.zeros((n_dir,N))
     for n,direction in enumerate(directions):

      P = jnp.einsum('i,ic->c',direction,P_vec)   
     
      #--------------------------------------
      if direction == [1,0,0]:
         ii = ind_extremes[0,1]

      if direction == [0,1,0]:
         ii = ind_extremes[1,1]

      if direction == [0,0,1]:
         ii = ind_extremes[2,1]

      if direction == [1,0]:
         ii = ind_extremes[0,1]

      if direction == [0,1]:
         ii = ind_extremes[1,1]

      rho_dep = [kappa_ij,P,i_mat,j_mat,ii,kappad_ij,kappad_ji]

      kappa_and_gradient =  partial(compute_kappa_and_gradient,rho_dep=rho_dep,N=N)

      if not direct:
       #ITERATIVE
       (kappa[n],jacobian[n]),(T_mat,call_count)  = utils.solver(partial(L,aux=aux),P,x0[n],\
                                                   kappa_and_gradient,early_termination=True,tol = options.setdefault('atol',1e-7),verbose=False)
       x0[n,:] = T_mat
      else:
       #DIRECT-----------------------------------------
       P   = P.at[12].set(0.0)
       T_mat  = lu.solve(np.array(P))
       kappa[n],jacobian[n] = kappa_and_gradient(T_mat)
      #-------------------------------------------------
     
     return (kappa*factor,1),jacobian*factor

   
    return func

def get_solver(**kwargs):

    return mi.compose(fourier(**kwargs))

