import nlopt
import numpy as np
import shelve
import os
import time
import nlopt
from functools import partial,update_wrapper
import jax
from jax import value_and_grad as value_and_grad
from jax import  jit
import scipy
import jax.numpy as jnp
import scipy
import scipy.sparse.linalg as spla
import scipy.sparse as sp
from icosphere import icosphere
import openbte.utils as utils
from jax import custom_vjp
from typing import Callable
import openbte.inverse.matinverse as mi



def get_reflection(s):
    #Get a reflection vector

    M = s.shape[0]
    reflection = np.zeros(M)
    for a,sa1 in enumerate(s):
        sa1 /= np.linalg.norm(sa1)
        found = False
        for b,sa2 in enumerate(s):
            sa2 /= np.linalg.norm(sa2)

            if abs(np.dot(sa1,sa2) + 1) < 1e-3:
                reflection[a] = int(b)
                found = True
                break
            
        if not found:
            print('No reflection found',a,sa1)
            quit()

    return np.array(reflection,int)


def get_directions(dim,**options):

    if dim == 2:
     n_phi = options.setdefault('n_phi',48)
     Dphi  = 2*np.pi/n_phi
     phi   = np.linspace(0,2.0*np.pi-Dphi,n_phi,endpoint=True)
     s = jnp.array([np.cos(phi),np.sin(phi)]).T 

    if dim == 3: 
     nu = options['nu']  
     vertices, faces = icosphere(nu)

     vv = vertices[faces]
     npts= vv.shape[0]

     s = np.zeros((npts,3))
     for n,v in enumerate(vv):
       tmp    = np.mean(v,axis=0)
       s[n] = tmp/np.linalg.norm(tmp)

    return s,get_reflection(s) 


@jax.jit
def compute_kappa_and_gradient(x,data,dir_dep):

       a = time.time()
       ki,kj,P = dir_dep

       [_,_,t_flat,_,dt_sparse,dt_sparseT,i_mat,j_mat,gm,gp,_,GG,_,reflection,f,gm_reflected] = data

       (nq,n_elems) = P.shape

       T_mat = x.reshape((nq,n_elems))

       #Apply symmetry relationship
       G_mat = -T_mat[reflection]

       #Compute kappa--------
       kappa  = -jnp.einsum('n,un   ->',t_flat[ki],gm[:,ki],optimize=True)
       kappa +=  jnp.einsum('uc,uc->',P,G_mat)

       #Perturbation---
       g      =   jnp.zeros(n_elems)

       tmp  = jnp.einsum('uk,uk->k',G_mat[:,i_mat[ki]],gm[:,ki]) + jnp.einsum('uk,uk->k',G_mat[:,j_mat[ki]],gp[:,ki])
       g   = g.at[i_mat[ki]].add(-dt_sparse[ki] *tmp)
       g   = g.at[j_mat[ki]].add(-dt_sparseT[ki]*tmp)
       

       #Matrix
       tmp = jnp.einsum('qk,qk,qk->k',gm,T_mat[:,j_mat],G_mat[:,i_mat])
       g = g.at[i_mat].add(-tmp*dt_sparse)
       g = g.at[j_mat].add(-tmp*dt_sparseT)

       #Boundary
       tmp = jnp.einsum('qk,uk,uk,qk->k',gm,GG,T_mat[:,i_mat],G_mat[:,i_mat])
       g = g.at[i_mat].add(-tmp*dt_sparse)
       g = g.at[j_mat].add(-tmp*dt_sparseT)

       #From kappa
       tmp = gm[:,ki].sum(axis=0) + jnp.einsum('uk,uk->k',T_mat[:,i_mat[ki]],gp[:,ki]) + jnp.einsum('uk,uk->k',T_mat[:,j_mat[ki]],gm[:,ki])
       g   = g.at[i_mat[ki]].add(-dt_sparse[ki] *tmp)
       g   = g.at[j_mat[ki]].add(-dt_sparseT[ki]*tmp) 

       return [kappa/f,g/f]



def get_common(**options):
    #Parse--
    directions = options['directions']
    grid       = options['grid']
    n_dir,dim  = np.array(directions).shape
    aux        = mi.get_grid(grid,dim)

    N          = int(grid**2) if dim == 2 else int(grid**3)
    Knt        = options['Kn']
    Kn         = Knt*grid
    s,reflection = get_directions(dim,**options)

    M = len(s)

    #factor---
    f = 1/2*Kn**2*M if dim == 2 else 1/3 * grid *Kn**2*M
    
    #New----------------------------
    G        = Kn*jnp.einsum('qj,nj->qn',s,aux['normals'],optimize=True)
    gp       = G.clip(min=0); gm = G.clip(max=0)
    GG       = jnp.einsum('qn,n->qn',gp,1/gm.sum(axis=0))
    D        = jnp.zeros((N,M)).at[aux['i']].add(gp.T).T
    #-------------------------------

    return [aux['i'],aux['j'],gm,gp,D,GG,aux['ind_extremes'],reflection,f,(N,M)]

@jit
def sparse_dense_product(i,j,data,X):
     return jnp.zeros_like(X.T).at[i].add(data.T * X.T[j]).T


@partial(jax.jit)
def get_rho_dependent(rho,aux2):

        [i_mat,j_mat,gm,gp,D,GG,ind_extremes,reflection,f,(N,M)] = aux2

        dim = len(ind_extremes)
        k0 = 1e-12;k1 = 1
        (nq,n_elems) = D.shape
        rho = k0 + rho*(k1-k0)
        t_flat = 2*rho[i_mat] * rho[j_mat]/(rho[i_mat] + rho[j_mat])
        dt_sparse  = (k1-k0)*0.5*jnp.power(t_flat/rho[i_mat],2) 
        dt_sparseT = (k1-k0)*0.5*jnp.power(t_flat/rho[j_mat],2) 

        gm_direct = jnp.einsum('un,n->un',gm,t_flat)
        gp_direct = jnp.einsum('un,n->un',gp,t_flat)
        gm_reflected = gm_direct-gm

        P_vec     = jnp.zeros((dim,n_elems,nq))

        for i in range(dim):        
         ii_1  = ind_extremes[i,1]   
         ii_0  = ind_extremes[i,0]

         P_vec =  P_vec.at[(i,i_mat[ii_1])].add((-gm_direct[:,ii_1]).T )
         P_vec =  P_vec.at[(i,i_mat[ii_0])].add(( gm_direct[:,ii_0]).T )

        return gm_direct,gp_direct,t_flat,P_vec,dt_sparse,dt_sparseT,i_mat,j_mat,gm,gp,D,GG,ind_extremes,reflection,f,gm_reflected


@jax.jit
def operator(X,aux3):
    D,i_mat,j_mat,gm_direct,GG,gm_reflected = aux3
   
    M,N = D.shape
    X = X.reshape(D.shape)
 
    boundary = jnp.zeros_like(D.T).at[i_mat].add(jnp.einsum('un,qn,qn->nu',gm_reflected,X[:,i_mat],GG)).T - X.sum(axis=0)/M

    return (X + jnp.multiply(D,X)  + sparse_dense_product(i_mat,j_mat,gm_direct,X) + boundary).flatten()




def bte(**options)->Callable:

  common = get_common(**options)
  directions = options['directions']
  n_dir,dim = np.array(directions).shape
  (N,M) = common[-1]

  #fourier_solver = mi.compose(fourier(**options))
   
  x0        = np.zeros((n_dir,M*N))
  kappa     = np.zeros(n_dir)
  jacobian  = np.zeros((n_dir,N))

  def func(rho):

    aux = get_rho_dependent(rho,common)

    [gm_direct,gp_direct,t_flat,P_vec,dt_sparse,dt_sparseT,i_mat,j_mat,gm,gp,D,GG,ind_extremes,reflection,f,gm_reflected] = aux

    L         = partial(operator,aux3=(D,i_mat,j_mat,gm_direct,GG,gm_reflected))

    for n,direction in enumerate(directions):

     P     = jnp.einsum('i,iuc->cu',direction,P_vec)   

     if direction == [1,0,0]:
         kj = ind_extremes[0,0]
         ki = ind_extremes[0,1]

     if direction == [0,1,0]:
         kj = ind_extremes[1,0]
         ki = ind_extremes[1,1]

     if direction == [0,0,1]:
         kj = ind_extremes[2,0]
         ki = ind_extremes[2,1]

     if direction==[1,0]:
          kj = ind_extremes[0,0]
          ki = ind_extremes[0,1]

     if direction==[0,1]:
          kj = ind_extremes[1,0]
          ki = ind_extremes[1,1]

     dir_dep = ki,kj,P

     kappa_and_gradient = partial(compute_kappa_and_gradient,data=aux,dir_dep=dir_dep)

     (kappa[n],jacobian[n]),(T_mat,call_count)  = utils.solver(L,P.flatten(),x0[n],kappa_and_gradient,verbose=False,\
                                                  early_termination=True,tol=options.setdefault('atol',1e-4))
     x0[n,:] = T_mat
     #flux = jnp.einsum('uc,ui->ci',T_mat.reshape(D.shape),sigma)

    #return (kappa/f,T_mat.reshape((nq,n_elems))),jacobian/f
    return (kappa,1),jacobian

  func.total_call = 0
  return func

def get_solver(**kwargs):

    return mi.compose(bte(**kwargs))


