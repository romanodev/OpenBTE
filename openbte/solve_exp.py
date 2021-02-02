from __future__ import absolute_import
import numpy as np
from scipy.sparse.linalg import splu
from termcolor import colored, cprint 
from openbte.utils import *
import scipy.sparse as sp
import time
import sys
import scipy
from cachetools import cached
from scipy.sparse.linalg import gmres
import scipy.sparse.linalg as spla
from matplotlib.pyplot import *
from openbte.utils import *
from openbte.fourier import *
from mpi4py import MPI

comm = MPI.COMM_WORLD




def sparse_dense_product(i,j,data,X):
     '''
     This solves B_ucc' X_uc' -> A_uc

     B_ucc' : sparse in cc' and dense in u. Data is its vectorized data COO descrition
     X      : dense matrix
     '''

     tmp = np.zeros_like(X)
     np.add.at(tmp.T,i,data.T * X.T[j])   

     return tmp


def solve_exp(argv):

   #here you go---------------------------
   max_bte_iter   =  10000
   max_bte_error = 1e-6
   #---------------------------


   #Import data------------------------
   mesh    = argv['geometry']
   mat     = argv['material']
   f       = argv['fourier']
   Wdiag   = mat['Wdiag']
   sigma   = mat['sigma']*1e9
   CT = Wdiag/np.sum(Wdiag)
   #-----------------------------------


   #parallel set up------------
   n_parallel = len(sigma)
   block =  n_parallel//comm.size
   rr = range(block*comm.rank,n_parallel) if comm.rank == comm.size-1 else range(block*comm.rank,block*(comm.rank+1))
   n_block = len(rr)

   n_elems = len(mesh['elems'])
   X0 = np.tile(f['temperature'],(n_parallel,1))

   #preparation
   G = np.einsum('qj,jn->qn',sigma[rr],mesh['k'],optimize=True)
   Gp = G.clip(min=0); Gm = G.clip(max=0)
   n_elems = len(mesh['elems'])
   D = np.zeros((n_block,n_elems))
   np.add.at(D.T,mesh['i'],Gp.T)

   #Boundary----
   if len(mesh['db']) > 0: #boundary

      Gb = np.einsum('qj,jn->qn',sigma,mesh['db'],optimize=True)
      Gbp2 = Gb.clip(min=0); Gbm2 = Gb.clip(max=0);
      np.add.at(D.T,mesh['eb'],Gbp2[rr].T)
      tot = 1/Gb.clip(max=0).sum(axis=0); 
      GG  = np.einsum('qs,s->qs',Gbp2,tot)

   #Periodic---
   sss = np.asarray(mesh['pp'][:,0],dtype=int)
   P = np.zeros_like(D)
   np.add.at(P.T,mesh['i'][sss],-(mesh['pp'][:,1]*Gm[:,sss]).T)
   del G,Gp,Gbp2,tot
   #-------

   i = mesh['i']
   j = mesh['j']
   im = np.concatenate((mesh['i'],list(np.arange(n_elems))))
   jm = np.concatenate((mesh['j'],list(np.arange(n_elems))))
   data = np.concatenate((Gm,D+Wdiag[rr,None]),axis=1)
   X  = shared_array(np.tile(f['temperature'],(n_parallel,1)))
   R  = shared_array(np.zeros_like(X))
   OP = shared_array(np.zeros_like(X))
   #--------------------

   kappa_mask = mesh['kappa_mask']
   kappa_old  = f['meta'][0]
   eb = mesh['eb']

   (nm,n_elems) = X.shape

   #Post processing---

   def get_boundary(X):

      B = np.zeros((n_block,n_elems)) 
      #For now this is global--
      TB =  np.einsum('us,us->s',X[:,eb],GG)
      tmp = np.einsum('c,uc->cu',TB,Gbm2)
      #------------------------
      np.add.at(B.T,eb,tmp[:,rr])

      return B

   def L(X):

       OP[rr,:] = np.multiply(D,X[rr]) + sparse_dense_product(i,j,Gm,X[rr]) + np.matmul(W[rr,:],X) - get_boundary(X)

       comm.Barrier() 
       
       return OP


   def L_RTA(X):

       OP[rr,:] = np.multiply(D,X[rr]) + sparse_dense_product(i,j,Gm,X[rr]) + np.einsum('uc,u->uc',X,Wdiag)  - np.einsum('vc,v,u->uc',X,CT,Wdiag) - get_boundary(X)

       comm.Barrier() 

       return OP


   def residual_rta(X):
     R= L_RTA(X) 
     R[rr,:] -= P
     comm.Barrier()

     return -np.einsum('u,uc->uc',1/Wdiag,R)

   def residual(X):
     R = L(X) 
     R[rr,:] -= P
     comm.Barrier()
     return -R

   @cached(cache={})
   def get_lu(q):
        A = sp.csc_matrix((data[q],(im,jm)),shape=(n_elems,n_elems),dtype=np.float64)
        return sp.linalg.splu(A)


   def compute_kappa(X):
      kappa = np.zeros(1)
      kappap = np.einsum('u,uc,c->',sigma[rr,0],X[rr],kappa_mask)
      comm.Allreduce([np.array([kappap]),MPI.DOUBLE],[kappa,MPI.DOUBLE],op=MPI.SUM)
      comm.Barrier()
      return kappa[0]



   if comm.rank == 0:
    print(flush=True)
    print('      Iter    Thermal Conductivity [W/m/K]     Residual''',flush=True)
    print(colored(' -----------------------------------------------------------','green'),flush=True)


   def solve_RTA(X,kappa_old):
    X_old = X.copy()   
    kk = 0
    error = 1
    error_old = 1
    while kk < max_bte_iter and error > max_bte_error:
      B        = P  + get_boundary(X) + np.einsum('vc,v,u->uc',X,CT,Wdiag)
      X[rr,:]  = np.array([get_lu(n).solve(B[n]) for n,q in enumerate(rr)])

      comm.Barrier()
      kappa = compute_kappa(X)
      R     = residual_rta(X)

      error = abs(kappa_old-kappa)/abs(kappa)
      if error/error_old > 2 and GGMRES and kk > 0:
         return X_old,kappa_old 
      error_old = error
      kappa_old = kappa
      if comm.rank == 0:
       print('{0:8d} {1:24.4E} {2:22.4E}'.format(kk,kappa,np.linalg.norm(R,ord='fro')),flush=True)
      kk +=1
      comm.Barrier()
      X_old = X.copy()

    return X,kappa

     
    



   #X = X - np.einsum('ui,ci->uc',WS,gradT)   

   X,kappa = solve_RTA(X,kappa_old)

   #X,kappa = solve_FULL(X,kappa_old)

   quit()

   X,kappa = ggmres(L,X,compute_kappa,residual,20,rr)

   T = np.einsum('qc,q->c',X,tc)
   J = np.einsum('qj,qc->cj',sigma,X)*1e-18

   return {'kappa':kappa,'temperature':T,'flux':J}

   






