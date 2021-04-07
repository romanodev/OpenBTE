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


def solve_full(argv):

   #here you go---------------------------
   max_bte_iter   =  argv.setdefault('max_bte_iter',100)
   max_ggmres_iter   =  argv.setdefault('max_ggmres_iter',100)
   tolerance = argv.setdefault('tolerance',1e-13)
   GGMRES = argv.setdefault('ggmres',True)

   #---------------------------


   #Import data---
   mesh    = argv['geometry']
   mat     = argv['material']
   f = argv['fourier']
   factor  = mat['alpha'][0][0]
   W       = mat['W'] 
   Wdiag   = np.diag(W)
   sigma   = mat['sigma']*1e9
   Wod     = W-np.diag(np.diag(W))
   Winv    = load_data('Winv')
   WS      = load_data('winvsigma')

   gradT = compute_grad(f['temperature'],**argv)
   CT   = Wdiag/np.sum(Wdiag)

   #parallel set up------------
   n_parallel = len(sigma)
   block =  n_parallel//comm.size
   rr = range(block*comm.rank,n_parallel) if comm.rank == comm.size-1 else range(block*comm.rank,block*(comm.rank+1))
   n_block = len(rr)


   n_elems = len(mesh['elems'])
   X0 = np.tile(f['temperature'],(W.shape[0],1))
   grad_fourier = f['grad']

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
   X  = shared_array(np.tile(f['temperature'],(W.shape[0],1)))
   R  = shared_array(np.zeros_like(X))
   OP = shared_array(np.zeros_like(X))
   #--------------------

   #Winv = load_data('Winv')['Winv']


   kappa_mask = mesh['kappa_mask']/factor
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

   alpha = argv.setdefault('alpha',1)  
   def solve_FULL(X,kappa_old):
    X_old = X.copy()   
    kk = 0
    error = 1
    error_old = 1
    r_old = 1e4
    while kk < max_bte_iter and r_old > tolerance:

      #B =  np.einsum('vc,v,u->uc',X_old,CT,Wdiag)[rr] + P + get_boundary(X_old)
      B        = -np.dot(Wod[rr],X_old) + P  + get_boundary(X_old)
      X[rr,:]  =  np.array([get_lu(n).solve(B[n]) for n,q in enumerate(rr)])
      
      kappa = compute_kappa(X)
      R     = residual(X)
      r = np.linalg.norm(R,ord='fro')
      if r > r_old and GGMRES and kk > 0:
         return X_old,kappa_old,True 
      r_old = r
      if comm.rank == 0:
       print('{0:8d} {1:24.4E} {2:22.4E}'.format(kk,kappa,r,flush=True))
      kk +=1
      comm.Barrier()
      X_old = alpha*X.copy()+ (1-alpha)*X_old


    return X,kappa,False
    
   X,kappa,gg = solve_FULL(X,kappa_old)

   if gg:
    X,kappa = ggmres(L,X,compute_kappa,residual,20,rr,max_ggmres_iter,tolerance)

   comm.Barrier()
   T = np.einsum('qc,q->c',X,CT)
   J = np.einsum('qj,qc->cj',sigma,X)*1e-18/factor

   bte =  {'kappa':[kappa],'temperature':T,'flux':J}
 
   argv['bte'] = bte


   






