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


def ggmres(L,i,j,eb,D,Gm,GG,Gbm2,W,C,sigma,kappa_mask,X,compute_kappa,residual):

      def compute_norm_D(A,D):
       return np.sqrt(np.einsum('ij,i,ij->',A,D,A))

      print(' ')
      print(colored('                        G-GMRES','green'),flush=True)
      print(colored(' -----------------------------------------------------------','green'),flush=True)


      #PARSE---------------------------------------
      nr = 20
      #--------------------------------------------

      #SET UP---
      k = nr


      def arnoldi(Q,H):

       for j in range(k):
         #Arnoldi-------------------------------------
         V         = L(Q[j])
         H[:,j]    = np.einsum('ikj,k,kj->i',Q,DD,V)
         V        -= np.einsum('ikj,i->kj',Q,H[:,j])
         H[j+1,j]  = compute_norm_D(V,DD)
         Q[j+1]    = V/H[j+1,j]   

       return Q,H

      for kk in range(100):
       a = time.time()   

       X0 = X
       R = residual(X0)

       #Compute WEIGHTS------------------------------
       #Method 3
       DD = np.mean(np.absolute(R),axis=1)
       #---------------------------------------------  

       R0   = compute_norm_D(R,DD)
       Q = np.zeros((k+1,X0.shape[0],X0.shape[1]))
       Q[0] = R/R0
       H = np.zeros((k+1,k))
       beta2 = np.zeros(k+1);beta2[0] = R0

       Q,H = arnoldi(Q,H)
       #--------------
       
       c   = np.linalg.lstsq(H,beta2)[0]
       X   = X0 + np.einsum('knm,k->nm',Q[:-1],c)
       r   = np.linalg.norm(residual(X),ord='fro')

       #kappa = np.einsum('u,uc,c->',sigma[:,0],X,kappa_mask)
       kappa = compute_kappa(X)

       #error = R/R0
       print('{0:8d} {1:24.4E} {2:22.4E}'.format(kk,kappa,r),flush=True)


      return X,kappa
  

def sparse_dot(rows,cols,data,b,n_elems):

    c = np.zeros(n_elems)
    np.add.at(c,rows,data * b[cols])

    return c

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
   max_bte_iter   =  10000
   max_bte_error = 1e-6
   GGMRES = argv.setdefault('ggmres',False)

   #Import data---

   mesh    = argv['geometry']
   mat     = argv['material']
   f = argv['fourier']
   factor  = mat['alpha'][0][0]
   W       = mat['W'] 
   Wdiag   = np.diag(W)
   sigma   = mat['sigma']*1e9
   Wod     = W-np.diag(np.diag(W))

   n_elems = len(mesh['elems'])
   X0 = np.tile(f['temperature'],(W.shape[0],1))
   grad_fourier = f['grad']

   #preparation
   G = np.einsum('qj,jn->qn',sigma,mesh['k'],optimize=True)
   Gp = G.clip(min=0); Gm = G.clip(max=0)
   n_elems = len(mesh['elems'])
   D = np.zeros((W.shape[0],n_elems))
   np.add.at(D.T,mesh['i'],Gp.T)
   sss = np.asarray(mesh['pp'][:,0],dtype=int)
   P = np.zeros_like(D)
   np.add.at(P.T,mesh['i'][sss],-(mesh['pp'][:,1]*Gm[:,sss]).T)

   #Boundary----
   if len(mesh['db']) > 0: #boundary

      Gb = np.einsum('qj,jn->qn',sigma,mesh['db'],optimize=True)
      Gbp2 = Gb.clip(min=0); Gbm2 = Gb.clip(max=0);
      np.add.at(D.T,mesh['eb'],Gbp2.T)
      tot = 1/Gb.clip(max=0).sum(axis=0); 
      GG  = np.einsum('qs,s->qs',Gbp2,tot)
   del G,Gp,Gbp2,tot
   #-------

   i = mesh['i']
   j = mesh['j']
   im = np.concatenate((mesh['i'],list(np.arange(n_elems))))
   jm = np.concatenate((mesh['j'],list(np.arange(n_elems))))
   data = np.concatenate((Gm,D+Wdiag[...,None]),axis=1)
   X = np.tile(f['temperature'],(W.shape[0],1))
   #--------------------

   Winv = load_data('Winv')['Winv']

   CT = Wdiag/np.sum(Wdiag)

   kappa_mask = mesh['kappa_mask']/factor
   kappa_old  = f['meta'][0]
   eb = mesh['eb']

   (nm,n_elems) = X.shape

   #Post processing---

   def get_boundary(X):

      B = np.zeros_like(X)  
      TB =  np.einsum('us,us->s',X[:,eb],GG)
      tmp = np.einsum('c,nc->cn',TB,Gbm2)
      np.add.at(B.T,eb,tmp)

      return B


   L = lambda X: np.multiply(D,X) + sparse_dense_product(i,j,Gm,X) + np.matmul(W,X) - get_boundary(X)

   def residual(X):
     return P  - L(X) 

   @cached(cache={})
   def get_lu(q):
        A = sp.csc_matrix((data[q],(im,jm)),shape=(n_elems,n_elems),dtype=np.float64)
        return sp.linalg.splu(A)

   def compute_kappa(X):
      return np.einsum('u,uc,c->',sigma[:,0],X,kappa_mask)

   print(flush=True)
   print('      Iter    Thermal Conductivity [W/m/K]     Residual''',flush=True)
   print(colored(' -----------------------------------------------------------','green'),flush=True)


   def solve_RTA(X,kappa_old):

    kk = 0
    error = 1
    while kk < 1 and error > max_bte_error:
      B =  np.einsum('uc,u,v->vc',X,CT,Wdiag) + P + get_boundary(X)
      X  = np.array([get_lu(q).solve(B[q]) for q in range(nm)])
      kappa = compute_kappa(X)
      error = abs(kappa_old-kappa)/abs(kappa)
      kappa_old = kappa
      print('{0:8d} {1:24.4E} {2:22.4E} {3:22.4E}'.format(kk,kappa,error,residual(X)),flush=True)
      kk += 1

    return X,kappa
     
   def solve_FULL(X,kappa_old):

    X_old = X.copy()   
    kk = 0
    error = 1
    error_old = 1
    alpha = 1
    while kk < max_bte_iter and error > max_bte_error:
      B = -np.dot(Wod,X) +  P + get_boundary(X)
      X  = np.array([get_lu(q).solve(B[q]) for q in range(nm)])
      kappa = compute_kappa(X)
      R     = residual(X)
      error = abs(kappa_old-kappa)/abs(kappa)
      if error/error_old > 2 and GGMRES and kk > 0:
         return X_old,kappa_old 
      error_old = error
      kappa_old = kappa
      print('{0:8d} {1:24.4E} {2:22.4E}'.format(kk,kappa,np.linalg.norm(R,ord='fro')),flush=True)
      kk +=1
      X = alpha * X + (1-alpha)*X_old
      X_old = X.copy() 

    return X,kappa
     
   X,kappa = solve_FULL(X,kappa_old)

   X,kappa = ggmres(L,i,j,eb,D,Gm,GG,Gbm2,W,P,sigma,kappa_mask,X,compute_kappa,residual)

   T = np.einsum('qc,q->c',X,tc)
   J = np.einsum('qj,qc->cj',sigma,X)*1e-18

   return {'kappa':kappa,'temperature':T,'flux':J}

   






