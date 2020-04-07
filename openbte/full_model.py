import deepdish as dd
import sys
import numpy as np
from scipy.sparse.linalg import lsqr 
from scipy.sparse.linalg import inv as sinv
from scipy.linalg import inv as inv
from scipy.sparse import csc_matrix
import scipy
import deepdish as dd



def compute_kappa(W,sigma):


 dd = len(W.nonzero()[0])/W.shape[0]/W.shape[1]    
 if dd > 0.1:   
  invW = inv(W)
  return np.einsum('ui,uq,qj->ij',sigma,invW,sigma)  #you can try this to be faster but the matrix is singular.
 else: 
  W = csc_matrix(W)
  
 a = np.zeros((3,sigma.shape[0]))
 for i in range(3):
  a[i],_,_,_,_,_,_,_,_,_ = lsqr(W,sigma[:,i])

 kappa = np.zeros((3,3))
 for i in range(3):
  for j in range(3):
      kappa[i,j] = np.dot(a[i],sigma[:,j])  

 return kappa



def compute_energy_conservation(W):

 bottom = np.sum(np.absolute(W))
 r = np.sum(np.absolute(np.einsum('ij->j',W)))
 print('')
 print('   After:')
 print('Row check:' + str(r/bottom))
 c = np.sum(np.absolute(np.einsum('ij->i',W)))
 print('Column Check:' + str(c/bottom))


def energy_conserving(W):

 print('   Before:')
 bottom = np.sum(np.absolute(W))
 r = np.sum(np.absolute(np.einsum('ij->j',W)))
 print('Row check:' + str(r/bottom))
 c = np.sum(np.absolute(np.einsum('ij->i',W)))
 print('Column Check:' + str(c/bottom))
 nm = np.shape(W)[0]
 delta = np.einsum('ij->j',W)
 b = -2*np.concatenate((delta,delta))
 A = np.zeros((2*nm,2*nm))
 A[0:nm,0:nm] = nm*np.eye(nm)
 A[nm:,nm:] = nm*np.eye(nm)
 A[0:nm,nm:] = 1
 A[nm:,0:nm] = 1
 l = np.linalg.solve(A,b)
 lr = l[:nm]
 lv = l[nm:]
 beta = np.zeros((nm,nm))
 for i in range(nm):
  for j in range(nm):
   beta[i,j] = -(lr[j]+lv[i])/2
 W -= beta

 bottom = np.sum(np.absolute(W))
 r = np.sum(np.absolute(np.einsum('ij->j',W)))
 print('')
 print('   After:')
 print('Row check:' + str(r/bottom))
 c = np.sum(np.absolute(np.einsum('ij->i',W)))
 print('Column Check:' + str(c/bottom))


def generate_full(**argv):

 filename = 'full.h5'

 kb = 1.380641e-23 #J/K
 hbar = 6.62607011e-34 #Js

 ftol=1e-10
 stol=1e6

 print(' ')
 print('Importing ' + filename + ' ... ',end= '')
 data = dd.io.load(filename)
 print(' done')
 print(' ')
 T0 = data['T']

 print('Temperature: ' + str(T0) + ' K')

 print(' ')

 print('Eliminating zero-frequency modes ...')

 w = data['f']*2*np.pi
 index = (w>ftol).nonzero()[0]
 exclude = (w<=ftol).nonzero()[0]
 w = w[index]

 print('Eliminated ',end='')
 print(len(exclude),end='')
 print(' modes with tolerance of ',end='')
 print(ftol,end='')
 print(' 1/s')
 print('... done')
 print(' ')


 print('Computing heat capacity ... ',end= '')
 eta = w*hbar/T0/kb/2
 C = kb*np.power(eta,2)*np.power(np.sinh(eta),-2) #J/K
 print('... done')
 print(' ')

 print('Computing sigmas ... ',end= '')
 factor = data['alpha']
 v = data['v'][index]
 sigma = np.einsum('u,ui->ui',C,v)/factor

 print('... done')
 print(' ')

 print('Simmetrizing scattering matrix A ...',end= '')
 A = data['A']
 A = np.delete(A,exclude,0)
 A = np.delete(A,exclude,1)
 A = 0.5*A + 0.5*A.T
 nm = A.shape[0]
 

 print('Computing W ...',end= '')
 W = np.einsum('u,uq,q->uq',w,A,w)/kb/T0/T0*hbar*hbar/factor

 print('... done')
 print(' ')

 print('Making W energy conserving ...',end= '')
 energy_conserving(W)
 #energy_conserving_square(W)
 print('... done')
 print(' ')
 print('Computing kappa bulk ...')


 print(' done')
 print(' ')

 kappa = compute_kappa(W,sigma)


 print('... done')
 print(' ')

 Wod = np.empty_like(W)
 for i in range(nm):
  for j in range(0,i):
   Wod[i,j] = -W[i,j]
 for i in range(nm):
   Wod[i,i] = 0

 tc = C/sum(C)

 a = np.diag(W)
 F = np.einsum('i,ij->ij',1/a,sigma)

 kappa_initial = data['kappa'] 


 data = {'temp':tc,'B':Wod,'F':F,'G':sigma,'kappa':kappa,'scale':1/a,'ac':np.zeros(nm)}

 return data








