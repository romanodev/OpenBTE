import sys
import numpy as np
from scipy.sparse.linalg import lsqr 
from scipy.sparse.linalg import inv as sinv
from scipy.linalg import inv as inv
from scipy.sparse import csc_matrix
import scipy
from .utils import *
import time

def energy_conserving(W):


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

 return W

def full(**argv):

 filename = argv.setdefault('input_filename','full.npz')

 print(' ')
 print('Importing ' + filename + ' ... ',end= '')
 data = load_data('full')
 print(' done')
 print(' ')

 print('Computing sigmas ... ',end= '')
 v = data['v']
 C = data['C']
 sigma = np.einsum('u,ui->ui',C,v)

 if np.linalg.norm(sigma,axis=0)[2] == 0.0:
     dim = 2
 else:
     dim = 3


 print('... done')
 print(' ')

 print('Simmetrizing scattering matrix A ...',end= '')
 W = data['W']

 #Cut zero velocity modes -----
 tmp = np.linalg.norm(sigma,axis=1) 
 index = (tmp>0).nonzero()[0]
 exclude = (tmp==0).nonzero()[0]
 sigma = sigma[index]
 W = np.delete(W,exclude,0)
 W = np.delete(W,exclude,1)
 tc = C[index]/sum(C[index])
 print(np.shape(W))
 #-----------------------------


 W = 0.5*W + 0.5*W.T
 nm = W.shape[0]

 print('Making W energy conserving ...',end= '')
 W = energy_conserving(W)
 print('... done')
 print(' ')
 print('Computing kappa bulk ...')

 #kappa = np.einsum('ui,uq,qj->ij',sigma,np.linalg.pinv(W),sigma)/data['alpha']
 
 k_xx  = compute_kappa(W*data['alpha'],sigma[:,0])
 k_yy  = compute_kappa(W*data['alpha'],sigma[:,1])
 kappa = np.array([[k_xx,0],[0,k_yy]])


 print(kappa)
 print(' done')
 print(' ')
 #----------

 data = {'tc':tc,'W':W,'sigma':sigma[:,:dim],'kappa':kappa[:dim,:dim],'model':[10],'alpha':data['alpha']}

 return data








