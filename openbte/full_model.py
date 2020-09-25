import sys
import numpy as np
from scipy.sparse.linalg import lsqr 
from scipy.sparse.linalg import inv as sinv
from scipy.linalg import inv as inv
from scipy.sparse import csc_matrix
import scipy
from .utils import *
#import deepdish as dd




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

def generate_full(**argv):

 filename = 'full.npz'

 print(' ')
 print('Importing ' + filename + ' ... ',end= '')
 #data = np.load(filename,allow_pickle=True)
 data = load_data('full')
 #data = dd.io.load(filename)
 print(' done')
 print(' ')

 print('Computing sigmas ... ',end= '')
 v = data['v']
 C = data['C']
 sigma = np.einsum('u,ui->ui',C,v)

 print('... done')
 print(' ')

 print('Simmetrizing scattering matrix A ...',end= '')
 W = data['W']
 W = 0.5*W + 0.5*W.T
 nm = W.shape[0]

 print('Making W energy conserving ...',end= '')
 W = energy_conserving(W)
 print('... done')
 print(' ')
 print('Computing kappa bulk ...')
 kappa = np.einsum('ui,uq,qj->ij',sigma,np.linalg.pinv(W),sigma)/data['alpha']
 print(kappa)
 print(' done')
 print(' ')
 #----------
 #test
 #Adinv = np.diag(1/np.diag(W))
 #Aod = W - np.diag(np.diag(W))
 #Add = np.einsum('ij,i->ij',Aod,1/np.diag(W))
 #x0 = np.einsum('lk,kj->lj',Adinv,sigma)
 #x_old = np.zeros_like(x0)
 #for n in range(40):
 #    print(n) 
 #    x = x0 - np.einsum('kl,lj->kj',Add,x_old)
 #    print(np.einsum('li,lj->ij',sigma,x)/data['alpha'])
 #    x_old = x.copy()
 #quit()
 #----------

 #postprocessing----
 B = -np.einsum('i,ij->ij',1/np.diag(W),W-np.diag(np.diag(W)))
 Wod = -(W-np.diag(np.diag(W)))
 Wd = np.tril(Wod)
 B = np.empty_like(Wd)
 B[Wd.nonzero()] = Wd[Wd.nonzero()]
 #------------------

 tc = C/sum(C)

 F = np.einsum('i,ij->ij',1/np.diag(W),sigma)

 data = {'tc':tc,'VMFP':F[:,:2],'sigma':sigma[:,:2],'kappa':kappa,'B':B,'scale':1/np.diag(W),'model':[10],'alpha':data['alpha']}

 return data








