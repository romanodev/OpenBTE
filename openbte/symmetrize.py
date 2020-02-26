import numpy as np
import pickle


def main() :
 W = np.load('W')


 #Make W energy conserving
 print('Make W energy conserving...')
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
 print('Row check:' + str(r/bottom))

 c = np.sum(np.absolute(np.einsum('ij->i',W)))
 print('Column Check:' + str(c/bottom))

 print('.... Done.')


sigma = np.load('sigma')

invW = np.linalg.inv(W)
tc = np.load('tc')

kappa = np.einsum('ui,uq,qj->ij',sigma,invW,sigma)

print(kappa)

data = {'tc':tc,'sigma':sigma,'W':W,'kappa':kappa}

pickle.dump(data,open('material.p','wb'))

if __name__ == "__main__":
  main()
