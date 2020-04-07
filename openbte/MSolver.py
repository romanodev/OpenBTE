import numpy as np
import scipy.sparse as sp
import time

class MultiSolver(object):

  def __init__(self,row,col,A,d,**argv):
 
    (n_batch,_) = A.shape
    self.tt = np.float64   
    self.lu = {}
    for i in range(n_batch):
      S = sp.csc_matrix((A[i],(row,col)),shape=(d,d),dtype=self.tt)
      lu = sp.linalg.splu(S,permc_spec='COLAMD')
      self.lu.update({i:lu})
    
  def solve(self,B,common = []):

   B = B.astype(self.tt)
   x = np.zeros_like(B) 
   for i in self.lu.keys():
     RHS = B[i]
     if len(common) > 0:
       RHS += common  
     x[i] = self.lu[i].solve(RHS)
   return x


