import numpy as np
import scipy.sparse as sp

class MultiSolver(object):

  def __init__(self,row,col,A,A0,**argv):


   if argv.setdefault('cpu'):  
    (nbatch,d) = np.shape(A0)
    self.scale = np.zeros(nbatch)
    self.lu = {}
    for i in range(nbatch):
      self.scale[i] = np.max(A[i])  
      if self.scale[i] == 0:
        self.scale[i] = 1
      S1 = sp.csc_matrix((A[i]/self.scale[i],(row,col)),shape=(d,d),dtype=np.float64)
      S = S1 + sp.diags(A0[i]/self.scale[i],format='csc')
      lu = sp.linalg.splu(S,permc_spec='COLAMD')
      self.lu.update({i:lu})


  def solve(self,B):    

   x = np.zeros_like(B) 
   for i in self.lu.keys():
    x[i] = self.lu[i].solve(B[i]/self.scale[i])

   return x
      


