from sqlitedict import SqliteDict
import pickle5 as pickle
from multiprocessing import Array,Value,Process
import multiprocessing
import numpy as np
import scipy.sparse.linalg as spla
from typing import Callable,Tuple
import os
from os.path import exists
from jax import numpy as jnp
import scipy
from scipy.ndimage import rotate


def solver(LL,P,x0,callback = lambda x:x,verbose=False,maxiter=200,early_termination=False,tol=1e-4,inplace=True,filename = None):
  """A wrapper to GMRES solver. It allows to set an early termination criteria based on user-provided functions"""

  size = len(P)
  
  L   = spla.LinearOperator((size,size),lambda x:LL(x))

  values = callback(x0)

  values = [1.0 for _ in values]

  X_final = np.zeros(size)

  call_count = [0]
  def callback_wrapper(x):

      call_count[0] +=1
      value = callback(x)

      #write to file--     
      if not filename == None:
            with  open(filename, "a")  as f:
                for v in value:
                 f.write(str(v))
                f.write("\n") 


      value_norm = [np.linalg.norm(v) for v in value]

      error = np.array([1 if values[i] == 0 else abs((value_norm[i]-values[i])/value_norm[i])  for i in range(len(values))])
    
      if verbose:
          print('Iter: ',call_count[0],' Error: ',error,' value ',value_norm)

      #Check error--
      X_final[:] = x
      values[:] = value_norm
      if (np.all(error < tol) or call_count[0] > maxiter) and early_termination:
         raise Exception
      #------------

  try:

      if early_termination:
          tol_gmres = 1e-24 #so that it does not interface
      else:    
          tol_gmres = tol
      out = scipy.sparse.linalg.lgmres(L,P,x0=x0,tol=tol_gmres,callback=callback_wrapper,maxiter=maxiter)

  except Exception as err:
      print(err)
      pass

  return callback(X_final),(X_final,call_count[0])

def load_rta(material,source='database'):

   from openbte.objects import MaterialRTA
   
   return MaterialRTA(*load(material,source=source).values())

def load_full(material,source='database'):

   from openbte.objects import MaterialFull
  

   return MaterialFull(*load(material,source=source).values())

def compute_kappa(W,factor,sigma,suppression = []):
 """ Compute effective thermal conductivity tensor given the scattering operator"""

 M = spla.LinearOperator(W.shape,lambda x: x/np.sqrt(np.diag(W)))

 def get_kappa_scipy(i):

    b  = sigma[:,i]
    x0 = b/np.diag(W)

    x  = spla.cg(W,b,M=M,x0=x0,atol=1e-19)[0]

    if not len(suppression) == 0:
       b = np.einsum('u,uv->v',b,suppression) 

    return np.dot(x,b),x

 k_xx,Sx  = get_kappa_scipy(0)
 k_yy,Sy  = get_kappa_scipy(1)
 kappa = np.array([[k_xx,0],[0,k_yy]])

 G = np.vstack((Sx,Sy)).T

 return kappa/factor,G


def run(target          : Callable, \
        n_tasks         : Tuple,\
        shared_variables = {}):
    """Run a parallel job"""

    from openbte.objects import SharedMemory
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-np', help='Number of processors',default=multiprocessing.cpu_count())
    n_process = int(parser.parse_args(args=[]).np)

    #Init shared variables
    sh = SharedMemory(n_process)
    for name,value in shared_variables.items():
        sh[name] = value
    #--------------------
    indices =  [np.array_split(np.arange(n_task),n_process) for n_task in n_tasks]

    processes = [ Process(name='process_' + str(k),target=target, args=(sh,inds)) for k,inds in enumerate(zip(*indices))]

    for p in processes:
        if not p == None:
         p.start()
    for p in processes:
        if not p == None:
         p.join()
    return sh      

def save(filename,data):
      """Save dictionary"""

      with SqliteDict(filename + '.db',autocommit=True,encode=pickle.dumps, decode=pickle.loads) as db:
        for key,value in data.items():
          db[key] = value

def sparse_dense_product(i,j,data,X):
     '''
     This solves B_ucc' X_uc' -> A_uc

     B_ucc' : sparse in cc' and dense in u. Data is its vectorized data COO description
     X      : dense matrix
     '''

     tmp = np.zeros_like(X)
     np.add.at(tmp.T,i,data.T * X.T[j])

     return tmp


def sparse_dense_product_jax(i,j,data,X):
     '''
     This solves B_ucc' X_uc' -> A_uc

     B_ucc' : sparse in cc' and dense in u. Data is its vectorized data COO descrition
     X      : dense matrix
     '''
     return jnp.zeros_like(X.T).at[i].add(data.T * X.T[j]).T


     

def load_readonly(filename):
 """Load dictionary"""

 return SqliteDict(filename + '.db',encode=pickle.dumps, decode=pickle.loads)


def load(filename,source='database'):
 """Load dictionary"""

 if source == 'database':
     prefix = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/openbte/materials/'
 else:
     prefix = './'

 full_name = prefix + filename + '.db'

 if exists(full_name):
  data = {}
  with SqliteDict(full_name,autocommit=True,encode=pickle.dumps, decode=pickle.loads) as db:
     for key,value in db.items():
        data[key] = value
 else:
    print('{} does not exist'.format(full_name))
    quit()


 return data       
