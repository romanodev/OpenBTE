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
import functools
import scipy
from scipy.ndimage import rotate
#import joblib
import subprocess,os
import __main__ as main

def _regularize(phi):

    phi[phi<0]       += 2*np.pi
    phi[phi>2*np.pi] -= 2*np.pi

    return phi

def _n(phi):

   a = np.sin(phi)
   b = np.sin(phi)
   if phi.ndim == 1:
    return np.stack((np.sin(phi),np.cos(phi))).T
   else:
    return np.swapaxes(np.stack((np.sin(phi),np.cos(phi))).T,0,1)


def generate_mesh_2D(filename='mesh'):
    """Generate 2D mesh"""

    with open(os.devnull, 'w') as devnull:
          output = subprocess.check_output(("gmsh -format msh2 -2 " + filename + ".geo -o " + filename + ".msh").split(), stderr=devnull)


def split(S,B,normalize=True):

    phi_B   = _regularize(np.arctan2(B[:,0],B[:,1]))
    phi_S   = _regularize(np.arctan2(S[...,0],S[...,1]))
    Dphi    = phi_S[...,1]-phi_S[...,0]
    if Dphi.ndim == 1:
        Dphi = Dphi[0]

    #Compute magnitude---------------
    mag_s = np.linalg.norm(S,axis=-1)
    mag_B = np.linalg.norm(B,axis=-1)
    mag   = np.einsum('...,k->...k',mag_s,mag_B)
    #---------------------------------

    delta = 1e-4

    DeltaPhi = _regularize(phi_S[...,np.newaxis] - phi_B[np.newaxis,:])

    #This does a rotation automatically
    ap = np.einsum('...,...i,ki->...k',mag_s,_n(phi_S + Dphi/2 - np.pi/2),B)
    am = np.einsum('...,...i,ki->...k',mag_s,_n(phi_S - Dphi/2 - np.pi/2),B)
    a  = ap-am
    gp = a.clip(min=0)
    gm = a.clip(max=0)

    #Correction at n + np.pi/2
    r  = np.logical_and(DeltaPhi >= np.pi/2-Dphi/2+delta,DeltaPhi <= np.pi/2 + Dphi/2-delta)
    gp[r.nonzero()] =  (mag  - am)[r.nonzero()]
    gm[r.nonzero()] =  (ap   - mag)[r.nonzero()]

    #Correction at n - np.pi/2
    r  = np.logical_and(DeltaPhi >= 3*np.pi/2-Dphi/2+delta,DeltaPhi <= 3*np.pi/2 + Dphi/2-delta)
    gp[r.nonzero()] =   (ap  + mag)[r.nonzero()]
    gm[r.nonzero()] =  -(mag + am)[r.nonzero()]

    factor = 1/Dphi if normalize else 1


    return gm*factor,\
           gp*factor

    


def solver(LL,P,x0,callback = lambda x:x,verbose=False,maxiter=200,early_termination=False,tol=1e-4,inplace=True,filename = None,MM=None):
  """A wrapper to GMRES solver. It allows to set an early termination criteria based on user-provided functions"""

  size = len(P)
  
  L   = spla.LinearOperator((size,size),lambda x:LL(x))

  if not MM == None:
   M   = spla.LinearOperator((size,size),lambda x:MM(x))
  else:
   M = None   


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
      out = scipy.sparse.linalg.lgmres(L,P,x0=x0,tol=tol_gmres,callback=callback_wrapper,maxiter=maxiter,M=M)

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
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('-np', help='Number of processors',default=multiprocessing.cpu_count())
    if not hasattr(main, '__file__'):
     args = parser.parse_args(args='')
    else: 
     args = parser.parse_args()

    n_process = int(args.np)
    print('n_process: ',n_process)

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

def fast_interpolation(fine,coarse,bound=False,scale='linear') :
 """Vectorized Interpolation"""

 if scale == 'log':
   fine    = np.log10(fine)
   coarse  = np.log10(coarse)
 #--------------

 m2 = np.argmax(coarse >= fine[:,np.newaxis],axis=1)
 m1 = m2-1
 a2 = (fine-coarse[m1])/(coarse[m2]-coarse[m1])
 a1 = 1-a2

 if bound == 'periodic':
  Delta = coarse[-1]-coarse[-2]  
  a = np.where(m2==0)[0] #this can be either regions
  m1[a] = len(coarse) -1
  m2[a] = 0
  fine[fine < Delta/2] += 2*np.pi 
  a2[a] = (fine[a] - coarse[-1])/ Delta
  a1 = 1-a2

 if bound == 'extent':

   #Small values
   al = np.where(fine<coarse[0])[0] 
   m2[al] = 1; 
   m1[al] = 0;
   a2[al] = (fine[al]-coarse[0])/ (coarse[1]-coarse[0])
   a1[al] = 1-a2[al]


   #Large values
   ar = np.where(fine>coarse[-1])[0]
   m2[ar] = len(coarse)-1; 
   m1[ar] = len(coarse)-2;
   a2[ar] = (fine[ar]-coarse[-2])/ (coarse[-1]-coarse[-2])
   a1[ar] = 1-a2[ar]

 return a1,a2,m1,m2


def compute_polar(mfp_bulk):
     """Covert from real to angular space"""
     phi_bulk = np.array([np.arctan2(m[0],m[1]) for m in mfp_bulk])
     phi_bulk[np.where(phi_bulk < 0) ] = 2*np.pi + phi_bulk[np.where(phi_bulk <0)]
     r = np.linalg.norm(mfp_bulk[:,:2],axis=1) #absolute values of the projection
     return r,phi_bulk 

def compute_spherical(mfp_bulk):
 """Covert from real to spherical space"""
 r = np.linalg.norm(mfp_bulk,axis=1) #absolute values of the projection
 phi_bulk = np.array([np.arctan2(m[0],m[1]) for m in mfp_bulk])
 phi_bulk[np.where(phi_bulk < 0) ] = 2*np.pi + phi_bulk[np.where(phi_bulk <0)]
 theta_bulk = np.array([np.arccos((m/r[k])[2]) for k,m in enumerate(mfp_bulk)])

 return r,phi_bulk,theta_bulk


'''
def cache(func):
    """Simple cache"""
    #from openbte import registry

    @functools.wraps(func)
    def wrapper_decorator(*args):
        #---------
        hash_obj = joblib.hash(args)

        if not hasattr(func,'hashes'):
           func.hashes  = {} 

        if not hash_obj in func.hashes.keys():
           value = func(*args)
           func.hashes[hash_obj] = value
        else:   
           value = func.hashes[hash_obj] 
        return value

    return wrapper_decorator
'''
