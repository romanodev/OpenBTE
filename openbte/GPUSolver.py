# ### Interface cuSOLVER PyCUDA

from __future__ import print_function
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import scipy.sparse as sp
import ctypes
from scipy.sparse import rand
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
import time
import numpy.linalg as la
import scipy.sparse.linalg as sla
import sys
#import sparseqr
import scikits.umfpack as um


# cuSparse
_libcusparse = ctypes.cdll.LoadLibrary('libcusparse.so')
_libcusparse.cusparseCreate.restype = int
_libcusparse.cusparseCreate.argtypes = [ctypes.c_void_p]

_libcusparse.cusparseDestroy.restype = int
_libcusparse.cusparseDestroy.argtypes = [ctypes.c_void_p]

_libcusparse.cusparseCreateMatDescr.restype = int
_libcusparse.cusparseCreateMatDescr.argtypes = [ctypes.c_void_p]

_libcusparse.cusparseDestroyMatDescr.restype = int
_libcusparse.cusparseDestroyMatDescr.argtypes = [ctypes.c_void_p]

# cuSOLVER
_libcusolver = ctypes.cdll.LoadLibrary('libcusolver.so')

_libcusolver.cusolverSpCreate.restype = int
_libcusolver.cusolverSpCreate.argtypes = [ctypes.c_void_p]

_libcusolver.cusolverSpDestroy.restype = int
_libcusolver.cusolverSpDestroy.argtypes = [ctypes.c_void_p]


_libcusolver.cusolverSpXcsrqrAnalysisBatched.restype = int
_libcusolver.cusolverSpXcsrqrAnalysisBatched.argtypes= [ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_void_p(),
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p() #info
                                            ]


_libcusolver.cusolverSpDcsrqrBufferInfoBatched.restype = int
_libcusolver.cusolverSpDcsrqrBufferInfoBatched.argtypes= [ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_void_p, #info
                                            ctypes.c_void_p,
                                            ctypes.c_void_p
                                            ]


_libcusolver.cusolverSpDcsrqrsvBatched.restype = int
_libcusolver.cusolverSpDcsrqrsvBatched.argtypes= [ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p]
_cusp_handle = ctypes.c_void_p()
status = _libcusparse.cusparseCreate(ctypes.byref(_cusp_handle))
assert(status == 0)
cusp_handle = _cusp_handle.value

_cuso_handle = ctypes.c_void_p()
status = _libcusolver.cusolverSpCreate(ctypes.byref(_cuso_handle))
assert(status == 0)
cuso_handle = _cuso_handle.value

tt = np.float64

def SolveCPU(row,col,A,B,A0,B0):

    #A = A.astype(tt)
    #B = B.astype(tt)
    (nbatch,nnz) = np.shape(A)
    (_,d) = np.shape(B)
    xs = []
    umfpack = um.UmfpackContext()
    xs = []
    init = False
    for i in range(nbatch):
      S1 = sp.csc_matrix((A[i],(row,col)),shape=(d,d),dtype=np.float64)
      S = S1 + sp.diags(A0[i],format='csc') + sp.eye(d,format='csc') 
      if init==False:
       umfpack.symbolic(S)
       init = True
      umfpack.numeric(S)
      x = umfpack.solve( um.UMFPACK_A,S,B[i]+B0)
      xs.append(x)
    return np.array(xs)

def get_lu(row,col,A,A0,d):


    (nbatch,_) = np.shape(A)
    lu = {}
    for i in range(nbatch):
      S1 = sp.csc_matrix((A[i],(row,col)),shape=(d,d),dtype=np.float64)
      S = S1 + sp.diags(A0[i],format='csc') + sp.eye(d,format='csc') 
      lu.update({i:sp.linalg.splu(S)})

    return lu

def solve_from_lu(lu,B,B0):

  x = np.zeros_like(B) 
  for i in lu.keys():
   x[i] = lu[i].solve(B[i]+B0)

  return x

#def get_lu_2(row,col,A,N):

#    A = A.astype(tt)
#    (nbatch,nnz) = np.shape(A)

#    umfpack = um.UmfpackContext()
#    S = sp.csr_matrix((A[0],(row,col)),shape=(N,N),dtype=np.float64)
#    umfpack.symbolic(S)

    
#    t1 = time.time()
#    xs = []
#    for i in range(nbatch):
#      S = sp.csr_matrix((A[i],(row,col)),shape=(N,N),dtype=np.float64)
#      umfpack.numeric(S)

#    return np.array(xs)

   
#def solve_from_lu(lu,B):

#  x = np.zeros_like(B) 
#  for i in lu.keys():
#   x[i] = lu[i].solve(B[i])
#  return x

#def get_lu(row,col,A,N):

#    t1 = time.time()
#    (nbatch,nnz) = np.shape(A)
#    lu = {}

#    for i in range(nbatch):
#      S = sp.csc_matrix((A[i],(row,col)),shape=(N,N),dtype=float)
#3      lu.update({i:sp.linalg.splu(S)})
#    return lu


def SolveGPU(row,col,A,B):

   (nbatch,nnz) = np.shape(A)
   (_,d) = np.shape(B)
   Acsr = sp.csr_matrix( (np.ones(len(col)),(row,col)), shape=(d,d),dtype=tt).sorted_indices()
   dcsrIndPtr = gpuarray.to_gpu(Acsr.indptr)
   dcsrColInd = gpuarray.to_gpu(Acsr.indices)

   n = ctypes.c_int(d) 
   m = ctypes.c_int(d)  
   nbatch = ctypes.c_int(nbatch)
   nnz = ctypes.c_int(nnz)


   # create cusparse handle
   descrA = ctypes.c_void_p()
   info = ctypes.c_void_p()
   _libcusolver.cusolverSpCreateCsrqrInfo(ctypes.byref(info))

   # create MatDescriptor
   status = _libcusparse.cusparseCreateMatDescr(ctypes.byref(descrA))
   assert(status == 0)


   _libcusolver.cusolverSpXcsrqrAnalysisBatched(cuso_handle,
                                 n,
                                 m,
                                 nnz,
                                 descrA,
                                 int(dcsrIndPtr.gpudata),
                                 int(dcsrColInd.gpudata),
                                 info)


   global_data = []
   for i in range(nbatch.value):
     Acsr = sp.csr_matrix((A[i,:],(row,col)),shape=(n.value,m.value),dtype=tt).sorted_indices()
     global_data.append(Acsr.data)
   data = np.ascontiguousarray(global_data,dtype=tt)
  
   dcsrVal = gpuarray.to_gpu(data.astype(tt))


   B = np.ascontiguousarray(B.flatten(),dtype=tt)  
   b = gpuarray.to_gpu(B.astype(tt))
   dx = pycuda.gpuarray.empty_like(b,dtype=tt)

   b1 = ctypes.c_int()
   b2 = ctypes.c_int()

   _libcusolver.cusolverSpDcsrqrBufferInfoBatched(cuso_handle,
                           n,
                           m,
                           nnz,
                           descrA,
                           int(dcsrVal.gpudata),
                           int(dcsrIndPtr.gpudata),
                           int(dcsrColInd.gpudata),
                           nbatch,
                           info,
                           ctypes.byref(b1),
                           ctypes.byref(b2)
                           );

   
   print(b2.value/1024/1024)
   print(b1.value/1024/1024)
   w_buffer = gpuarray.zeros(b2.value, dtype=tt)
   
   t1 = time.time() 
   res = _libcusolver.cusolverSpDcsrqrsvBatched(cuso_handle,
                                 n,
                                 m,
                                 nnz,
                                 descrA,
                                 int(dcsrVal.gpudata),
                                 int(dcsrIndPtr.gpudata),
                                 int(dcsrColInd.gpudata),
                                 int(b.gpudata),
                                 int(dx.gpudata),
                                 nbatch,
                                 info,
                                 int(w_buffer.gpudata))
   x= dx.get()
   #print(gpuarray.min(dx),gpuarray.max(dx))
   print(time.time()-t1) 
   
   status = _libcusolver.cusolverSpDestroy(cuso_handle)
   assert(status == 0)
   status = _libcusparse.cusparseDestroy(cusp_handle)
   assert(status == 0)

 
   return dx.get().reshape((nbatch.value,n.value))


class MSparse(object):

  def __init__(self,a1,a2,d,k,reordering=False,new=False):

   self.new = False
   self.reordering = reordering
   if new:
    self.new = True
    self.dcsrIndPtr = gpuarray.to_gpu(a1)
    self.dcsrColInd = gpuarray.to_gpu(a2)
    self.n = ctypes.c_int(d) 
    self.m = ctypes.c_int(d)  
    self.nbatch = ctypes.c_int(k)
    self.nnz = ctypes.c_int(len(a2))
   else:
    #Experimental reordering
    if self.reordering:
     A = sp.coo_matrix( (np.ones(len(a2)),(a1,a2)), shape=(d,d),dtype=float)
     _, _, E, rank = sparseqr.qr(A)
     self.P = sparseqr.permutation_vector_to_matrix(E) #coo
    
     Acsr = (A*self.P.tocsr()).sorted_indices()
    else:
     Acsr = sp.csr_matrix( (np.ones(len(a2)),(a1,a2)), shape=(d,d),dtype=np.float64)
   
    Acsr.indptr = Acsr.indptr.astype(dtype=np.int32)
    Acsr.indices = Acsr.indices.astype(dtype=np.int32)
    self.dcsrIndPtr = gpuarray.to_gpu(Acsr.indptr)
    self.dcsrColInd = gpuarray.to_gpu(Acsr.indices)
    self.n = ctypes.c_int(d) 
    self.m = ctypes.c_int(d)  
    self.nbatch = ctypes.c_int(k)
    self.row = a1
    self.col = a2
    self.nnz = ctypes.c_int(len(a2))



   _cusp_handle = ctypes.c_void_p()
   status = _libcusparse.cusparseCreate(ctypes.byref(_cusp_handle))
   assert(status == 0)
   self.cusp_handle = _cusp_handle.value

   _cuso_handle = ctypes.c_void_p()
   status = _libcusolver.cusolverSpCreate(ctypes.byref(_cuso_handle))
   assert(status == 0)
   self.cuso_handle = _cuso_handle.value

   # create cusparse handle
   self.descrA = ctypes.c_void_p()
   self.info = ctypes.c_void_p()
   _libcusolver.cusolverSpCreateCsrqrInfo(ctypes.byref(info))

   # create MatDescriptor
   status = _libcusparse.cusparseCreateMatDescr(ctypes.byref(descrA))
   assert(status == 0)


   _libcusolver.cusolverSpXcsrqrAnalysisBatched(self.cuso_handle,
                                 self.n,
                                 self.m,
                                 self.nnz,
                                 self.descrA,
                                 int(self.dcsrIndPtr.gpudata),
                                 int(self.dcsrColInd.gpudata),
                                 self.info)

   

    # Destroy handles
    #status = _libcusolver.cusolverSpDestroy(self.cuso_handle)
    #assert(status == 0)
    #status = _libcusparse.cusparseDestroy(self.cusp_handle)
    #assert(status == 0)
  def add_LHS(self,data):

   self.data = data.astype(np.float64)
   if self.new:
    self.dcsrVal = gpuarray.to_gpu(data.astype(np.float64)).ravel()
   else:
    global_data = []
    for n in range(self.nbatch.value):
     Acsr = sp.csr_matrix((self.data[n],(self.row,self.col)),shape=(self.n.value,self.m.value),dtype=np.float64).sorted_indices()
     global_data.append(Acsr.data)
    self.dcsrVal = gpuarray.to_gpu(np.array(global_data,dtype=np.float64))

   b1 = ctypes.c_int()
   b2 = ctypes.c_int()

   _libcusolver.cusolverSpDcsrqrBufferInfoBatched(self.cuso_handle,
                           self.n,
                           self.m,
                           self.nnz,
                           self.descrA,
                           int(self.dcsrVal.gpudata),
                           int(self.dcsrIndPtr.gpudata),
                           int(self.dcsrColInd.gpudata),
                           self.nbatch,
                           self.info,
                           ctypes.byref(b1),
                           ctypes.byref(b2)
                           );

   
   print(b2.value/1024/1024)
   print(b1.value/1024/1024)
   self.w_buffer = gpuarray.zeros(b2.value, dtype=self.dcsrVal.dtype) 

  def solve(self,b):
    #b = np.array(b)
    b = b.astype(np.float64)
    self.b = b

    #Solve scipy
    xs = []
    for i in range(self.nbatch.value):
      #S = sp.csr_matrix((self.data[i]/max(self.data[i]),self.dcsrColInd.get(),self.dcsrIndPtr.get()),shape=(self.n.value,self.n.value),dtype=float)
      #x = sp.linalg.spsolve(S,self.b[i]/max(self.data[i]))
      #S = sp.csr_matrix((self.data[i],self.dcsrColInd.get(),self.dcsrIndPtr.get()),shape=(self.n.value,self.n.value),dtype=np.float64)
      S = sp.csr_matrix((self.data[i],(self.row,self.col)),shape=(self.n.value,self.n.value),dtype=np.float64)
      x = sp.linalg.spsolve(S,self.b[i])
      xs.append(x)
    #print(np.min(xs),np.max(xs))
    #  print(min(x),max(x))
    #  print(x)
    #quit()


    transfer = False
    if isinstance(b, np.ndarray):
     b = gpuarray.to_gpu(b.astype(np.float64)).ravel()
     transfer = True 
    dx = pycuda.gpuarray.zeros_like(b,np.float64) 
    #print(gpuarray.min(b),gpuarray.max(b))
    
    #print(self.n) 
    #print(self.m) 
    #print(self.nbatch) 
    #print(self.nnz) 
    #print(len(self.dcsrVal)) 
    #print(len(self.dcsrColInd)) 
    #print(len(self.dcsrIndPtr)) 
    t4 = time.time()
    res = _libcusolver.cusolverSpDcsrqrsvBatched(self.cuso_handle,
                                 self.n,
                                 self.m,
                                 self.nnz,
                                 self.descrA,
                                 int(self.dcsrVal.gpudata),
                                 int(self.dcsrIndPtr.gpudata),
                                 int(self.dcsrColInd.gpudata),
                                 int(b.gpudata),
                                 int(dx.gpudata),
                                 self.nbatch,
                                 self.info,
                                 int(self.w_buffer.gpudata))
   
    # Destroy handles
    #status = _libcusolver.cusolverSpDestroy(self.cuso_handle)
    #assert(status == 0)
    #status = _libcusparse.cusparseDestroy(self.cusp_handle)
    #assert(status == 0)
    print(gpuarray.min(dx),gpuarray.max(dx))

    t5 = time.time()
    #print(t5-t4)
    #print(gpuarray.min(dx),gpuarray.max(dx))   
    xx= dx.get()
    
    aa = xx.reshape((self.nbatch.value,self.n.value))
    print(np.allclose(aa,xs,atol=1e-4,rtol=1e-4))
    #print(np.min(aa,axis=1))
    #print(np.min(aa,axis=0))

    #if transfer:
    # dx = dx.get()  # Get result as numpy array

    #if self.reordering:
    # dx = np.array([self.P.dot(dx[n]) for n in range(self.nbatch.value)])


    return aa

    # Return result

  def free_memory(self):
 
    # Destroy handles
    status = _libcusolver.cusolverSpDestroy(self.cuso_handle)
    assert(status == 0)
    status = _libcusparse.cusparseDestroy(self.cusp_handle)
    assert(status == 0)
    status = _libcusparse.cusparseDestroyMatDescr(self.descrA)
    assert(status == 0)



# Test
if __name__ == '__main__':

  NN = [764]
  BB = [1000]
  tgpu = np.zeros((3,3))
  tcpu = np.zeros((3,3))
  for n,N in enumerate(NN):
   for b,nbatch in enumerate(BB):

    #N = 100
    #nbatch = 1000
    m = sp.diags([1, -2, 1,4,5], [-2,-1, 0, 1,2], shape=(N, N),format='coo')
    A = np.random.random_sample((nbatch,m.nnz))
    B = np.random.random_sample((nbatch,N))



    Xgpu = SolveGPU(m.row,m.col,A,B)
    Xcpu = SolveCPU(m.row,m.col,A,B)
    print(np.allclose(Xgpu,Xcpu,rtol=1e-6,atol=1e-6))

    #m = MSparse(m.row,m.col,N,nbatch)
    #m.add_LHS(A)
    #X = m.solve(B)
    #m.free_memory()

    #-----------------------------------
    #Acsr = sp.csr_matrix((np.arange(len(m.row)),(m.row,m.col)),shape=(N,N),dtype=np.int32)
    #rot = Acsr.data
    #indptr  = Acsr.indptr
    #indices = Acsr.indices
    #m = MSparse(indptr,indices,N,nbatch,new=True)
    
    #A = np.array([ A[n,rot]  for n in range(nbatch)])
    #m.add_LHS(A)
    #X2 = m.solve(B)
    #print(np.allclose(X1,X2))
 
    #m.free_memory()

    #--------------------
    #quit()


    #tgpu[n,b] = t2-t1 
    #tcpu[n,b] = t3-t2

  #print(tgpu)
  #print(tcpu)










