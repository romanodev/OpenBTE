import numpy as np
import scipy.sparse as sp
import time

class MultiSolver(object):

  def __init__(self,row,col,A,d,M,**argv):

   self.tt = np.float64
   self.mode = argv.setdefault('mode','cpu')
   (self.n_batch,self.nnz) = np.shape(A)
   if self.mode == 'cpu':  
    #self.scale = np.zeros(nbatch)
    self.lu = {}
    for i in range(self.n_batch):
      #self.scale[i] = np.max(A[i])  
      #if self.scale[i] == 0:
      #  self.scale[i] = 1
      S = sp.csc_matrix((A[i],(row,col)),shape=(d,d),dtype=np.float64)
      lu = sp.linalg.splu(S,permc_spec='COLAMD')
      self.lu.update({i:lu})
   else: 
    self.A = A
    self.d = d
    self.row = row
    self.col = col
    self.M = M
    
    print('GPU!')
    import ctypes
    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit
    self.dx = pycuda.gpuarray.empty((self.n_batch,self.d),dtype=self.tt)
    #cuSOLVER
    self._libcusparse = ctypes.cdll.LoadLibrary('libcusparse.so')
    self._libcusparse.cusparseCreate.restype = int
    self._libcusparse.cusparseCreate.argtypes = [ctypes.c_void_p]

    self._libcusparse.cusparseDestroy.restype = int
    self._libcusparse.cusparseDestroy.argtypes = [ctypes.c_void_p]
    self._libcusparse.cusparseDestroyMatDescr.restype = int
    self._libcusparse.cusparseDestroyMatDescr.argtypes = [ctypes.c_void_p]

      # cuSOLVER
    self._libcusolver = ctypes.cdll.LoadLibrary('libcusolver.so')
    self._libcusolver.cusolverSpCreate.restype = int 
    self._libcusolver.cusolverSpCreate.argtypes = [ctypes.c_void_p]
    self._libcusolver.cusolverSpDestroy.restype = int
    self._libcusolver.cusolverSpDestroy.argtypes = [ctypes.c_void_p]
    self._libcusolver.cusolverSpXcsrqrAnalysisBatched.restype = int
    self._libcusolver.cusolverSpXcsrqrAnalysisBatched.argtypes= [ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_void_p(),
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p() #info
                                            ]
    #cusolverSpDcsrqrBufferInfoBatched
    self._libcusolver.cusolverSpDcsrqrBufferInfoBatched.restype = int
    self._libcusolver.cusolverSpDcsrqrBufferInfoBatched.argtypes= [ctypes.c_void_p,
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

    #cusolverSpDcsrqrsvBatched
    self._libcusolver.cusolverSpDcsrqrsvBatched.restype = int
    self._libcusolver.cusolverSpDcsrqrsvBatched.argtypes= [ctypes.c_void_p,
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

      #cusp_handle
    _cusp_handle = ctypes.c_void_p()
    status = self._libcusparse.cusparseCreate(ctypes.byref(_cusp_handle))
    assert(status == 0)
    self.cusp_handle = _cusp_handle.value

    #cuso_handle
    _cuso_handle = ctypes.c_void_p()
    status = self._libcusolver.cusolverSpCreate(ctypes.byref(_cuso_handle))
    assert(status == 0)
    self.cuso_handle = _cuso_handle.value
 
    # create MatDescriptor
    self._libcusparse.cusparseDestroyMatDescr.restype = int
    self._libcusparse.cusparseDestroyMatDescr.argtypes = [ctypes.c_void_p]
    self._libcusparse.cusparseCreateMatDescr.restype = int
    self._libcusparse.cusparseCreateMatDescr.argtypes = [ctypes.c_void_p]
    self.descrA = ctypes.c_void_p()
    status = self._libcusparse.cusparseCreateMatDescr(ctypes.byref(self.descrA))
    assert(status == 0)
 
    #info
    self.info = ctypes.c_void_p()
    status = self._libcusolver.cusolverSpCreateCsrqrInfo(ctypes.byref(self.info))
    assert(status==0)
    self._libcusolver.cusolverSpDestroyCsrqrInfo.argtypes = [ctypes.c_void_p]
    self._libcusolver.cusolverSpDestroyCsrqrInfo.restype = int

    

    #init
    self.n = ctypes.c_int(self.d)
    self.m = ctypes.c_int(self.d)
    b1 = ctypes.c_int()
    b2 = ctypes.c_int()
    self.n_batch = ctypes.c_int(self.n_batch)
    self.nnz = ctypes.c_int(self.nnz)

    Acsr = sp.csr_matrix( (np.ones(len(self.col)),(self.row,self.col)), shape=(self.d,self.d),dtype=self.tt).sorted_indices()
    self.dcsrIndPtr = gpuarray.to_gpu(Acsr.indptr)
    self.dcsrColInd = gpuarray.to_gpu(Acsr.indices)

    self._libcusolver.cusolverSpXcsrqrAnalysisBatched(self.cuso_handle,
                                 self.n,
                                 self.m,
                                 self.nnz,
                                 self.descrA,
                                 int(self.dcsrIndPtr.gpudata),
                                 int(self.dcsrColInd.gpudata),
                                 self.info)
     
    #get data---
    global_data = []
    for i in range(self.n_batch.value):
        Acsr = sp.csr_matrix((self.A[i,:],(self.row,self.col)),shape=(self.n.value,self.m.value),dtype=self.tt).sorted_indices()
        global_data +=list(Acsr.data)
    global_data = np.array(global_data,dtype=self.tt)
    data = np.ascontiguousarray(global_data,dtype=self.tt)
    self.dcsrVal = gpuarray.to_gpu(data)
      

    #get info
    self._libcusolver.cusolverSpDcsrqrBufferInfoBatched(self.cuso_handle,
                           self.n,
                           self.m,
                           self.nnz,
                           self.descrA,
                           int(self.dcsrVal.gpudata),
                           int(self.dcsrIndPtr.gpudata),
                           int(self.dcsrColInd.gpudata),
                           self.n_batch,
                           self.info,
                           ctypes.byref(b1),
                           ctypes.byref(b2)
                           );

    #  print(b2.value/1024/1024)
    #  print(b1.value/1024/1024)

    self.w_buffer = gpuarray.zeros(b2.value, dtype=self.tt)
    


  def solve(self,B):    
   B = B.astype(self.tt)
   if self.mode =='cpu': 
    x = np.zeros_like(B) 
    for i in self.lu.keys():
     x[i] = self.lu[i].solve(B[i])
    return x
   else: 
      import pycuda.gpuarray as gpuarray
      import pycuda.autoinit
      b = gpuarray.to_gpu(B.flatten().astype(self.tt))
      a = time.time()
      self._libcusolver.cusolverSpDcsrqrsvBatched(self.cuso_handle,
                                 self.n,
                                 self.m,
                                 self.nnz,
                                 self.descrA,
                                 int(self.dcsrVal.gpudata),
                                 int(self.dcsrIndPtr.gpudata),
                                 int(self.dcsrColInd.gpudata),
                                 int(b.gpudata),
                                 int(self.dx.gpudata),
                                 self.n_batch,
                                 self.info,
                                 int(self.w_buffer.gpudata))
      print(time.time() -a)                             
      
      return np.reshape(self.dx.get(),(self.n_batch.value,self.d))

  def release(self):

     if self.mode =='gpu':
      status = self._libcusolver.cusolverSpDestroy(self.cuso_handle)
      assert(status == 0)
      status = self._libcusparse.cusparseDestroy(self.cusp_handle)
      assert(status == 0)
      status = self._libcusparse.cusparseDestroyMatDescr(self.descrA)
      assert(status == 0)
      status = self._libcusolver.cusolverSpDestroyCsrqrInfo(self.info)
      assert(status == 0)


