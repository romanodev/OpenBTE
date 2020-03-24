import numpy as np
import scipy.sparse as sp

class MultiSolver(object):

  def __init__(self,row,col,A,A0,**argv):

   self.tt = np.float64
   self.mode = argv.setdefault('mode','cpu')
   if self.mode == 'cpu':  
    (nbatch,d) = np.shape(A0)
    #self.scale = np.zeros(nbatch)
    self.lu = {}
    for i in range(nbatch):
      #self.scale[i] = np.max(A[i])  
      #if self.scale[i] == 0:
      #  self.scale[i] = 1
      S1 = sp.csc_matrix((A[i],(row,col)),shape=(d,d),dtype=np.float64)
      S = S1 + sp.diags(A0[i],format='csc')
      lu = sp.linalg.splu(S,permc_spec='COLAMD')
      self.lu.update({i:lu})
   else: 
    self.A = A
    self.A0 = A0
    self.row = row
    self.col = col


  def solve(self,B):    
   B = B.astype(self.tt)
   if self.mode =='cpu': 
    x = np.zeros_like(B) 
    for i in self.lu.keys():
     x[i] = self.lu[i].solve(B[i])
    return x
   else: 
      
      import ctypes
      import pycuda.gpuarray as gpuarray
      import pycuda.autoinit

      #cuSOLVER
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
      #cusolverSpDcsrqrBufferInfoBatched
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

      #cusolverSpDcsrqrsvBatched
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

      #cusp_handle
      _cusp_handle = ctypes.c_void_p()
      status = _libcusparse.cusparseCreate(ctypes.byref(_cusp_handle))
      assert(status == 0)
      cusp_handle = _cusp_handle.value

      #cuso_handle
      _cuso_handle = ctypes.c_void_p()
      status = _libcusolver.cusolverSpCreate(ctypes.byref(_cuso_handle))
      assert(status == 0)
      cuso_handle = _cuso_handle.value
 
      # create MatDescriptor
      descrA = ctypes.c_void_p()
      status = _libcusparse.cusparseCreateMatDescr(ctypes.byref(descrA))
      assert(status == 0)
 
      #info
      info = ctypes.c_void_p()
      status = _libcusolver.cusolverSpCreateCsrqrInfo(ctypes.byref(info))
      assert(status==0)

      #init
      
      (n_batch,d) = np.shape(B)
      n = ctypes.c_int(d)
      m = ctypes.c_int(d)
      b1 = ctypes.c_int()
      b2 = ctypes.c_int()
      n_batch = ctypes.c_int(n_batch)

     
      nnz = ctypes.c_int(len(self.row))
      Acsr = sp.csr_matrix( (np.ones(len(self.col)),(self.row,self.col)), shape=(d,d),dtype=self.tt).sorted_indices()
     

      dcsrIndPtr = gpuarray.to_gpu(Acsr.indptr)
      dcsrColInd = gpuarray.to_gpu(Acsr.indices)

      _libcusolver.cusolverSpXcsrqrAnalysisBatched(cuso_handle,
                                 n,
                                 m,
                                 nnz,
                                 descrA,
                                 int(dcsrIndPtr.gpudata),
                                 int(dcsrColInd.gpudata),
                                 info)
     

      #get data---
      global_data = []
      for i in range(n_batch.value):
        
        Acsr = sp.csr_matrix((self.A[i,:],(self.row,self.col)),shape=(n.value,m.value),dtype=self.tt).sorted_indices()
        Acsr = Acsr + sp.diags(self.A0[i],format='csr')
        
        global_data +=list(Acsr.data)

      
      global_data = np.array(global_data,dtype=self.tt)
      data = np.ascontiguousarray(global_data,dtype=self.tt)



      dcsrVal = gpuarray.to_gpu(data)
      

      #get info
      _libcusolver.cusolverSpDcsrqrBufferInfoBatched(cuso_handle,
                           n,
                           m,
                           nnz,
                           descrA,
                           int(dcsrVal.gpudata),
                           int(dcsrIndPtr.gpudata),
                           int(dcsrColInd.gpudata),
                           n_batch,
                           info,
                           ctypes.byref(b1),
                           ctypes.byref(b2)
                           );

    #  print(b2.value/1024/1024)
    #  print(b1.value/1024/1024)

      w_buffer = gpuarray.zeros(b2.value, dtype=self.tt)

      b = gpuarray.to_gpu(B.astype(self.tt))
      dx = pycuda.gpuarray.empty_like(b)

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
                                 n_batch,
                                 info,
                                 int(w_buffer.gpudata))
      x = dx.get()
  
      status = _libcusolver.cusolverSpDestroy(cuso_handle)
      assert(status == 0)
      status = _libcusparse.cusparseDestroy(cusp_handle)
      assert(status == 0)


      return x

