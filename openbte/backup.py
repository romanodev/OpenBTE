import numpy as np
import scipy.sparse as sp

class MultiSolver(object):

  def __init__(self,row,col,A,A0,**argv):

   self.mode = argv.setdefault('mode','cpu')
   if self.mode == 'cpu':  
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
   else: 
    self.A = A
    self.A0 = A0
    self.row = row
    self.col = col


  def solve(self,B):    

   if self.mode =='cpu': 
    x = np.zeros_like(B) 
    for i in self.lu.keys():
     x[i] = self.lu[i].solve(B[i]/self.scale[i])

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
      tt = np.float64
      (n_batch,d) = np.shape(B)
      n = ctypes.c_int(d)
      m = ctypes.c_int(d)
      b1 = ctypes.c_int()
      b2 = ctypes.c_int()
      n_batch = ctypes.c_int(n_batch)

      row = np.concatenate((self.row,np.array(range(d))))
      col = np.concatenate((self.col,np.array(range(d))))
      nnz = ctypes.c_int(len(row))
      Acsr = sp.csr_matrix( (np.ones(len(col)),(row,col)), shape=(d,d),dtype=tt).sorted_indices()


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
      for i in range(nbatch.value):
        Acsr = sp.csr_matrix((np.concatenate((A[i,:],np.ones(d))),(row,col)),shape=(n.value,m.value),dtype=tt).sorted_indices()
        global_data.append(Acsr.data)
      data = np.ascontiguousarray(global_data,dtype=tt)

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
                           nbatch,
                           info,
                           ctypes.byref(b1),
                           ctypes.byref(b2)
                           );

      print(b2.value/1024/1024)
      print(b1.value/1024/1024)


      #dcsrIndPtr = gpuarray.to_gpu(Acsr.indptr)
     # (nbatch,nnz) = np.shape(self.A)     
     # Acsr = sp.csr_matrix( (np.ones(len(self.col)),(self.row,self.col)), shape=(d,d),dtype=tt).sorted_indices()
     # dcsrIndPtr = gpuarray.to_gpu(Acsr.indptr)
     # dcsrColInd = gpuarray.to_gpu(Acsr.indices)
      status = _libcusolver.cusolverSpDestroy(cuso_handle)
      assert(status == 0)
      status = _libcusparse.cusparseDestroy(cusp_handle)
      assert(status == 0)




      print('g')
