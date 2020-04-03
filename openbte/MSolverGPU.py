import numpy as np
import time
import scipy.sparse as sp



def solve_gpu(row,col,A,d,X,BM,BB,Gbp,SS,P,EB):
    print('GPU!')
    import sparseqr
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import skcuda.linalg as linalg
    import skcuda.misc as misc
    import ctypes
    linalg.init()


    tt = np.float64
    

    #cuSOLVER
    _libcusparse = ctypes.cdll.LoadLibrary('libcusparse.so')
    _libcusparse.cusparseCreate.restype = int
    _libcusparse.cusparseCreate.argtypes = [ctypes.c_void_p]

    _libcusparse.cusparseDestroy.restype = int
    _libcusparse.cusparseDestroy.argtypes = [ctypes.c_void_p]
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
    _libcusparse.cusparseDestroyMatDescr.restype = int
    _libcusparse.cusparseDestroyMatDescr.argtypes = [ctypes.c_void_p]
    _libcusparse.cusparseCreateMatDescr.restype = int
    _libcusparse.cusparseCreateMatDescr.argtypes = [ctypes.c_void_p]
    descrA = ctypes.c_void_p()
    status = _libcusparse.cusparseCreateMatDescr(ctypes.byref(descrA))
    assert(status == 0)
 
    #info
    info = ctypes.c_void_p()
    status = _libcusolver.cusolverSpCreateCsrqrInfo(ctypes.byref(info))
    assert(status==0)
    _libcusolver.cusolverSpDestroyCsrqrInfo.argtypes = [ctypes.c_void_p]
    _libcusolver.cusolverSpDestroyCsrqrInfo.restype = int


    #init variable----
    (_,nnz) = A.shape
    (n_batch,d) = X.shape
    n = ctypes.c_int(d)
    m = ctypes.c_int(d)
  
    n_batch = ctypes.c_int(n_batch)
    nnz = ctypes.c_int(nnz)
 
    #--------------------- 
    
    #Acsr = sp.csr_matrix((np.arange(len(col),dtype=int),(row,col)),shape=(d,d),dtype=int).sorted_indices()
    
    #Get E---
    Acoo = sp.coo_matrix((np.ones(len(col)),(row, col)),shape=(d,d),dtype=tt)
    _, _, E, _ = sparseqr.qr(Acoo)
    E = sparseqr.permutation_vector_to_matrix(E).tocsr()
    Acsr = sp.csr_matrix((np.arange(len(col),dtype=tt)+1e-4,(row,col)),shape=(d,d),dtype=tt)
    Acsr = E*Acsr
    print(np.arange(len(col),dtype=tt)+1e-4)
    quit()

    Acsr = Acsr.astype(int).sorted_indices()
    
    #here we scrambled the indeces in order to have consistent order with the indptr and colind
    dcsrVal = gpuarray.to_gpu(A[:,Acsr.data].flatten())
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
     
    b1 = ctypes.c_int()
    b2 = ctypes.c_int()

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

    print(b2.value/1024/1024)
    print(b1.value/1024/1024)

    w_buffer = gpuarray.zeros(b2.value, dtype=tt)
    
   
    #Trasnfer to GPU
    BM = gpuarray.to_gpu(np.ascontiguousarray(BM,dtype=tt))
    P = gpuarray.to_gpu(P).ravel()
    BB = gpuarray.to_gpu(np.ascontiguousarray(BB.flatten(),dtype=tt))
    Gbp = gpuarray.to_gpu(np.ascontiguousarray(Gbp,dtype=tt))
    EB = gpuarray.to_gpu(np.ascontiguousarray(EB,dtype=tt))
    SS = gpuarray.to_gpu(np.ascontiguousarray(SS,dtype=tt))
    X = gpuarray.to_gpu(np.ascontiguousarray(X,dtype=tt)).ravel()
   #--------------------------------   

    for i in range(1):
    
      #Processing-------------------------------------
      X = X.reshape(((n_batch.value,d)))
      DeltaT = linalg.dot(BM,X).ravel()
      tmp = linalg.dot(X,Gbp,transa='T')
      tmp = misc.multiply(tmp,EB)
      Bm = linalg.dot(SS,tmp,transb='T').ravel()
      b = P + Bm + DeltaT
      
      #print('processing',a7 -a1, flush=True) 
      
      #------------------------------------------------
      X = gpuarray.empty(n_batch.value*d,dtype=tt)
      #b = gpuarray.empty(n_batch.value*d,dtype=tt)
      a = time.time() 
      _libcusolver.cusolverSpDcsrqrsvBatched(cuso_handle,
                                 n,
                                 m,
                                 nnz,
                                 descrA,
                                 int(dcsrVal.gpudata),
                                 int(dcsrIndPtr.gpudata),
                                 int(dcsrColInd.gpudata),
                                 int(b.gpudata),
                                 int(X.gpudata),
                                 n_batch,
                                 info,
                                 int(w_buffer.gpudata))
                            
      print(X.get().nonzero())
      kappa = linalg.dot(X,BB)
      print(time.time()-a)
      print(kappa)
                                      
      #a = time.time()

    print('start release')
    status = _libcusolver.cusolverSpDestroy(cuso_handle)
    assert(status == 0)
    status = _libcusparse.cusparseDestroy(cusp_handle)
    assert(status == 0)
    status = _libcusparse.cusparseDestroyMatDescr(descrA)
    assert(status == 0)
    status = _libcusolver.cusolverSpDestroyCsrqrInfo(info)
    assert(status == 0)
    print('ebd release')
