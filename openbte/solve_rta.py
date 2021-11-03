


def solve_rta(geometry,material,temperatures,options_solve_rta)->'solver':

    def get_m(Master,data):
      Master.data = data
      return Master

    def get_boundary(RHS,eb,n_elems):

     if len(eb)  > 0:
      RHS_tmp = np.zeros(n_elems) 
      np.add.at(RHS_tmp,eb,RHS)
     else: 
      return np.zeros(n_elems)  

     return RHS_tmp


    import numpy as np
    from scipy.sparse.linalg import splu
    from termcolor import colored, cprint 
    import openbte.utils as utils
    from mpi4py import MPI
    import scipy.sparse as sp
    import time
    import sys
    import scipy
    from cachetools import cached,LRUCache
    from cachetools.keys import hashkey
    import scipy.sparse.linalg as spla
    comm = MPI.COMM_WORLD
      
    #Options
    verbose       = options_solve_rta.setdefault('verbose',True)
    max_bte_iter  = options_solve_rta.setdefault('max_bte_iter',20)
    max_bte_error = options_solve_rta.setdefault('max_bte_error',1e-3)
    keep_lu       = options_solve_rta.setdefault('keep_lu',True)
    method        = options_solve_rta.setdefault('method','direct')
    verbose        = options_solve_rta.setdefault('verbose',False)

    #-------------
    Nlu = 1e5 if keep_lu else 0
    cache_compute_lu = LRUCache(maxsize=Nlu)
    #------------ 

    X             = temperatures['data']
    kappa_fourier = temperatures['kappa'][0]
    


    #T_mat = fdata['Temperature_Fourier'] - jnp.einsum('qi,ci->qc',F,gradT)

    n_elems = len(geometry['elems'])
    mfp = material['mfp_sampled']*1e9

   
    sigma = material['sigma']*1e9
    dim = int(geometry['meta'][2])
    F = material['VMFP']
    #Get discretization----
    n_serial,n_parallel = sigma.shape[:2]

    block =  n_parallel//comm.size
    rr = range(block*comm.rank,n_parallel) if comm.rank == comm.size-1 else range(block*comm.rank,block*(comm.rank+1))

    if comm.rank == 0 and ['verbose']:
       print(flush=True)
       print('      Iter    Thermal Conductivity [W/m/K]      Error ''',flush=True)
       print(colored(' -----------------------------------------------------------','green'),flush=True)

    if len(geometry['db']) > 0:
      if comm.rank == 0: 
       Gb   = np.einsum('mqj,js->mqs',material['sigma'],geometry['db'],optimize=True)
       Gbp2 = Gb.clip(min=0)
       tmp = Gb.clip(max=0).sum(axis=0).sum(axis=0)
       tot = np.divide(1, tmp, out=np.zeros_like(tmp), where=tmp!=0)
       #data = {'GG': np.einsum('mqs,s->mqs',Gbp2,tot),'tot':tot}
       data = {'tot':tot}
       #del tot, Gbp2
       del Gbp2
      else: data = None
      if comm.size > 1:
       data = utils.create_shared_memory_dict(data)
       #GG   = data['GG']
       tot  = data['tot']
      else: 
       #GG   = data['GG']
       tot   = data['tot']

    Gbp_new = np.einsum('mqj,js->mqs',material['sigma'][:,rr,:],geometry['db'],optimize=True).clip(min=0)
    GG_new  = np.einsum('mqs,s->mqs',Gbp_new,tot)


    #Bulk properties---
    G = np.einsum('qj,jn->qn',F[rr],geometry['k'],optimize=True)
    Gp = G.clip(min=0); Gm = G.clip(max=0)
    D = np.zeros((len(rr),len(geometry['elems'])))
    np.add.at(D.T,geometry['i'],Gp.T)
    DeltaT = X.copy()

    #---boundary----------------------
    if len(geometry['db']) > 0: #boundary
     tmp = np.einsum('rj,jn->rn',material['VMFP'][rr],geometry['db'],optimize=True)  
     Gbp2 = tmp.clip(min=0); Gbm2 = tmp.clip(max=0);
     np.add.at(D.T,geometry['eb'],Gbp2.T)
     TB = DeltaT[geometry['eb']]

    #Periodic---
    sss = np.asarray(geometry['pp'][:,0],dtype=int)
    P = np.zeros((len(rr),n_elems))
    np.add.at(P.T,geometry['i'][sss],-(geometry['pp'][:,1]*Gm[:,sss]).T)
    del G,Gp
    #----------------------------------------------------

    kappa_vec = [kappa_fourier]
    kappa_old = kappa_vec[-1]
    error = 1
    kk = 0
    kappa_tot = np.zeros(1)

    kappa,kappap = np.zeros((2,n_serial,n_parallel))
    material['tc'] = material['tc']/np.sum(material['tc'])
    i  = geometry['i']
    j  = geometry['j']
    im = np.concatenate((i,list(np.arange(n_elems))))
    jm = np.concatenate((j,list(np.arange(n_elems))))
    Master = sp.csc_matrix((np.arange(len(im))+1,(im,jm)),shape=(n_elems,n_elems),dtype=np.float64)
    conversion = np.asarray(Master.data-1,int)
    J = np.zeros((n_elems,dim))

    @cached(cache=cache_compute_lu)
    def compute_lu(n,m):

       data = np.concatenate((mfp[m]*Gm[n],mfp[m]*D[n]+np.ones(n_elems)))
       A = get_m(Master,data[conversion])
       return sp.linalg.splu(A)

    def solve_iterative(n,m,B,X0):

        data = np.concatenate((mfp[m]*Gm[n],mfp[m]*D[n]+np.ones(n_elems)))
        A = get_m(Master,data[conversion])

        return spla.lgmres(A,B,x0=X0)[0]

    while kk <max_bte_iter and error > max_bte_error:

        #Multiscale scheme-----------------------------
        DeltaTp = np.zeros_like(DeltaT)
        TBp = np.zeros_like(TB)
        Jp = np.zeros_like(J)
        gradDeltaT = utils.compute_grad_common(DeltaT,geometry)
        #kappa_bal,kappa_balp = np.zeros((2,n_parallel))

                                             
        RHS = -np.einsum('c,nc->nc',TB,Gbm2) if len(geometry['db']) > 0 else  np.zeros((n_parallel,0)) 

        for n,q in enumerate(rr):
           
           for m in range(n_serial):
                 #----------------------------------------
                 B = DeltaT +  mfp[m]*(P[n] + get_boundary(RHS[n],geometry['eb'],n_elems))
                 if method =='direct':
                  X   =  compute_lu(n,m).solve(B)
                 else:
                  X =  solve_iterative(n,m,B,X)

                 kappap[m,q] = np.dot(geometry['kappa_mask'],X-DeltaT)

                 DeltaTp += X*material['tc'][m,q]

                 if len(geometry['eb']) > 0:
                  #np.add.at(TBp,np.arange(geometry['eb'].shape[0]),-X[geometry['eb']]*GG[m,q])
                  np.add.at(TBp,np.arange(geometry['eb'].shape[0]),-X[geometry['eb']]*GG_new[m,n])

                 Jp += np.einsum('c,j->cj',X,sigma[m,q,0:dim])*1e-9

        DeltaT_old = DeltaT.copy()

        comm.Barrier()

        comm.Allreduce([DeltaTp,MPI.DOUBLE],[DeltaT,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Jp,MPI.DOUBLE],[J,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([kappap,MPI.DOUBLE],[kappa,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([TBp,MPI.DOUBLE],[TB,MPI.DOUBLE],op=MPI.SUM)
        kappa_totp = np.array([np.einsum('mq,mq->',sigma[:,rr,int(geometry['meta'][-2])],kappa[:,rr])])
        comm.Allreduce([kappa_totp,MPI.DOUBLE],[kappa_tot,MPI.DOUBLE],op=MPI.SUM)

      
        kk +=1
        error = abs(kappa_old-kappa_tot[0])/abs(kappa_tot[0])
        kappa_old = kappa_tot[0]
        kappa_vec.append(kappa_tot[0])
        if verbose and comm.rank == 0:   
         print('{0:8d} {1:24.4E} {2:22.4E}'.format(kk,kappa_vec[-1],error),flush=True)

    if verbose and comm.rank == 0:
      print(colored(' -----------------------------------------------------------','green'),flush=True)


    cache_compute_lu.clear()

    #Add a dummy dimesion--
    if dim == 2:J = np.append(J, np.zeros((n_elems,1)), axis=1)

    data = {'Temperature_BTE':DeltaT,'Flux_BTE':J,'mfp_nano_sampled':kappa,'kappa':kappa_tot,'Temperature_Fourier':temperatures['data'],'Flux_Fourier':temperatures['flux'],'kappa_fourier':temperatures['kappa']}

    #Compute vorticity---
    if options_solve_rta.setdefault('compute_vorticity',False):
        data.update({'vorticity_BTE'    :compute_vorticity(geometry,J)['vorticity']}) 
        data.update({'vorticity_Fourier':compute_vorticity(geometry,temperatures['flux'])['vorticity']}) 
    #-------------------
    comm.Barrier()
    return data





          
     








