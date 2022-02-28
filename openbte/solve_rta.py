


def solve_rta(geometry,material,first_guess,options_solve_rta)->'solver':


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
    from openbte.first_guess import first_guess as fourier
    comm = MPI.COMM_WORLD

    #Options
    verbose       = options_solve_rta.setdefault('verbose',True)
    max_bte_iter  = options_solve_rta.setdefault('max_bte_iter',20)
    max_bte_error = options_solve_rta.setdefault('max_bte_error',1e-3)
    keep_lu       = options_solve_rta.setdefault('keep_lu',True)
    method        = options_solve_rta.setdefault('method','direct')
    verbose       = options_solve_rta.setdefault('verbose',False)

    #-------------
    Nlu = 1e5 if keep_lu else 0
    cache_compute_lu = LRUCache(maxsize=Nlu)
    #------------ 

    X             = first_guess['Temperature_Fourier']
    kappa_fourier = first_guess['kappa_fourier'][0]
    
    n_elems = len(geometry['elems'])
    mfp = material['mfp_sampled']*1e9

    sigma = material['sigma']*1e9
    dim = int(geometry['meta'][2])
    F = material['VMFP']
 
    scaling_factor = np.linalg.norm(sigma[0,0])/mfp/np.linalg.norm(F[0])*1e-18
    n_serial,n_parallel = sigma.shape[:2]

    block =  n_parallel//comm.size
    rr = range(block*comm.rank,n_parallel) if comm.rank == comm.size-1 else range(block*comm.rank,block*(comm.rank+1))

    if comm.rank == 0 and verbose:   
       print(flush=True)
       print('      Iter    Thermal Conductivity [W/m/K]         Residual ''',flush=True)
       print(colored(' -----------------------------------------------------------','green'),flush=True)

    if len(geometry['db']) > 0:
      if comm.rank == 0: 
       Gb   = np.einsum('mqj,js->mqs',material['sigma'],geometry['db'],optimize=True)
       Gbp2 = Gb.clip(min=0)
       tmp = Gb.clip(max=0).sum(axis=0).sum(axis=0)
       tot = np.divide(1, tmp, out=np.zeros_like(tmp), where=tmp!=0)
       data = {'tot':tot}
       del Gbp2
      else: data = None
      if comm.size > 1:
       data = utils.create_shared_memory_dict(data)
       tot  = data['tot']
      else: 
       tot   = data['tot']


    def get_m(Master,data):
      Master.data = data
      return Master

    def get_boundary(RHS):

     if len(geometry['eb'])  > 0:
      RHS_tmp = np.zeros(n_elems) 
      np.add.at(RHS_tmp,geometry['eb'],RHS)
     else: 
      return np.zeros(n_elems)  

     return RHS_tmp

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

    #---Thermalizing----------------------

    RHS_isothermal = np.zeros((len(rr),n_elems))
    tmp = np.einsum('rj,jn->rn',material['VMFP'][rr],geometry['dfixed'],optimize=True)  
    np.add.at(D.T,geometry['efixed'],tmp.clip(min=0).T)
    np.add.at(RHS_isothermal.T,geometry['efixed'],(tmp.clip(max=0)*geometry['fixed_temperature']).T)
    #------------------------------------

    #Periodic---
    P = np.zeros((len(rr),n_elems))
    if len(geometry['pp']) > 0:
     sss = np.asarray(geometry['pp'][:,0],dtype=int)
     np.add.at(P.T,geometry['i'][sss],-(geometry['pp'][:,1]*Gm[:,sss]).T)
    del G,Gp
    #----------------------------------------------------

    kappa_vec = [kappa_fourier]
    kappa_old = kappa_vec[-1]
    error = 1e6
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

    def solve_iterative(n,m,B,X0): #Faster for larger structures

        data = np.concatenate((mfp[m]*Gm[n],mfp[m]*D[n]+np.ones(n_elems)))
        A = get_m(Master,data[conversion])
        return spla.lgmres(A,B,x0=X0)[0]

    #TODO: This needs to be improved
    B_res = geometry['generation'][np.newaxis,np.newaxis,:]+np.einsum('m,qc->mqc',mfp,P) + RHS_isothermal 
    ap = np.array([np.sum(np.square(B_res[0]))])
    a = np.array([0.0])
    comm.Allreduce([ap,MPI.DOUBLE],[a,MPI.DOUBLE],op=MPI.SUM)
    B_res = np.sqrt(a[0])

    #---------------------------------
    
    def residual(m,n,q,x):
      """Compute Residual"""
      return mfp[m]*np.einsum('c,c->c',D[n],x) + mfp[m]*utils.sparse_dense_product(i,j,Gm[n],x) + x - geometry['generation']-mfp[m]*(P[n]+ RHS_isothermal[n])


    DeltaT_old = np.zeros_like(DeltaT)
    Boundary_old = np.zeros((len(rr),len(geometry['eb'])))
    while kk <max_bte_iter and error > max_bte_error:

        #Multiscale scheme-----------------------------
        DeltaTp    = np.zeros_like(DeltaT)
        r,rp       = np.zeros((2,n_elems))
        divJ,divJp = np.zeros((2,n_elems))
        TBp     = np.zeros_like(TB)
        Jp      = np.zeros_like(J)
        gradDeltaT = utils.compute_grad_common(DeltaT,geometry)
        Boundary = -np.einsum('c,nc->nc',TB,Gbm2) if len(geometry['db']) > 0 else  np.zeros((n_parallel,0))

        for n,q in enumerate(rr):
           
           for m in range(n_serial):
                 #----------------------------------------
                 B = DeltaT +  mfp[m]*(P[n] + get_boundary(Boundary[n]) + RHS_isothermal[n]) + geometry['generation'] 

                 if method =='direct':
                  X =  compute_lu(n,m).solve(B)
                 else:
                  X =  solve_iterative(n,m,B,X)

                 rp    += residual(m,n,q,X)-DeltaT_old - mfp[m]*get_boundary(Boundary_old[n]) 

                 kappap[m,q] = np.dot(geometry['kappa_mask'],X-DeltaT)

                 DeltaTp += X*material['tc'][m,q]

                 if len(geometry['eb']) > 0:
                  np.add.at(TBp,np.arange(geometry['eb'].shape[0]),-X[geometry['eb']]*GG_new[m,n])

                 Jp += np.einsum('c,j->cj',X,sigma[m,q,0:dim])*1e-9

        DeltaT_old   = DeltaT.copy()
        Boundary_old = Boundary.copy()
        comm.Barrier()

        comm.Allreduce([DeltaTp,MPI.DOUBLE],[DeltaT,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Jp,MPI.DOUBLE],[J,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([kappap,MPI.DOUBLE],[kappa,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([rp,MPI.DOUBLE],[r,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([TBp,MPI.DOUBLE],[TB,MPI.DOUBLE],op=MPI.SUM)
        kappa_totp = np.array([np.einsum('mq,mq->',sigma[:,rr,int(geometry['meta'][5])],kappa[:,rr])])
        comm.Allreduce([kappa_totp,MPI.DOUBLE],[kappa_tot,MPI.DOUBLE],op=MPI.SUM)

        if kk == 0:
           R0 = np.linalg.norm(r)
           error = 1
        else:   
           error = np.linalg.norm(r)/R0

        kk +=1
        kappa_old = kappa_tot[0]
        kappa_vec.append(kappa_tot[0])
        if verbose and comm.rank == 0:   
         print('{0:8d} {1:24.4E} {2:26.4E}'.format(kk,kappa_vec[-1],error),flush=True)

        comm.Barrier()

    comm.Barrier()

    if verbose and comm.rank == 0:
      print(colored(' -----------------------------------------------------------','green'),flush=True)


    cache_compute_lu.clear()

    #Add a dummy dimesion--
    if dim == 2:J = np.append(J, np.zeros((n_elems,1)), axis=1)

    data = {'Temperature_BTE':DeltaT,'Flux_BTE':J,'mfp_nano_sampled':kappa,'kappa':kappa_tot}

  
    comm.Barrier()
    #Compute vorticity---
    if options_solve_rta.setdefault('compute_vorticity',False):
        tmp2 = None
        if comm.rank == 0:
            tmp2  = {'vorticity_BTE':utils.compute_vorticity(geometry,J)['vorticity'],'vorticity_Fourier':utils.compute_vorticity(geometry,first_guess['Flux_Fourier'])['vorticity']} 
        tmp = comm.bcast(tmp2,root=0)
        data.update(comm.bcast(tmp))
    #-------------------
    comm.Barrier()
    return data





          
     








