from __future__ import absolute_import
import numpy as np
from scipy.sparse.linalg import splu
from termcolor import colored, cprint 
from .utils import *
from .fourier import *
import deepdish as dd
from mpi4py import MPI
import scipy.sparse as sp
import time
import sys
import scipy


comm = MPI.COMM_WORLD

      
def get_m(Master,data):
    
    Master.data = data
    return Master

def get_boundary(RHS,eb,n_elems):


    RHS_tmp = np.zeros(n_elems) 
    np.add.at(RHS_tmp,eb,RHS)

    #for c,i in enumerate(eb): RHS_tmp[i] += RHS[c]

    return RHS_tmp


def print_multiscale(argv,MM):

        total = argv['n_serial']*argv['n_parallel']

        print(flush=True)
        print('                  Multiscale Diagnostics        ''',flush=True)
        print(colored(' -----------------------------------------------------------','green'),flush=True)
        diff = int(MM[0])/total
        bal = int(MM[1])/total
        print(colored(' BTE:              ','green') + str(round((1-diff-bal)*100,2)) + ' %',flush=True)
        print(colored(' FOURIER:          ','green') + str(round(diff*100,2)) + ' %',flush=True)
        print(colored(' BALLISTIC:        ','green') + str(round(bal*100,2)) + ' %',flush=True)
        print(colored(' -----------------------------------------------------------','green'),flush=True)



def solve_interface(**argv):

    mesh    = argv['mesh']
    mat     = argv['mat']
    fourier = argv['fourier']
    n_elems = len(mesh['elems'])
    mfp = mat['mfp_sampled']

    if comm.rank == 0 and argv['verbose']:
       print(flush=True)
       print('      Iter    Thermal Conductivity [W/m/K]      Error ''',flush=True)
       print(colored(' -----------------------------------------------------------','green'),flush=True)

    if len(mesh['db']) > 0:
      if comm.rank == 0:  
       Gb   = np.einsum('mqj,jn->mqn',mat['sigma'],mesh['db'],optimize=True)
       Gbp2 = Gb.clip(min=0);
       with np.errstate(divide='ignore', invalid='ignore'):
          tot = 1/Gb.clip(max=0).sum(axis=1); tot[np.isinf(tot)] = 0
       data = {'GG': np.einsum('mqs,ms->mqs',Gbp2,tot)}
       del tot, Gbp2
      else: data = None
      GG = create_shared_memory_dict(data)['GG']
   

    #Get transmission----
    transmission_mo = np.ones((argv['n_serial']*argv['n_parallel'],argv['n_serial']*argv['n_parallel']))
    reflection = np.zeros((argv['n_serial']*argv['n_parallel'],argv['n_serial']*argv['n_parallel']))
    #----
 
    #Bulk properties---
    G = np.einsum('qj,jn->qn',mat['VMFP'][argv['rr']],mesh['k'],optimize=True)
    Gp = G.clip(min=0); Gm = G.clip(max=0)
    D = np.zeros((len(argv['rr']),len(mesh['elems'])))

    #a = time.time()
    #for n,i in enumerate(mesh['i']):  D[:,i] += Gp[:,n]
    np.add.at(D.T,mesh['i'],Gp.T)

    #quit()
    DeltaT = fourier['temperature_fourier'].copy()

    #Transmission Properties---
    #Tr  = np.einsum('qj,jn->qn',mat['VMFP'][argv['rr']],mesh['intk'],optimize=True)
    #Trm = G.clip(max=0)
    #--------------------------

    #---boundary----------------------
    TB = np.zeros((argv['n_serial'],len(mesh['eb'])))   
    if len(mesh['db']) > 0: #boundary
     tmp = np.einsum('rj,jn->rn',mat['VMFP'][argv['rr']],mesh['db'],optimize=True)  
     Gbp2 = tmp.clip(min=0); Gbm2 = tmp.clip(max=0);

     np.add.at(D.T,mesh['eb'],Gbp2.T)
     TB = np.tile(DeltaT[mesh['eb']],(argv['n_serial'],1))

     #for n,i in enumerate(mesh['eb']):
     #    D[:, i]  += Gbp2[:,n]
     #    TB[:,n]  = DeltaT[i] 



    #Periodic---
    P = np.zeros((len(argv['rr']),n_elems))
    for ss,v in mesh['pp']: P[:,mesh['i'][int(ss)]] -= Gm[:,int(ss)].copy()*v
    del G,Gp
    #----------------------------------------------------

    lu =  {}
    #X = DeltaT.copy()
    kappa_vec = [fourier['meta'][0]]
    kappa_old = kappa_vec[-1]
    error = 1
    kk = 0
    kappa_tot = np.zeros(1)
    MM = np.zeros(2)
    Mp = np.zeros(2)
    kappa,kappap = np.zeros((2,argv['n_serial'],argv['n_parallel']))
    #termination = True
    transitionp = np.zeros(argv['n_parallel'])
    transition = 1e4 * np.ones(argv['n_parallel'])

    T_old = np.tile(DeltaT,(argv['n_parallel'],argv['n_serial']))

    Master = sp.csc_matrix((np.arange(len(mesh['im']))+1,(mesh['im'],mesh['jm'])),shape=(n_elems,n_elems),dtype=np.float64)
    conversion = np.asarray(Master.data-1,np.int) 

    J = np.zeros((n_elems,argv['dim']))

    cache = {}
    while kk < argv['max_bte_iter'] and error > argv['max_bte_error']:
       
        a = time.time()
        if argv['multiscale']: tf,tfg = solve_fourier(mat['mfp_average'],DeltaT,argv)
        #Multiscale scheme-----------------------------
        diffusive = 0
        bal = 0
        DeltaTp = np.zeros_like(DeltaT)
        TBp = np.zeros_like(TB)
        Jp = np.zeros_like(J)
        kappa_bal,kappa_balp = np.zeros((2,argv['n_parallel']))
        Sup,Supp   = np.zeros((2,len(mat['kappam'])))

        RHS = -np.einsum('mc,nc->mnc',TB,Gbm2) if len(mesh['db']) > 0 else  np.zeros((argv['n_serial'],argv['n_parallel'],0))   

        for n,q in enumerate(argv['rr']):
           
           fourier = False
           for m in range(argv['n_serial']):
               
                interface = np.zeros(n_elems) 
                for tt,(c1,c2) in enumerate(zip(mesh['inti'],mesh['intj'])):
                    interface[c1] += np.einsum('v,u->',transmission_mo[q,:],T_old[:,c2])*Trm[n,tt] #tranmission c2->c2
                    interface[c1] += np.einsum('v,u->',reflection[q,:],T_old[:,c1])*Trm[n,tt] #reflection  c1->c1

                 
                if len(RHS[m,n]) > 0:
                 B = DeltaT +  mfp[m]*(P[n] +  get_boundary(RHS[m,n],mesh['eb'],n_elems))
                else: 
                 B = DeltaT +  mfp[m]*P[n] 

                B += mfp[m]*interface 

                A = get_m(Master,np.concatenate((mfp[m]*Gm[n],mfp[m]*D[n]+np.ones(n_elems)))[conversion])
                X = (lu[(m,n)] if (m,n) in lu.keys() else lu.setdefault((m,n),sp.linalg.splu(A))).solve(B)


                kappap[m,q] = np.dot(mesh['kappa_mask'],X)
 
                if argv['multiscale']:
                 error = abs(kappap[m,q] - kappa_fourier_m[m])/abs(kappap[m,q])
                 if error < argv['multiscale_error'] and m <= transition[q]:
                  transitionp[q] = m 
                  #Vectorize
                  kappap[:m+1,q] = kappa_fourier_m[:m+1]
                  diffusive += m
                  Xm = tf[:m+1] - np.einsum('m,i,mci->mc',mat['mfp_sampled'][:m+1],mat['VMFP'][q,:argv['dim']],tfg[:m+1])
                  DeltaTp += np.einsum('mc,m->c',Xm,mat['tc'][:m+1,q])  

                  #for c,i in enumerate(mesh['eb']): TBp[:m+1,c] -= Xm[:m+1,i]*GG[:m+1,q,c]
                  np.add.at(TBp[:m+1].T,np.arange(mesh['eb'].shape[0]),-(Xm[:m+1,mesh['eb']]*GG[:m+1,q]).T)

                  Jp += np.einsum('mc,mj->cj',Xm,mat['sigma'][:m+1,q,0:argv['dim']])*1e-18
                  Supp += np.einsum('m,mu->u',kappap[:m+1,q],mat['suppression'][:m+1,q,:])*argv['kappa_factor']*1e-9
                  break

                DeltaTp += X*mat['tc'][m,q]

               
                if len(RHS[m,n]) > 0:
                 np.add.at(TBp[m],np.arange(mesh['eb'].shape[0]),-X[mesh['eb']]*GG[m,q])

                Jp += np.einsum('c,j->cj',X,mat['sigma'][m,q,0:argv['dim']])*1e-18
                Supp += kappap[m,q]*mat['suppression'][m,q,:]
                
 

        Mp[0] = diffusive
        Mp[1] = bal
        comm.Barrier()

        comm.Allreduce([DeltaTp,MPI.DOUBLE],[DeltaT,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Jp,MPI.DOUBLE],[J,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([kappap,MPI.DOUBLE],[kappa,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Mp,MPI.DOUBLE],[MM,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([transitionp,MPI.DOUBLE],[transition,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([TBp,MPI.DOUBLE],[TB,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Supp,MPI.DOUBLE],[Sup,MPI.DOUBLE],op=MPI.SUM)
        kappa_totp = np.array([np.einsum('mq,mq->',mat['sigma'][:,argv['rr'],0],kappa[:,argv['rr']])])
        comm.Allreduce([kappa_totp,MPI.DOUBLE],[kappa_tot,MPI.DOUBLE],op=MPI.SUM)
 

        if argv['multiscale'] and comm.rank == 0: print_multiscale(argv,MM) 

        kk +=1
        error = abs(kappa_old-kappa_tot[0])/abs(kappa_tot[0])
        kappa_old = kappa_tot[0]
        kappa_vec.append(kappa_tot[0])
        if argv['verbose'] and comm.rank == 0:   
         print('{0:8d} {1:24.4E} {2:22.4E}'.format(kk,kappa_vec[-1],error),flush=True)

    if argv['verbose'] and comm.rank == 0:
      print(colored(' -----------------------------------------------------------','green'),flush=True)

    output =  {'kappa':kappa_vec,'temperature':DeltaT,'flux':J,'suppression':Sup}
    #if self.multiscale:   
    #    output.update({'suppression_diffusive':Supd,'suppression_ballistic':Supb})

    return output



          


     








