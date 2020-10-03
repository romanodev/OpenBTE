from __future__ import absolute_import
import numpy as np
from scipy.sparse.linalg import splu
from termcolor import colored, cprint 
from .utils import *
from .fourier import *
#import deepdish as dd
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

    if len(eb)  > 0:
     RHS_tmp = np.zeros(n_elems) 
     np.add.at(RHS_tmp,eb,RHS)
    else: 
      return np.zeros(n_elems)  

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



def solve_rta(**argv):

    mesh    = argv['mesh']
    mat     = argv['mat']
    fourier = argv['fourier']
    n_elems = len(mesh['elems'])
    mfp = mat['mfp_sampled']*1e9
    sigma = mat['sigma']*1e9

    if comm.rank == 0 and argv['verbose']:
       print(flush=True)
       print('      Iter    Thermal Conductivity [W/m/K]      Error ''',flush=True)
       print(colored(' -----------------------------------------------------------','green'),flush=True)

    if len(mesh['db']) > 0:
      if comm.rank == 0:  
       Gb   = np.einsum('mqj,jn->mqn',mat['sigma'],mesh['db'],optimize=True)
       Gbp2 = Gb.clip(min=0);
       with np.errstate(divide='ignore', invalid='ignore'):
          tot = 1/Gb.clip(max=0).sum(axis=0).sum(axis=0); tot[np.isinf(tot)] = 0
       data = {'GG': np.einsum('mqs,s->mqs',Gbp2,tot)}
       del tot, Gbp2
      else: data = None
      GG = create_shared_memory_dict(data)['GG']
 
    #Bulk properties---
    G = np.einsum('qj,jn->qn',mat['VMFP'][argv['rr']],mesh['k'],optimize=True)
    Gp = G.clip(min=0); Gm = G.clip(max=0)
    D = np.zeros((len(argv['rr']),len(mesh['elems'])))

    #for n,i in enumerate(mesh['i']):  D[:,i] += Gp[:,n]
    np.add.at(D.T,mesh['i'],Gp.T)

    DeltaT = fourier['temperature_fourier'].copy()
    #--------------------------

    #---boundary----------------------
    TB = np.zeros((argv['n_serial'],len(mesh['eb'])))   
    if len(mesh['db']) > 0: #boundary
     tmp = np.einsum('rj,jn->rn',mat['VMFP'][argv['rr']],mesh['db'],optimize=True)  
     Gbp2 = tmp.clip(min=0); Gbm2 = tmp.clip(max=0);

     np.add.at(D.T,mesh['eb'],Gbp2.T)
     TB = DeltaT[mesh['eb']]

     #for n,i in enumerate(mesh['eb']):
     #    D[:, i]  += Gbp2[:,n]
     #    TB[:,n]  = DeltaT[i] 


    #Periodic---
    sss = np.asarray(mesh['pp'][:,0],dtype=int)
    P = np.zeros((len(argv['rr']),n_elems))
    #for (_,v),ss in zip(mesh['pp'],sss): P[:,mesh['i'][ss]] -= Gm[:,ss].copy()*v
    np.add.at(P.T,mesh['i'][sss],-(mesh['pp'][:,1]*Gm[:,sss]).T)
    del G,Gp
    #----------------------------------------------------

    lu =  {}
    kappa_vec = [fourier['meta'][0]]
    kappa_old = kappa_vec[-1]
    error = 1
    kk = 0
    kappa_tot = np.zeros(1)
    MM = np.zeros(2)
    Mp = np.zeros(2)
    kappa,kappap = np.zeros((2,argv['n_serial'],argv['n_parallel']))
    transitionp = np.zeros(argv['n_parallel'])
    transition = 1e4 * np.ones(argv['n_parallel'])


    #T_old = np.tile(DeltaT,(argv['n_parallel'],argv['n_serial']))

    Master = sp.csc_matrix((np.arange(len(mesh['im']))+1,(mesh['im'],mesh['jm'])),shape=(n_elems,n_elems),dtype=np.float64)
    conversion = np.asarray(Master.data-1,np.int) 

    J = np.zeros((n_elems,argv['dim']))

  
    #DELTA---
       
    #----------------


    cache = {}
    while kk < argv['max_bte_iter'] and error > argv['max_bte_error']:

        #for ll in mesh['periodic_sides']:
        # (k1,k2) = mesh['side_elem_map_vec'][ll] 
        # DT = abs(DeltaT[k1] - DeltaT[k2])
        # if DT > 0.1:
        #print(DT)


        a = time.time()
        if argv['multiscale']: tf,tfg = solve_fourier(mat['mfp_average'],DeltaT,argv)
        #Multiscale scheme-----------------------------
        diffusive = 0
        bal = 0
        DeltaTp = np.zeros_like(DeltaT)
        TBp = np.zeros_like(TB)
        Jp = np.zeros_like(J)
        kappa_bal,kappa_balp = np.zeros((2,argv['n_parallel']))
        #Sup,Supp   = np.zeros((2,len(mat['kappam'])))
        #Supd,Supdp   = np.zeros((2,len(mat['kappam'])))
        #Supb,Supbp   = np.zeros((2,len(mat['kappam'])))

        RHS = -np.einsum('c,nc->nc',TB,Gbm2) if len(mesh['db']) > 0 else  np.zeros((argv['n_serial'],argv['n_parallel'],0)) 

        for n,q in enumerate(argv['rr']):
           
           #Get ballistic kappa
           if argv['multiscale']:
              kappa_fourier_m =  np.einsum('c,mc->m',mesh['kappa_mask'],tf) - np.einsum('c,m,i,mci->m',mesh['kappa_mask'],mfp,mat['VMFP'][q,0:argv['dim']],tfg)
              #Supdp += np.einsum('m,mu->u',kappa_fourier_m,mat['suppression'][:,q,:])*1e9


              B = DeltaT +  mfp[-1]*(P[n] +get_boundary(RHS[n],mesh['eb'],n_elems))
              A = get_m(Master,np.concatenate((mfp[-1]*Gm[n],mfp[-1]*D[n]+np.ones(n_elems)))[conversion])
              if not argv['keep_lu']:
                  X_bal = sp.linalg.spsolve(A,B,use_umfpack=True)
              else: 
                  X_bal = (lu[(-1,n)] if (-1,n) in lu.keys() else lu.setdefault((-1,n),sp.linalg.splu(A))).solve(B)

              kappa_bal = np.dot(mesh['kappa_mask'],X_bal)

              idx = np.argwhere(np.diff(np.sign(kappa_bal*np.ones(argv['n_serial']) - kappa_fourier_m))).flatten()
              if len(idx) == 0: idx = [argv['n_serial']-1]
           else: idx = [argv['n_serial']-1]

           fourier = False
           for m in range(argv['n_serial'])[idx[0]::-1]:
                 
                 B = DeltaT +  mfp[m]*(P[n] +get_boundary(RHS[n],mesh['eb'],n_elems))
                 A = get_m(Master,np.concatenate((mfp[m]*Gm[n],mfp[m]*D[n]+np.ones(n_elems)))[conversion])
                 if not argv['keep_lu']:
                  X = sp.linalg.spsolve(A,B,use_umfpack=True)
                 else: 
                  X = (lu[(m,n)] if (m,n) in lu.keys() else lu.setdefault((m,n),sp.linalg.splu(A))).solve(B)

                 kappap[m,q] = np.dot(mesh['kappa_mask'],X)#-DeltaT)

                 if argv['multiscale']:
                  error = abs(kappap[m,q] - kappa_fourier_m[m])/abs(kappap[m,q])
                  if error < argv['multiscale_error'] and m <= transition[q]:

                   fourier = True   
                   if argv.setdefault('transition',False):
                       transitionp[q] = m
                   else:
                       transitionp[q] = 1e4

                   #Vectorize
                   #kappap[:m+1,q] = kappa_fourier_m[:m+1]
                   kappap[:m,q] = kappa_fourier_m[:m]
                   diffusive += m
                   #Xm = tf[:m+1] - np.einsum('m,i,mci->mc',mat['mfp_sampled'][:m+1],mat['VMFP'][q,:argv['dim']],tfg[:m+1])
                   Xm = tf[:m] - np.einsum('m,i,mci->mc',mat['mfp_sampled'][:m],mat['VMFP'][q,:argv['dim']],tfg[:m])
                   #DeltaTp += np.einsum('mc,m->c',Xm,mat['tc'][:m+1,q])  
                   DeltaTp += np.einsum('mc,m->c',Xm,mat['tc'][:m,q])  
                  
                   #np.add.at(TBp,np.arange(mesh['eb'].shape[0]),-np.einsum('mc,mc->c',Xm[:,mesh['eb']],GG[:m+1,q]))
                  
                   
                   if len(mesh['eb']) > 0:
                    np.add.at(TBp,np.arange(mesh['eb'].shape[0]),-np.einsum('mc,mc->c',Xm[:,mesh['eb']],GG[:m,q]))
                   #Jp += np.einsum('mc,mj->cj',Xm,sigma[:m+1,q,0:argv['dim']])*1e-18
                   Jp += np.einsum('mc,mj->cj',Xm,sigma[:m,q,0:argv['dim']])*1e-18
                   break

                 DeltaTp += X*mat['tc'][m,q]
                #for c,i in enumerate(mesh['eb']): TBp[m,c] -= X[i]*GG[m,q,c]

                 if len(mesh['eb']) > 0:
                  np.add.at(TBp,np.arange(mesh['eb'].shape[0]),-X[mesh['eb']]*GG[m,q])

                 Jp += np.einsum('c,j->cj',X,sigma[m,q,0:argv['dim']])*1e-18
                 #Supp += kappap[m,q]*mat['suppression'][m,q,:]*1e9
                
 
           for m in range(argv['n_serial'])[idx[0]+1:]:


               B = DeltaT +  mfp[m]*(P[n] +get_boundary(RHS[n],mesh['eb'],n_elems))
               A = get_m(Master,np.concatenate((mfp[m]*Gm[n],mfp[m]*D[n]+np.ones(n_elems)))[conversion])
               if not argv['keep_lu']:
                  X = sp.linalg.spsolve(A,B,use_umfpack=True)
               else: 
                  X = (lu[(m,n)] if (m,n) in lu.keys() else lu.setdefault((m,n),sp.linalg.splu(A))).solve(B)

               kappap[m,q] = np.dot(mesh['kappa_mask'],X)

               error_bal = abs(kappap[m,q] - kappa_bal)/abs(kappap[m,q])
               if error_bal < argv['multiscale_error'] and \
                  abs(kappap[m-1,q] - kappa_bal)/abs(kappap[m-1,q]) < argv['multiscale_error'] and  m > int(len(mat['mfp_sampled'])/2):

                   kappap[m:,q] = kappa_bal
                   bal += argv['n_serial']-m
                   DeltaTp += X_bal*np.sum(mat['tc'][m:,q])
                   #for c,i in enumerate(mesh['eb']): TBp[m:,c] -= X_bal[i]*GG[m:,q,c]

                   np.add.at(TBp,np.arange(mesh['eb'].shape[0]),-np.einsum('c,mc->c',X_bal[mesh['eb']],GG[m:,q]))

                   Jp += np.einsum('c,j->cj',X_bal,np.sum(sigma[m:,q,0:argv['dim']],axis=0))*1e-18
                   #Supp += np.einsum('m,mu->u',kappap[m:,q],mat['suppression'][m:,q,:])*1e9
                   break

               DeltaTp += X*mat['tc'][m,q]
               
               #for c,i in enumerate(mesh['eb']): TBp[m,c] -= X[i]*GG[m,q,c]
               #np.add.at(TBp[m],np.arange(mesh['eb'].shape[0]),-X[mesh['eb']]*GG[m,q])
               if len(mesh['eb'])>0:
                np.add.at(TBp,np.arange(mesh['eb'].shape[0]),-X[mesh['eb']]*GG[m,q])

               Jp += np.einsum('c,j->cj',X,sigma[m,q,0:argv['dim']])*1e-18
               #Supp += kappap[m,q]*mat['suppression'][m,q,:]*1e9
          

        Mp[0] = diffusive
        Mp[1] = bal
        comm.Barrier()

        comm.Allreduce([DeltaTp,MPI.DOUBLE],[DeltaT,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Jp,MPI.DOUBLE],[J,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([kappap,MPI.DOUBLE],[kappa,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Mp,MPI.DOUBLE],[MM,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([transitionp,MPI.DOUBLE],[transition,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([TBp,MPI.DOUBLE],[TB,MPI.DOUBLE],op=MPI.SUM)
        #comm.Allreduce([Supp,MPI.DOUBLE],[Sup,MPI.DOUBLE],op=MPI.SUM)
        #comm.Allreduce([Supdp,MPI.DOUBLE],[Supd,MPI.DOUBLE],op=MPI.SUM)
        #comm.Allreduce([Supbp,MPI.DOUBLE],[Supb,MPI.DOUBLE],op=MPI.SUM)
        kappa_totp = np.array([np.einsum('mq,mq->',sigma[:,argv['rr'],int(mesh['meta'][-1])],kappa[:,argv['rr']])])
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


    #print(np.dot(mesh['kappa_mask'],DeltaT)*1e18)
    output =  {'kappa':kappa_vec,'temperature':DeltaT,'flux':J,'kappa_mode':kappa-np.dot(mesh['kappa_mask'],DeltaT)}#,'suppression':Sup}
    #output =  {'kappa':kappa_vec,'temperature':DeltaT,'flux':J,'kappa_mode':kappa}#,'suppression':Sup}
    #if argv['multiscale']:   
    #    output.update({'suppression_diffusive':Supd})
    #    output.update({'suppression_diffusive':Supd,'suppression_ballistic':Supb})

    return output



          


     








