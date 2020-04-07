from __future__ import absolute_import
import numpy as np
from .MSolver import *
from scipy.sparse.linalg import splu
from termcolor import colored, cprint 
from .utils import *
import deepdish as dd

class Solver(object):

  def __init__(self,**argv):

        #-----IMPORT MESH-------------------
        if 'geometry' in argv.keys():
         self.mesh = argv['geometry'].data
        else: 
         #self.mesh = load_dictionary('geometry.h5')
         self.mesh = dd.io.load('geometry.h5')

        self.n_elems = self.mesh['n_elems'][0]

        #---------------------------------------
        self.verbose = argv.setdefault('verbose',True)
        self.alpha = argv.setdefault('alpha',1.0)
        self.n_side_per_elem = self.mesh['n_side_per_elem'][0]

        #-----IMPORT MATERIAL-------------------
        #self.mat = load_dictionary('material.h5')
        self.mat = dd.io.load('material.h5')


        self.kappa = self.mat['kappa']
        self.kappa_vec = np.zeros((len(self.mesh['elems']),3,3))

        for n in self.mesh['elem_mat_map'].keys():
            if self.mesh['elem_mat_map'][n] == 0:
                self.kappa_vec[n] = 0.5* np.eye(3)
            else:    
                self.kappa_vec[n] =  np.eye(3)


        for i in range(len(self.mesh['elems'])):
          self.kappa_vec[i] = self.kappa*np.eye(3)
         
        self.only_fourier = True
        if not argv.setdefault('only_fourier',False):
         self.only_fourier = False
         self.tc = self.mat['temp']
         self.n_index = len(self.tc)
         #Get collision matrix------------------------------------
         if len(self.mat['B']) == 0:
           self.coll = False
           self.ac = self.mat['ac']
         else:  
           self.coll = True
           tmp = self.mat['B']
           if np.allclose(tmp, np.tril(tmp)):
            tmp += tmp.T - 0.5*np.diag(np.diag(tmp))
           self.BM = np.einsum('i,ij->ij',self.mat['scale'],tmp)
         #-------------------------------------------------------

         self.sigma = self.mat['G']*1e9
         self.VMFP = self.mat['F']*1e9
         self.n_index = len(self.tc)
         #----------------IMORT MATERIAL-------
   
        self.kappa_factor = self.mesh['kappa_factor'][0]

        if self.verbose: self.print_logo()
  
        self.assemble_fourier()

        if self.verbose:
         print('                        SYSTEM INFO                 ')   
         print(colored(' -----------------------------------------------------------','green'))
         print(colored('  Space Discretization:                    ','green') + str(self.n_elems))
         if not self.only_fourier:
          print(colored('  Momentum Discretization:                 ','green') + str(len(self.tc)))
         print(colored('  Bulk Thermal Conductivity [W/m/K]:       ','green')+ str(round(self.mat['kappa'][0,0],4)))

        #solve fourier
        data = self.solve_fourier(argv)
        self.data = data
        self.data.update({'variables':{'temperature_fourier':'Fourier Temperature [K]','flux_fourier':'Fourier Flux [W/m/m/K]'}})
        #self.data.update({'variables':{'temperature_fourier':'Fourier Temperature [K]'}})


        if self.verbose:
         print(colored('  Fourier Thermal Conductivity [W/m/K]:    ','green') + str(round(data['kappa_fourier'][0],4)))
         print(colored(' -----------------------------------------------------------','green'))

        if argv.setdefault('only_fourier'):
         #save_dictionary(self.data,'solver.h5')
         dd.io.save('solver.h5',self.data)
        else:
         #if argv.setdefault('gpu',False):
         # self.solve_bte_gpu(**argv)
         #else:
          data = self.solve_bte(**argv)
          self.data.update(data)
          self.data.update({'variables':{'temperature':'BTE Temperature [K]','flux':'BTE Flux [W/m/m/K]'}})
          #save_dictionary(self.data,'solver.h5')
          dd.io.save('solver.h5',self.data)

        if self.verbose:
         print(' ')   
         print(colored('                 OpenBTE ended successfully','green'))
         print(' ')   


  def solve_bte(self,**argv):

     if self.verbose:
      print()
      print('      Iter    Thermal Conductivity [W/m/K]      Error ''')
      print(colored(' -----------------------------------------------------------','green'))

     temp_fourier = self.data['temperature_fourier']
     Tnew = temp_fourier.copy()
     TB = np.tile(temp_fourier,(self.n_side_per_elem,1)).T

     #Main matrix----------------------------------------------
     G = np.einsum('qj,jn->qn',self.VMFP,self.mesh['k'],optimize=True)
     Gp = G.clip(min=0); Gm = G.clip(max=0)
     D = np.ones((self.n_index,self.n_elems))
     for n,i in enumerate(self.mesh['i']): D[:,i] += Gp[:,n]

     
     #Compute boundary-----------------------------------------------
     Bm = np.zeros((self.n_index,self.n_elems))
     if len(self.mesh['db']) > 0:
      Gb = np.einsum('qj,jn->qn',self.VMFP,self.mesh['db'],optimize=True)
      Gbp = Gb.clip(min=0);Gbm2 = Gb.clip(max=0)
      for n,(i,j) in enumerate(zip(self.mesh['eb'],self.mesh['sb'])):
         D[:,i]  += Gbp[:,n]
         Bm[:,i] -= TB[i,j]*Gbm2[:,n]
      Gb = np.einsum('qj,jn->qn',self.sigma,self.mesh['db'],optimize=True)
      Gbp = Gb.clip(min=0); Gbm = Gb.clip(max=0)
      SS = np.einsum('qc,c->qc',Gbm2,1/Gbm.sum(axis=0))
     #---------------------------------------------------------------

     i = np.concatenate((self.mesh['i'],list(np.arange(self.n_elems))))
     j = np.concatenate((self.mesh['j'],list(np.arange(self.n_elems))))
     A = np.concatenate((Gm,D),axis=1)
     ms = MultiSolver(i,j,A,self.n_elems)

     #Periodic------------------
     P = np.zeros((self.n_index,self.n_elems))
     for n,(i,j) in enumerate(zip(self.mesh['i'],self.mesh['j'])): P[:,i] -= Gm[:,n]*self.mesh['B'][i,j]

     BB = -np.outer(self.sigma[:,0],np.sum(self.mesh['B_with_area_old'],axis=0))*self.kappa_factor*1e-18

     #---------------------------------------------
     kappa_vec = [self.data['kappa_fourier'][0]]
     miter = argv.setdefault('max_bte_iter',100)
     merror = argv.setdefault('max_bte_error',1e-4)
     alpha = argv.setdefault('alpha',1)

     error = 1
     kk = 0

     #---------------------
     X = np.tile(Tnew,(self.n_index,1))
     DeltaT = X
     X_old = X.copy()
     alpha = 1
     kappa_old = kappa_vec[-1]
     while kk < miter and error > merror:

      X_old = X.copy()

      if self.coll:
       DeltaT = np.matmul(self.BM,alpha*X+(1-alpha)*X_old) 
       X = ms.solve(P + Bm + DeltaT)
      else: 
       DeltaT = np.dot(X.T,self.ac)
       X = ms.solve(P + Bm,common = DeltaT)

      kappa = np.sum(np.multiply(BB,X))

      kk +=1
      if len(self.mesh['db']) > 0:
       Bm = np.zeros((self.n_index,self.n_elems))   
       for n,i in enumerate(self.mesh['eb']): 
         Bm[:,i] +=np.einsum('u,u,q->q',X[:,i],Gbp[:,n],SS[:,n],optimize=True)

      error = abs(kappa_old-kappa)/abs(kappa)
      kappa_old = kappa
      kappa_vec.append(kappa)
      if self.verbose:   
       print('{0:8d} {1:24.4E} {2:22.4E}'.format(kk,kappa_vec[-1],error))

     if self.verbose:
      print(colored(' -----------------------------------------------------------','green'))

     T = np.einsum('qc,q->c',X,self.tc)
     J = np.einsum('qj,qc->cj',self.sigma,X)*1e-9
     return {'kappa_vec':kappa_vec,'temperature':T,'flux':J}



  def get_decomposed_directions(self,i,j,rot=np.eye(3)):

     normal = self.mesh['normals'][i][j]
     dist   = self.mesh['dists'][i][j]
     v_orth = np.dot(normal,np.dot(rot,normal))/np.dot(normal,dist)
     v_non_orth = np.dot(rot,normal) - dist*v_orth
     return v_orth,v_non_orth

  def get_kappa(self,i,j,ll):

   if i ==j:
    return np.array(self.kappa_vec[i])
   

   normal = self.mesh['normals'][i][j]

   kappa_i = np.array(self.kappa_vec[i])
   kappa_j = np.array(self.kappa_vec[j])

   ki = np.dot(normal,np.dot(kappa_i,normal))
   kj = np.dot(normal,np.dot(kappa_j,normal))
   w  = self.mesh['interp_weigths'][ll][0]

   kappa = kj*kappa_i/(ki*(1-w) + kj*w)
 
   return kappa


    
  def assemble_fourier(self):

    F = sp.dok_matrix((self.n_elems,self.n_elems))
    B = np.zeros(self.n_elems)

    for ll in self.mesh['side_list']['active']:
      area = self.mesh['areas'][ll]  
      (i,j) = self.mesh['side_elem_map'][ll]
      vi = self.mesh['volumes'][i]
      vj = self.mesh['volumes'][j]
      kappa = self.get_kappa(i,j,ll)
      if not i == j:
       (v_orth,dummy) = self.get_decomposed_directions(i,j,rot=kappa)
       F[i,i] += v_orth/vi*area
       F[i,j] -= v_orth/vi*area
       F[j,j] += v_orth/vj*area
       F[j,i] -= v_orth/vj*area
       if  ll in self.mesh['side_list']['Periodic']:
        B[i] += self.mesh['periodic_values'][i][j][0]*v_orth/vi*area
        B[j] += self.mesh['periodic_values'][j][i][0]*v_orth/vj*area
    self.fourier = {'A':F.tocsc(),'B':B}
    


  def solve_fourier(self,argv):

    SU = splu(self.fourier['A'])

    C = np.zeros(self.n_elems)

    n_iter = 0
    kappa_old = 0
    error = 1  
    grad = np.zeros((self.n_elems,3))
    while error > argv.setdefault('max_fourier_error',1e-4) and \
                  n_iter < argv.setdefault('max_fourier_iter',10) :
    
        RHS = self.fourier['B'] + C
        temp = SU.solve(RHS)
        temp = temp - (max(temp)+min(temp))/2.0
        kappa_eff = self.compute_diffusive_thermal_conductivity(temp,grad)
        error = abs((kappa_eff - kappa_old)/kappa_eff)
        kappa_old = kappa_eff
        n_iter +=1
        grad = self.compute_grad(temp)
        
        C = self.compute_non_orth_contribution(grad)

    flux = np.einsum('cij,cj->ci',self.kappa_vec,grad)
    return {'flux_fourier':flux,'temperature_fourier':temp,'kappa_fourier':np.array([kappa_eff])}



  def compute_grad(self,temp):

   diff_temp = self.n_elems*[None]
   for i in range(len(diff_temp)):
      diff_temp[i] = len(self.mesh['elems'][i])*[0] 
   

   gradT = np.zeros((self.n_elems,3))
   for ll in self.mesh['side_list']['active'] :
    elems = self.mesh['side_elem_map'][ll]

    kc1 = elems[0]
    c1 = self.mesh['centroids'][kc1]

    ind1 = list(self.mesh['elem_side_map'][kc1]).index(ll)

    if not ll in self.mesh['side_list']['Boundary']:

     kc2 = elems[1]
     ind2 = list(self.mesh['elem_side_map'][kc2]).index(ll)
     temp_1 = temp[kc1]
     temp_2 = temp[kc2]

     if ll in self.mesh['side_list']['Periodic']:
      temp_2 -= self.mesh['periodic_values'][kc2][kc1][0]

     diff_t = temp_2 - temp_1
     
     diff_temp[kc1][ind1]  = diff_t
     diff_temp[kc2][ind2]  = -diff_t

   
   for k in range(self.n_elems) :
    tmp = np.dot(self.mesh['weigths'][k],diff_temp[k])
    gradT[k,0] = tmp[0] #THESE HAS TO BE POSITIVE
    gradT[k,1] = tmp[1]
    if self.mesh['dim'] == 3:
     gradT[k,2] = tmp[2]

   return gradT  


  def compute_non_orth_contribution(self,gradT) :

    C = np.zeros(self.n_elems)

    for ll in self.mesh['side_list']['active']:

     (i,j) = self.mesh['side_elem_map'][ll]

     if not i==j:

      area = self.mesh['areas'][ll]   
      w  = self.mesh['interp_weigths'][ll][0]
      #F_ave = w*np.dot(gradT[i],self.mat['kappa']) + (1.0-w)*np.dot(gradT[j],self.mat['kappa'])
      F_ave = w*np.dot(gradT[i],self.kappa_vec[i]) + (1.0-w)*np.dot(gradT[j],self.kappa_vec[j])
      grad_ave = w*gradT[i] + (1.0-w)*gradT[j]

      (_,v_non_orth) = self.get_decomposed_directions(i,j)#,rot=self.mat['kappa'])

      C[i] += np.dot(F_ave,v_non_orth)/2.0/self.mesh['volumes'][i]*area
      C[j] -= np.dot(F_ave,v_non_orth)/2.0/self.mesh['volumes'][j]*area

    return C


  def compute_diffusive_thermal_conductivity(self,temp,gradT):

   kappa = 0
   for l in self.mesh['flux_sides']:

    (i,j) = self.mesh['side_elem_map'][l]
    #(v_orth,v_non_orth) = self.get_decomposed_directions(i,j,rot=self.mat['kappa'])
    (v_orth,v_non_orth) = self.get_decomposed_directions(i,j,rot=self.get_kappa(i,j,l))

    deltaT = temp[i] - temp[j] - 1
    kappa -= v_orth *  deltaT * self.mesh['areas'][l]
    w  = self.mesh['interp_weigths'][l][0]
    grad_ave = w*gradT[i] + (1.0-w)*gradT[j]
    kappa += np.dot(grad_ave,v_non_orth)/2 * self.mesh['areas'][l]

   return kappa*self.kappa_factor

  def print_logo(self):


    #v = pkg_resources.require("OpenBTE")[0].version   
    print(' ')
    print(colored(r'''        ___                   ____ _____ _____ ''','green'))
    print(colored(r'''       / _ \ _ __   ___ _ __ | __ )_   _| ____|''','green'))
    print(colored(r'''      | | | | '_ \ / _ \ '_ \|  _ \ | | |  _|  ''','green'))
    print(colored(r'''      | |_| | |_) |  __/ | | | |_) || | | |___ ''','green'))
    print(colored(r'''       \___/| .__/ \___|_| |_|____/ |_| |_____|''','green'))
    print(colored(r'''            |_|                                ''','green'))
    print()
    print('                       GENERAL INFO')
    print(colored(' -----------------------------------------------------------','green'))
    print(colored('  Contact:          ','green') + 'romanog@mit.edu                       ') 
    print(colored('  Source code:      ','green') + 'https://github.com/romanodev/OpenBTE  ')
    print(colored('  Become a sponsor: ','green') + 'https://github.com/sponsors/romanodev ')
    print(colored('  Cloud:            ','green') + 'https://shorturl.at/cwDIP             ')
    print(colored('  Mailing List:     ','green') + 'https://shorturl.at/admB0             ')
    print(colored(' -----------------------------------------------------------','green'))
    print()   

