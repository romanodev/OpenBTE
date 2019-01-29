from __future__ import absolute_import

#import os
import numpy as np
from scipy.sparse import csc_matrix
from mpi4py import MPI
#import shutil
from scipy.sparse.linalg import splu



from scipy.sparse.linalg import spilu
import matplotlib.pylab as plt
from scipy.sparse import spdiags
import scipy.io
import deepdish as dd
from .geometry import Geometry
from scipy import interpolate
import sparse
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve_triangular
import time
from numpy.testing import assert_array_equal
import pickle
import os

def log_interp1d(xx, y, kind='linear'):
     yy = y.copy()
     
     scale = min(yy)
     yy -= min(yy)
     yy+= 1e-12
     logx = np.log10(xx)
     logy = np.log10(yy)
     lin_interp = interpolate.interp1d(logx,logy,kind=kind,fill_value='extrapolate')
     log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))  +scale -1e-12
     return log_interp


class Solver(object):

  def __init__(self,**argv):
   
   self.mesh = Geometry(type='load',filename = argv.setdefault('geometry_filename','geometry'))
   self.TT = 0.5*self.mesh.B * np.array([self.mesh.elem_volumes]).T    
   self.ms_error = argv.setdefault('ms_error',0.1)
   self.verbose = argv.setdefault('verbose',True)
   self.n_elems = len(self.mesh.elems)
   self.cache = os.getcwd() + '/.cache'
  
   self.argv = argv
   self.multiscale = argv.setdefault('multiscale',False)
   tmp = dd.io.load('material.hdf5')
   self.lu = {}
   self.lu_fourier = {}
   self.last_index = -1

   if self.mesh.dim == 3:
     self.mat = tmp['data_3D']
   else:   
     self.mat = tmp['data_2D']
       
   self.kappa_factor = self.mesh.kappa_factor
   if self.verbose: 
    self.print_logo()
    self.print_dof()


   #Compute directional connections-------
   if self.mesh.dim == 3:   
    self.n_index = self.mat['n_phi'] * self.mat['n_theta']
   else:   
    self.n_index = self.mat['n_phi']

   #start = time.time() 
   #self.compute_directional_connections()  
    
   
   #quit()
   #------------------------- 
   if self.verbose: self.print_bulk_kappa()
   
   self.assemble_fourier()
   
   #---------------------------
   
   #solve the BTE
   self.solve_bte(**argv)
   


   #if MPI.COMM_WORLD.Get_rank() == 0:
   # if os.path.isdir(self.cache):
   #  shutil.rmtree(self.cache)

   #SAVE FILE--------------------
   #if argv.setdefault('save',True):
   # if MPI.COMM_WORLD.Get_rank() == 0:
   #  dd.io.save('solver.hdf5', self.state)



  def get_solving_data2(self,index,n,TB,TL):


    nc = self.n_elems       
#   if  index != self.last_index:

    if os.path.exists(self.cache + '/P_' + str(index) + '.p') :
     self.A = scipy.sparse.load_npz(self.cache + '/A_' + str(index) + '.npz')
     self.K = scipy.sparse.load_npz(self.cache + '/K_' + str(index) + '.npz')
     self.HW_MINUS = np.load(open(self.cache +'/HW_MINUS_' + str(index) + '.p','rb'))
     self.HW_PLUS = np.load(open(self.cache +'/HW_PLUS_' + str(index) + '.p','rb'))
     self.P = np.load(open(self.cache +'/P_' + str(index) +'.p','rb'))
    else:
     angle = self.mat['control_angle'][index]
     aa = sparse.COO([0,1,2],angle,shape=(3))
     self.HW_MINUS = -sparse.tensordot(self.mesh.CM,aa,axes=1).clip(max=0)
     self.HW_PLUS = sparse.tensordot(self.mesh.CP,aa,axes=1).clip(min=0)
     test2  = sparse.tensordot(self.mesh.N,aa,axes=1)
     self.K = test2 * self.TT #broadcasting (B_ij * V_i)
     AM = test2.clip(max=0)
     self.P = (AM*self.mesh.B).sum(axis=1).todense()
     AP = spdiags(test2.clip(min=0).sum(axis=1).todense(),0,nc,nc,format='csc')
     CPB = spdiags(sparse.tensordot(self.mesh.CPB,aa,axes=1).clip(min=0),0,nc,nc,format='csc')
     self.A =  AP + AM + CPB
     scipy.sparse.save_npz(self.cache + '/A_' + str(index) + '.npz',self.A.tocsc())
     scipy.sparse.save_npz(self.cache + '/K_' + str(index) + '.npz',self.K.tocsc())
     self.P.dump(open(self.cache + '/P_' + str(index) + '.p','wb'))
     self.HW_MINUS.dump(open(self.cache + '/HW_MINUS_' + str(index) + '.p','wb'))
     self.HW_PLUS.dump(open(self.cache + '/HW_PLUS_' + str(index) + '.p','wb'))
 #    self.last_index = index

    #----------------------------------------------


    #global_index = index * self.mat['n_mfp'] +n
    #if global_index in self.lu.keys():
    # lu = self.lu[global_index]
    #else:
    F = scipy.sparse.eye(self.n_elems,format='csc') +  self.A * self.mat['mfp'][n] 
    lu = splu(F.tocsc())
    # self.lu.update({global_index:lu})

    RHS = self.mat['mfp'][n] * (self.P + np.multiply(TB[n],self.HW_MINUS)) + TL[n]      
    temp = lu.solve(RHS)

    s = self.K.dot(temp-TL[n]).sum() * self.mat['domega'][index] * 3 * self.kappa_factor

    return temp,s

      
  def solve_bte(self,**argv):

   comm = MPI.COMM_WORLD  
   rank = comm.rank 


   error = 1.0
   kappa_old = 0
   n_iter = 0
   while n_iter < argv.setdefault('max_bte_iter',10) and \
          error > argv.setdefault('max_bte_error',1e-2):


    #Solve Fourier of First Guess
    #--------------------------------------------------------------------------------------------------          
    if n_iter == 0:          
      n_mfp = 1
      kappa = [self.mat['kappa_bulk_tot']]
      G = [0]
      TL = [[]]
      TB = [[]]
    else: 
      n_mfp = self.mat['n_mfp']
      kappa = self.mat['kappa_mfe']
      G = self.mat['G']
      TL = TL.copy()
      TB = TB.copy()

    SDIFF,SDIFFp = np.zeros((2,n_mfp))          
    TDIFF,TDIFFp = np.zeros((2,n_mfp,self.n_elems))          
    TDIFFGrad,TDIFFGradp = np.zeros((2,n_mfp,self.n_elems,3))          

    block = self.mat['n_mfp'] // comm.size + 1
    for kk in range(block):
     n = rank*block + kk   
     if n < n_mfp :
      SDIFFp[n],TDIFFp[n],TDIFFGradp[n] = self.get_diffusive_suppression_function(kappa[n],G[n],TB[n],TL[n])
       
    comm.Allreduce([SDIFFp,MPI.DOUBLE],[SDIFF,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce([TDIFFp,MPI.DOUBLE],[TDIFF,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce([TDIFFGradp,MPI.DOUBLE],[TDIFFGrad,MPI.DOUBLE],op=MPI.SUM)

    if n_iter == 0: 
     TDIFFGrad=np.repeat(TDIFFGrad,self.mat['n_mfp'],axis = 0)
     TDIFF=np.repeat(TDIFF,self.mat['n_mfp'],axis = 0)
     TB = TDIFF.copy()
     TL = TDIFF.copy()
     SDIFF=SDIFF[0]*np.ones(self.mat['n_mfp'])
    #---------------------------------------------------------------------------------------------------------


    #Print diffusive Kappa---------
    if rank == 0:
     SUP_DIF = np.sum(np.multiply(self.mat['J0'],log_interp1d(self.mat['mfp'],SDIFF)(self.mat['trials'])),axis=1)   
    
     print(np.dot(SUP_DIF,self.mat['kappa_bulk']))


    Jp,J = np.zeros((2,self.mat['n_mfp'],self.n_elems))
    kernelp,kernel = np.zeros((2,self.mat['n_mfp']))
    Tp,T = np.zeros((2,self.mat['n_mfp'],self.n_elems))
    for kk in range(block):
     index = rank*block + kk   
     if index < self.n_index  :

      #Get ballistic suppression function
      temp_bal,a_bal = self.get_solving_data2(index,self.mat['n_mfp']-1,TB,TL)
      SBAL = a_bal/self.mat['mfp']
      SDIFF = SDIFF*pow(self.mat['control_angle'][index][self.mesh.direction],2)*3*self.mat['domega'][index]


      #if index == 16:
       #print(self.mat['direction_ave'][t][p])   
      # plt.plot(self.mat['mfp'],SBAL)
      # plt.plot(self.mat['mfp'],SDIFF)
       #plt.xscale('log')
       #plt.ylim([0,.1])
       #plt.show()
      #quit()  

      #idx   = np.argwhere(np.diff(np.sign(SDIFF - SBAL))).flatten()
      #idx  = [100]

      #S = np.zeros(self.mat['n_mfp'])
      #fourier = False
      #for n in range(self.mat['n_mfp'])[idx[0]::-1]:
      for n in range(self.mat['n_mfp']):

        #if  fourier :
        # s = SDIFF[n] * self.mat['mfp'][n]
        # if self.mesh.dim == 2:
        #  temp = TDIFF[n] - self.mat['mfp'][n]*np.dot(self.mat['polar_ave'][index],TDIFFGrad[n].T)
        # else: 
        #  t = int(index/self.mat['n_phi'])
        #  p = index%self.mat['n_phi']   
        #  temp = TDIFF[n] - self.mat['mfp'][n]*np.dot(self.mat['direction_ave'][t][p],TDIFFGrad[n].T)
        #else:  
        temp,s = self.get_solving_data2(index,n,TB,TL)
        # sup = s/self.mat['mfp'][n]
        # if abs((SDIFF[n] - sup)/sup) < 5e-2 and self.multiscale: fourier = True
        if n == 4:
         print(max(temp))
        Tp[n]      += temp*self.mat['domega'][index]
        Jp[n]      += np.multiply(temp,self.HW_PLUS)*self.mat['domega'][index]
        kernelp[n] += s
    
        #S[n] = s/self.mat['mfp'][n]

      #ballistic = False
      #for n in range(self.mat['n_mfp'])[idx[0]+1:-1]:

      # if  ballistic :
      #   s = SBAL[n]*self.mat['mfp'][n]
      #   temp = temp_bal
      # else:  
      #  temp,s = self.get_solving_data2(index,n,TB,TL)
      #  sup = s/self.mat['mfp'][n]
       #if abs((SBAL[n] - sup)/sup) < 5e-2 and self.multiscale: ballistic = True
       #Tp[n]      += temp*self.mat['domega'][index]
       #Jp[n]      += np.multiply(temp,self.HW_PLUS)*self.mat['domega'][index]
       #kernelp[n] += s


      #if index == 20 :
      # plt.plot(self.mat['mfp'],S)
       #plt.plot(self.mat['mfp'],SBAL)
       #plt.plot(self.mat['mfp'],SDIFF)
       #plt.plot(self.mat['mfp'][0:len(SUP_DIF)],SUP_DIF)
      # plt.xscale('log')
      # plt.ylim([0,.1])
      # plt.show()
       

    comm.Allreduce([Tp,MPI.DOUBLE],[T,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce([kernelp,MPI.DOUBLE],[kernel,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce([Jp,MPI.DOUBLE],[J,MPI.DOUBLE],op=MPI.SUM)
    for n in range(self.mat['n_mfp']):
     print(min(T[n]),max(T[n]))

    n_iter +=1
   
    #----------------------------------------------------------- 

    #Suppression function   
    SUP = np.sum(np.multiply(self.mat['J3'],log_interp1d(self.mat['mfp'],kernel.clip(min=1e-13))(self.mat['trials'])),axis=1)
    kappa = np.dot(self.mat['kappa_bulk'],SUP)
    error = abs(kappa-kappa_old)/abs(kappa)
    kappa_old = kappa


    if argv.setdefault('save_state',False) and rank == 0:
     dd.io.save('state.hdf5',{'TL':TL,'TB':TB,'SUP':SUP,'n_iter':n_iter,'error':error,'kappa':kappa})

    #Boundary temperature
    TB = 4.0*np.tile([np.sum(np.multiply(self.mat['J1'],log_interp1d(self.mat['mfp'],J.T[e])(self.mat['trials']))) \
         for e in range(self.n_elems)],(self.mat['n_mfp'],1))
    
    #Lattice temperature   
    TL = np.tile([np.sum(np.multiply(self.mat['J2'],log_interp1d(self.mat['mfp'],T.T[e])(self.mat['trials']))) \
        for e in range(self.n_elems)],(self.mat['n_mfp'],1))
    
         
    #Thermal conductivity   
    if rank==0:
     print(kappa)       
     

    #if n_iter ==  argv.setdefault('max_bte_iter',10)-1:
     #Solve Fourier-----------------------------------------------
     #if n_iter == argv.setdefault('max_bte_iter',10)-1:
   #argv.update({  'mfe_factor':1.0,\
   #               'verbose':False,\
   #               'lattice_temperature':TL.copy(),\
   #               'boundary_temperature':TB.copy(),\
   #               'kappa':self.mat['kappa_mfe'],\
   #               'interface_conductance':self.mat['G']})    
   #output = self.solve_fourier(**argv) 
   #kappa_fourier = output['kappa_fourier'] 
   



   #if rank==0:
   #  print(' ')  
   #  print(kappa)       

    #if self.multiscale:
     #SUP_DIF = np.sum(np.multiply(self.mat['J0'],log_interp1d(self.mat['mfp'],kappa_fourier/self.mat['kappa_mfe'])(self.mat['trials'])),axis=1)   
     #SUP_BAL = self.mat['mfp_bulk'][-1]*SUP[-1]/self.mat['mfp_bulk']
     #model

     #idx = np.argwhere(np.diff(np.sign(SUP_DIF - SUP_BAL))).flatten()
     #plt.scatter([self.mat['mfp_bulk'][idx[0]]],SUP[idx[0]])
     #plt.scatter([self.mat['mfp_bulk'][idx[0]]],SUP_BAL[idx[0]])
     #plt.scatter([self.mat['mfp_bulk'][idx[0]]],SUP_DIF[idx[0]])

     #plt.plot(self.mat['mfp_bulk'],SUP_DIF,'r')#,marker='o') 
   #  plt.plot(self.mat['mfp_bulk'],SUP) 
     #plt.plot(self.mat['mfp_bulk'],SUP_BAL,'b') 
   #  plt.xscale('log')
   #  plt.ylim([0,1])
   #  plt.show()
    
   
    
    
    

  def print_bulk_kappa(self):
      
    if MPI.COMM_WORLD.Get_rank() == 0:
     if self.verbose:
      #print('  ')
      #for region in self.region_kappa_map.keys():
       print('{:8.2f}'.format(self.mat['kappa_bulk_tot'])+ ' W/m/K')    
       #print(region.ljust(15) +  '{:8.2f}'.format(self.region_kappa_map[region])+ ' W/m/K')    

   
   
   

  def assemble_fourier(self) :

   if  MPI.COMM_WORLD.Get_rank() == 0:
    row_tmp = []
    col_tmp = []
    data_tmp = []
    row_tmp_b = []
    col_tmp_b = []
    data_tmp_b = []
    data_tmp_b2 = []
    B = np.zeros(self.n_elems)
    kappa_mask = []
    for kc1,kc2 in zip(*self.mesh.A.nonzero()):
     #ll = self.mesh.get_side_between_two_elements(kc1,kc2)

     #kappa = self.side_kappa_map[ll]
     kappa = 1.0
     #kappa = 100.0
     #print(kappa)        print(KAPPAp)


     (v_orth,dummy) = self.mesh.get_decomposed_directions(kc1,kc2)
     vol1 = self.mesh.get_elem_volume(kc1)
     #vol2 = self.mesh.get_elem_volume(kc2)
     row_tmp.append(kc1)
     col_tmp.append(kc2)
     data_tmp.append(-v_orth/vol1*kappa)
     row_tmp.append(kc1)
     col_tmp.append(kc1)
     data_tmp.append(v_orth/vol1*kappa)
     #----------------------
     row_tmp_b.append(kc1)
     col_tmp_b.append(kc2)
     tt = self.mesh.get_side_periodic_value(kc2,kc1)
     data_tmp_b.append(tt*v_orth*kappa)
     data_tmp_b2.append(tt)
     if tt == 1.0:
      kappa_mask.append([kc1,kc2])
     
     #---------------------
     B[kc1] += self.mesh.get_side_periodic_value(kc2,kc1)*v_orth/vol1*kappa

    #Boundary_elements
    FF = np.zeros(self.n_elems)
    boundary_mask = np.zeros(self.n_elems)
    for side in self.mesh.side_list['Boundary']:
     elem = self.mesh.side_elem_map[side][0]
     vol = self.mesh.get_elem_volume(elem)
     area = self.mesh.get_side_area(side)
     FF[elem] = area/vol
     boundary_mask[elem] = 1.0

    F = csc_matrix((np.array(data_tmp),(np.array(row_tmp),np.array(col_tmp))), shape=(self.n_elems,self.n_elems) )
    RHS = csc_matrix( (np.array(data_tmp_b),(np.array(row_tmp_b),np.array(col_tmp_b))), shape=(self.n_elems,self.n_elems) )
    PER = csc_matrix( (np.array(data_tmp_b2),(np.array(row_tmp_b),np.array(col_tmp_b))), shape=(self.n_elems,self.n_elems) )

    #-------
    data = {'F':F,'RHS':RHS,'B':B,'PER':PER,'boundary_mask':boundary_mask,'FF':FF,'kappa_mask':kappa_mask}
   else: data = None
   data =  MPI.COMM_WORLD.bcast(data,root=0)
   self.F = data['F']
   self.RHS = data['RHS']
   self.PER = data['PER']
   self.kappa_mask = data['kappa_mask']
   self.B = data['B']
   self.FF = data['FF']
   self.boundary_mask = data['boundary_mask']


  def get_diffusive_suppression_function(self,kappa,G,TB,TL):


    A  = kappa * self.F 
    B  = kappa * self.B
    if len(TL) > 0:
      B += np.multiply(self.FF,TB) * G + TL
      A += spdiags(self.FF,0,self.n_elems,self.n_elems) * G
      A += csc_matrix(scipy.sparse.eye(self.n_elems)) 
    else:
      A[10,10] = 1.0  

    SU = splu(A)
    C = np.zeros(self.n_elems)
    
    n_iter = 0
    kappa_old = 0
    error = 1        
    while error > self.argv.setdefault('max_fourier_error',1e-2) and \
                  n_iter < self.argv.setdefault('max_fourier_iter',10) :
                    
        RHS = B + C*kappa
        if len(TL) == 0.0: RHS[10] = 0.0      
        temp = SU.solve(RHS)
        temp = temp - (max(temp)+min(temp))/2.0
        (C,grad) = self.compute_non_orth_contribution(temp)
        
        kappa_eff = self.compute_diffusive_thermal_conductivity(temp)*kappa
        error = abs((kappa_eff - kappa_old)/kappa_eff)
        kappa_old = kappa_eff
        n_iter +=1

    s = kappa_eff/kappa
    

    return s,temp,grad




  def solve_fourier(self,**argv):
       
    rank = MPI.COMM_WORLD.rank    
    #Update matrices---------------------------------------------------
    G = argv.setdefault('interface_conductance',np.zeros(1))#*self.dom['correction']
    TL = argv.setdefault('lattice_temperature',np.zeros((1,self.n_elems)))
    TB = argv.setdefault('boundary_temperature',np.zeros((1,self.n_elems)))
    mfe_factor = argv.setdefault('mfe_factor',0.0)
    kappa_bulk = argv.setdefault('kappa',[1.0])
    

    n_index = len(kappa_bulk)

    #G = np.zeros(n_index)
    #-------------------------------
    KAPPAp,KAPPA = np.zeros((2,n_index))
    FLUX,FLUXp = np.zeros((2,n_index,3,self.n_elems))
    TLp,TLtot = np.zeros((2,n_index,self.n_elems))

    comm = MPI.COMM_WORLD  
    size = comm.size
    rank = comm.rank
    block = n_index // size + 1
    #block = argv.setdefault('fourier_cut_mfp',n_index) // size + 1
    for kk in range(block):
     index = rank * block + kk   
     if index < n_index :# and index < argv['fourier_cut_mfp']:
     #if index < argv.setdefault('fourier_cut_mfp',n_index) :
    
    #set up MPI------------------
    #for index in range(n_index):
      #Aseemble matrix-------------       
    
    
      A  = kappa_bulk[index] * self.F + mfe_factor * csc_matrix(scipy.sparse.eye(self.n_elems)) 
      A += mfe_factor * spdiags(self.FF,0,self.n_elems,self.n_elems) * G[index]  
      B  = kappa_bulk[index] * self.B + mfe_factor * TL[index] 
      B += mfe_factor * np.multiply(self.FF,TB[index]) * G[index]
      
      if mfe_factor == 0.0: A[10,10] = 1.0

      #if index in self.lu_fourier.keys():
      # SU = self.lu_fourier[index]

      #else:
      SU = splu(A)
      # self.lu_fourier.update({index:SU})

      C = np.zeros(self.n_elems)
      #--------------------------------------
     
      n_iter = 0
      kappa_old = 0
      error = 1        
      while error > argv.setdefault('max_fourier_error',1e-2) and \
                    n_iter < argv.setdefault('max_fourier_iter',10) :
                    
        RHS = B + C*kappa_bulk[index]
        if mfe_factor == 0.0: RHS[10] = 0.0      
        temp = SU.solve(RHS)
        temp = temp - (max(temp)+min(temp))/2.0
        (C,flux) = self.compute_non_orth_contribution(temp)
        
        KAPPAp[index] = self.compute_diffusive_thermal_conductivity(temp)*kappa_bulk[index]
        error = abs((KAPPAp[index] - kappa_old)/KAPPAp[index])
        kappa_old = KAPPAp[index]
        n_iter +=1
        
      #FLUXp = np.array([-self.elem_kappa_map[k]*tmp for k,tmp in enumerate(flux)])
      FLUXp[index] = flux.T
      TLp[index] = temp
     
        
    comm.Allreduce([KAPPAp,MPI.DOUBLE],[KAPPA,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce([TLp,MPI.DOUBLE],[TLtot,MPI.DOUBLE],op=MPI.SUM) #to be changed
    comm.Allreduce([FLUXp,MPI.DOUBLE],[FLUX,MPI.DOUBLE],op=MPI.SUM)  
    
    
    if argv.setdefault('verbose',True) and rank==0:

     print('  ')
     print('Thermal Conductivity: '.ljust(20) +  '{:8.2f}'.format(KAPPAp.sum())+ ' W/m/K')
     print('  ')
    


    return {'kappa_fourier':KAPPA,'temperature_fourier_gradient':FLUX,'temperature_fourier':TLtot}

      
  def compute_non_orth_contribution(self,temp) :

    gradT = self.mesh.compute_grad(temp)
    C = np.zeros(self.n_elems)
    for i,j in zip(*self.mesh.A.nonzero()):

     if not i==j:
      #Get agerage gradient----
      side = self.mesh.get_side_between_two_elements(i,j)


      w = self.mesh.get_interpolation_weigths(side,i)
      grad_ave = w*gradT[i] + (1.0-w)*gradT[j]
      #------------------------
      (dumm,v_non_orth) = self.mesh.get_decomposed_directions(i,j)

      
      C[i] += np.dot(grad_ave,v_non_orth)/2.0/self.mesh.get_elem_volume(i)#*kappa
      C[j] -= np.dot(grad_ave,v_non_orth)/2.0/self.mesh.get_elem_volume(j)#*kappa


    return C, gradT# [-self.elem_kappa_map[n]*tmp for n,tmp in enumerate(self.gradT)]


  def compute_inho_diffusive_thermal_conductivity(self,temp,mat=np.eye(3)):

   kappa = 0
   for i,j in zip(*self.PER.nonzero()):

    #ll = self.mesh.get_side_between_two_elements(i,j)
    #kappa_b = self.side_kappa_map[ll]
    (v_orth,dummy) = self.mesh.get_decomposed_directions(i,j,rot=mat)
   
    
    kappa += 0.5*v_orth *self.PER[i,j]*(temp[j]+self.PER[i,j]-temp[i])#*kappa_b

   return kappa*self.kappa_factor


  def compute_diffusive_thermal_conductivity(self,temp,mat=np.eye(3)):

   
   kappa = 0
   for (i,j) in self.kappa_mask:
    (v_orth,dummy) = self.mesh.get_decomposed_directions(i,j,rot=mat)
    kappa += v_orth  *(temp[j]+1.0-temp[i])

   #kappa = 0
   #for i,j in zip(*self.PER.nonzero()):
   # (v_orth,dummy) = self.mesh.get_decomposed_directions(i,j,rot=mat)
   # kappa += 0.5*v_orth *self.PER[i,j]*(temp[j]+self.PER[i,j]-temp[i])

   return kappa*self.kappa_factor

  def print_logo(self):

   if MPI.COMM_WORLD.Get_rank() == 0:
    print(' ')
    print(r'''  ___                   ____ _____ _____ ''' )
    print(r''' / _ \ _ __   ___ _ __ | __ )_   _| ____|''')
    print(r'''| | | | '_ \ / _ \ '_ \|  _ \ | | |  _|  ''')
    print(r'''| |_| | |_) |  __/ | | | |_) || | | |___ ''')
    print(r''' \___/| .__/ \___|_| |_|____/ |_| |_____|''')
    print(r'''      |_|                                ''')
    print('')
    print('Giuseppe Romano [romanog@mit.edu]')
    print(' ')

  def print_dof(self):

   if MPI.COMM_WORLD.Get_rank() == 0:
    if self.verbose:
     print(' ')
     print('Elements:          ' + str(self.n_elems))
     print('Azimuthal angles:  ' + str(self.mat['n_theta']))
     print('Polar angles:      ' + str(self.mat['n_phi']))
     print('Mean-free-paths:   ' + str(self.mat['n_mfp']))
     
    #Build data and perform superlu factorization---
    #lu_data = {'L':lu.L,'U':lu.U,'perm_c':lu.perm_c,'perm_r':lu.perm_r}
    #pickle.dump(lu_data,open(self.cache + '/lu_' + str(index*self.mat['n_mfp']+n) + '.p','wb'))
   #a = time.time()
   #b = time.time() 
   #start = time.time()
  # temp = self.spsolve_lu(lu_data,RHS)
   #c = time.time() 
   #print((c-b)/(b-a))
   #quit()
    #return Tp, kernelp, Jp
   #return Tp, np.zeros(self.n_elems), Jp

   #global_index = index*self.mat['n_mfp']+n
   #if  os.path.exists(self.cache + '/lu_' + str(global_index) + '.p') :
   # lu_data = pickle.load(open(self.cache + '/lu_' + str(index*self.mat['n_mfp']+n) + '.p','rb'))
   # self.data = pickle.load(open(self.cache + '/save_' + str(index) + '.p','rb'))
    
   #else:
  """
  def spsolve_lu(self,lu_data,b):
   perm_r = lu_data['perm_r']
   perm_c = lu_data['perm_c']
   L = lu_data['L'].tocsr()
   U = lu_data['U'].tocsr()
   if perm_r is not None:
      b_old = b.copy()
      for old_ndx, new_ndx in enumerate(perm_r):
        b[new_ndx] = b_old[old_ndx]
   c = spsolve_triangular(L, b, lower=True)
   px = spsolve_triangular(U, c, lower=False)
   return px[perm_c]
   """

