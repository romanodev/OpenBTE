from __future__ import absolute_import

import os
import numpy as np
#import time
from scipy.sparse import csc_matrix
from mpi4py import MPI
import shutil
from scipy.sparse.linalg import splu
#from scipy.sparse.linalg import lgmres
import matplotlib.pylab as plt
from scipy.sparse import diags
import scipy.io
import deepdish as dd
#from termcolor import colored
from .geometry import Geometry
from .material import Material
from scipy import interpolate

def log_interp1d(xx, yy, kind='linear'):
    
     scale = min(yy)
     yy -= min(yy)
     yy+= 1e-12
     logx = np.log10(xx)
     logy = np.log10(yy)
     lin_interp = interpolate.interp1d(logx,logy,kind=kind)
     log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))  +scale -1e-12
     
     return log_interp


class Solver(object):

  def __init__(self,**argv):
   
   self.mesh = Geometry(type='load',filename = argv.setdefault('geometry_filename','geometry'))
   self.ms_error = argv.setdefault('ms_error',0.1)
   self.verbose = argv.setdefault('verbose',True)

   self.n_elems = len(self.mesh.elems)
   self.dim = self.mesh.dim
   self.mfe = False

   self.cache = '.cache'
   self.multiscale = argv.setdefault('multiscale',False)
   self.only_fourier = argv.setdefault('only_fourier',False)
   #self.interface_conductance = argv.setdefault('interface_conductance',1e20)

   #Get material properties---------------------------------
   #path = '.'
   #files = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and \
   #      'material' in i]

   #self.materials = {}
   #for filename in files:
   mat = Material(model='load',filename='material.hdf5')
   #print(mat.state)
   #data = dd.io.load(argv.setdefault('filename','material.hdf5'))

  
   self.J0 = mat.state['J0']
   self.J1 = mat.state['J1']
   self.J2 = mat.state['J2']
   self.J3 = mat.state['J3']
   self.kappa_bulk = mat.state['kappa_bulk']
   
   self.kappa_bulk_tot = mat.state['kappa_bulk_tot']
   self.mfp_bulk = mat.state['mfp_bulk']*1e9
   self.rmfp = mat.state['rmfp']
   

   self.trials = mat.state['trials']
   self.n_mfp_bulk = len(self.mfp_bulk)
   #self.dom = mat.state['dom']   
    #gamma = mat.state['kappa_bulk_tot']/mat.state['mfp_bulk'][0]/mat.state['mfp_bulk'][0]
    #mat.state.update({'gamma':gamma})
    #self.materials.update({mat.state['region']:mat.state})
    #self.B0 = mat.state['B0']
    #self.B1 = mat.state['B1']
    #self.J = mat.state['J']
    #self.B2 = mat.state['B2']
    #self.factor = mat.state['factor']
    #self.kappa_bulk = mat.state['kappa_bulk']
    #self.kappa_bulk_tot = mat.state['kappa_bulk_tot']
    #self.mfp_bulk = mat.state['mfp_bulk']*1e9
   #-----------------------------------------------------
   
   

   
   
   self.multiscale = argv.setdefault('multiscale',False)

   self.rmfp = np.array(mat.state['rmfp']) #In nm
   self.n_rmfp = len(self.rmfp)

   #INITIALIZATION-------------------------
   self.n_el = len(self.mesh.elems)

   self.kappa_factor = self.mesh.kappa_factor

   if self.verbose:
    self.print_logo()

   self.dom = mat.state['dom']
   self.n_theta = self.dom['n_theta']
   self.n_phi = self.dom['n_phi']

   self.mesh.dom = self.dom

   self.print_dof()

   # self.n_index = self.n_phi * self.n_theta
   #---------------------

   #if self.compute_matrix:
   #directory = self.cache
   #if os.path.exists(directory):
   #  shutil.rmtree(directory,ignore_errors=True)
   #os.makedirs(directory)
   argv.setdefault('max_bte_iter',10)

   #if argv['max_bte_iter'] > 0:
   
   
   if self.dim == 2 :
    self.n_index = self.n_phi
    self.compute_directional_connections_2D()
   if self.dim == 3 :
    self.n_index = self.n_phi*self.n_theta
    self.compute_directional_connections_3D()   
    
   #self.compute_space_dependent_kappa_bulk()
   self.assemble_fourier()

   #---------------------------
   
   #solve the BTE
   
   self.compute_function(**argv)
   quit()


   if MPI.COMM_WORLD.Get_rank() == 0:
    if os.path.isdir(self.cache):
     shutil.rmtree(self.cache)

   #SAVE FILE--------------------
   if argv.setdefault('save',True):
    if MPI.COMM_WORLD.Get_rank() == 0:
     dd.io.save(argv.setdefault('filename','solver') + '.hdf5', self.state)


  def compute_directional_connections_3D(self):

    for index in range(self.n_index):

     t = int(index/self.n_phi)
     p = index%self.n_phi   
        
     angle = self.dom['polar_dir'][t]*self.dom['fphi']*self.dom['ftheta']   

     
     r=[];c=[];d=[]
     rk=[]; ck=[]; dk=[]
     P = np.zeros(self.n_elems)
     HW_minus = np.zeros(self.n_elems)
     HW_plus = np.zeros(self.n_elems)
     for i,j in zip(*self.mesh.A.nonzero()):
      (cm,cp,cmb,cpb) = self.mesh.get_angular_coeff_old(i,j,angle)
      r.append(i); c.append(i); d.append(cp) 
      v = self.mesh.get_side_periodic_value(i,j)
      rk.append(i); ck.append(j);
      dk.append(v*(cp+cm)*self.mesh.get_elem_volume(i)*0.5)
      r.append(i); c.append(j);  d.append(cm)
      P[i] += v*cm
      HW_plus[i]  += cpb
      HW_minus[i] -= cmb
     #--------------------------------------------------------------
     #--------------------------WRITE FILES---------------------
     A = csc_matrix( (d,(r,c)), shape=(self.n_elems,self.n_elems) )
     K = csc_matrix( (dk,(rk,ck)), shape=(self.n_elems,self.n_elems) )
     scipy.io.mmwrite(self.cache + '/A_' + str(index) + r'.mtx',A)
     scipy.io.mmwrite(self.cache + '/K_' + str(index) + r'.mtx',K)
     P.dump(open(self.cache + '/P_' + str(index) +r'.np','wb+'))
     HW_minus.dump(open(self.cache +'/HW_MINUS_' + str(index) +r'.np','wb+'))
     HW_plus.dump(open(self.cache + '/HW_PLUS_' + str(index) +r'.np','wb+'))


  def compute_directional_connections_2D(self):

    for index in range(self.n_index):

        
     angle = self.dom['polar_dir'][index]*self.dom['fphi']   
     r=[];c=[];d=[]
     rk=[]; ck=[]; dk=[]
     P = np.zeros(self.n_elems)
     HW_minus = np.zeros(self.n_elems)
     HW_plus = np.zeros(self.n_elems)
     
     for i,j in zip(*self.mesh.A.nonzero()):
      (cm,cp,cmb,cpb) = self.mesh.get_angular_coeff_old(i,j,angle)
      r.append(i); c.append(i); d.append(cp) 
      v = self.mesh.get_side_periodic_value(i,j)
      rk.append(i); ck.append(j);
      dk.append(v*(cp+cm)*self.mesh.get_elem_volume(i)*0.5)
      r.append(i); c.append(j);  d.append(cm)
      P[i] += v*cm
      HW_plus[i]  += cpb
      HW_minus[i] -= cmb
     #--------------------------------------------------------------
     #--------------------------WRITE FILES---------------------
     A = csc_matrix( (d,(r,c)), shape=(self.n_elems,self.n_elems) )
     K = csc_matrix( (dk,(rk,ck)), shape=(self.n_elems,self.n_elems) )
     scipy.io.mmwrite(self.cache + '/A_' + str(index) + r'.mtx',A)
     scipy.io.mmwrite(self.cache + '/K_' + str(index) + r'.mtx',K)
     P.dump(open(self.cache + '/P_' + str(index) +r'.np','wb+'))
     HW_minus.dump(open(self.cache +'/HW_MINUS_' + str(index) +r'.np','wb+'))
     HW_plus.dump(open(self.cache + '/HW_PLUS_' + str(index) +r'.np','wb+'))


      
  def solve_bte(self,**argv):


   #Variable declaration-----   
   
   argv.update({'kappa':[self.kappa_bulk_tot]})
   output = self.solve_fourier(**argv) #simple fourier----   
  
   temperature_fourier = np.tile(output['temperature_fourier'],(self.n_rmfp,1))
   #temperature_fourier_gradient = np.tile(output['temperature_fourier_gradient'],(self.n_rmfp,1,1))
   #ms = np.zeros((self.n_mfp,self.n_theta,self.n_phi))
   TB = temperature_fourier.copy()
   TL = temperature_fourier.copy()
         
   #kappa_fourier =  np.tile(output['kappa_fourier'],(self.n_rmfp))
   
    
   #FLUX = np.zeros((self.n_el,3))
   kappa_old = 0.0
   n_iter = 0
   error = 1.0
   kappa_mfe = np.square(self.rmfp)/2.0
   
   comm = MPI.COMM_WORLD  
   block = self.n_index // comm.size + 1
   lu = {}

   rank = comm.rank 


  
   while n_iter < argv.setdefault('max_bte_iter',10) and \
          error > argv.setdefault('max_bte_error',1e-2):

              
    #solve_fourier
    #if n_iter ==  argv.setdefault('max_bte_iter',10)-1:
     

    Jp,J = np.zeros((2,self.n_rmfp,self.n_elems))
    kernelp,kernel = np.zeros((2,self.n_rmfp))
    Tp,T = np.zeros((2,self.n_rmfp,self.n_elems))
    
    for kk in range(block):
     index = rank*block + kk   
     if index < self.n_index:
        
      #Read directional information-------------------------------------------------- 
      A = scipy.io.mmread(self.cache + '/A_' + str(index) + '.mtx').tocsc()
      P = np.load(open(self.cache + '/P_' + str(index) +r'.np','rb'))
      HW_MINUS = np.load(open(self.cache + '/HW_MINUS_' + str(index) +r'.np','rb'))
      HW_PLUS = np.load(open(self.cache + '/HW_PLUS_' + str(index) +r'.np','rb'))
      K = scipy.io.mmread(self.cache + '/K_' + str(index) + '.mtx').tocsc()
       
      for n in range(self.n_rmfp)[::-1]:
          
       local_index = index*self.n_rmfp + n   
       #Solve the system--------    
       RHS = self.rmfp[n]  * (P + np.multiply(TB[n],HW_MINUS)) + TL[n]      
       if not local_index in lu.keys():   
        F = scipy.sparse.eye(self.n_elems) +  A * self.rmfp[n] 
        lu.update({local_index:splu(F.tocsc())})   
       temp = lu[local_index].solve(RHS)        
       #--------------------------------
      
       Tp[n] += temp
       kernelp[n] += 3*2/4/np.pi*self.kappa_factor*K.dot(temp-TL[n]).sum()*self.dom['fphi']*self.dom['dphi'] 
       Jp[n] += 2.0*np.multiply(temp,HW_PLUS)
    
    
    comm.Allreduce([Tp,MPI.DOUBLE],[T,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce([kernelp,MPI.DOUBLE],[kernel,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce([Jp,MPI.DOUBLE],[J,MPI.DOUBLE],op=MPI.SUM)
    
    n_iter +=1
    
    
    #Solve Fourier-----------------------------------------------
    if self.multiscale:
     argv.update({'mfe_factor':1.0,\
                  'verbose':False,\
                  'lattice_temperature':TL.copy(),\
                  'boundary_temperature':TB.copy(),\
                  'kappa':kappa_mfe,\
                  'interface_conductance': self.rmfp/2})    
     output = self.solve_fourier(**argv) 
     kappa_fourier = output['kappa_fourier']
    #----------------------------------------------------------- 
    
    
    #Boundary temperature
    
    
  
    
    TB = np.tile([np.sum(np.multiply(self.J1,log_interp1d(self.rmfp,J.T[e])(self.trials))) \
         for e in range(self.n_elems)],(self.n_rmfp,1))*self.dom['dphi']/np.pi
    
    
    
    #Lattice temperature   
    TL = self.dom['dphi']/2.0/np.pi*np.tile([np.sum(np.multiply(self.J2,log_interp1d(self.rmfp,T.T[e])(self.trials))) \
        for e in range(self.n_elems)],(self.n_rmfp,1))
   
    #Suppression function   
    SUP = np.sum(np.multiply(self.J3,log_interp1d(self.rmfp,kernel.clip(min=1e-12))(self.trials)),axis=1)    
    
    kappa = np.dot(self.kappa_bulk,SUP)
   

    error = abs(kappa-kappa_old)/abs(kappa)
    
    kappa_old = kappa
     
    
       
    #Thermal conductivity   
    if rank==0:
     print(kappa)       
     
   if rank==0:
    
    if self.multiscale:
     SUP_DIF = np.sum(np.multiply(self.J0,log_interp1d(self.rmfp,kappa_fourier/kappa_mfe)(self.trials)),axis=1)   
     plt.plot(self.mfp_bulk,SUP_DIF,'r') 
    plt.plot(self.mfp_bulk,SUP) 
    #
    plt.xscale('log')
    plt.show()
    
   
    
    
    
    
    

  def print_bulk_kappa(self):
      
    if MPI.COMM_WORLD.Get_rank() == 0:
     if self.verbose:
      #print('  ')
      #for region in self.region_kappa_map.keys():
       print('{:8.2f}'.format(self.kappa_bulk_tot)+ ' W/m/K')    
       #print(region.ljust(15) +  '{:8.2f}'.format(self.region_kappa_map[region])+ ' W/m/K')    

  def compute_function(self,**argv):      
    
      
   #Bulk
   self.print_bulk_kappa()
   
   #First guess: fourier
   
   
   #BTE
   self.solve_bte(**argv)

   quit()
  

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


  def solve_fourier(self,**argv):
       
    rank = MPI.COMM_WORLD.rank    
    #Update matrices---------------------------------------------------
    G = argv.setdefault('interface_conductance',np.zeros(1))#*self.dom['correction']
    TL = argv.setdefault('lattice_temperature',np.zeros((1,self.n_el)))
    TB = argv.setdefault('boundary_temperature',np.zeros((1,self.n_el)))
    mfe_factor = argv.setdefault('mfe_factor',0.0)
    kappa_bulk = argv.setdefault('kappa',[1.0])
   
    
    
    #print(kappa_bulk)
    #G *=5.0
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
    for kk in range(block):
     index = rank*block + kk   
     if index < n_index:
    
    #set up MPI------------------
    #for index in range(n_index):
      #Aseemble matrix-------------       
     
      A  = kappa_bulk[index] * self.F + mfe_factor * csc_matrix(scipy.sparse.eye(self.n_elems)) 
      A += mfe_factor * diags([self.FF],[0]).tocsc() * G[index]  
      
      B  = kappa_bulk[index] * self.B + mfe_factor * TL[index] 
      B += mfe_factor * np.multiply(self.FF,TB[index]) * G[index]
      
      if mfe_factor == 0.0: A[10,10] = 1.0
      SU = splu(A)
      C = np.zeros(self.n_el)
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

    self.gradT = self.mesh.compute_grad(temp)
    C = np.zeros(self.n_el)
    for i,j in zip(*self.mesh.A.nonzero()):

     if not i==j:
      #ll = self.mesh.get_side_between_two_elements(i,j)
      #kappa = self.side_kappa_map[ll]
      #Get agerage gradient----
      side = self.mesh.get_side_between_two_elements(i,j)


      w = self.mesh.get_interpolation_weigths(side,i)
      grad_ave = w*self.gradT[i] + (1.0-w)*self.gradT[j]
      #------------------------
      (dumm,v_non_orth) = self.mesh.get_decomposed_directions(i,j)

      
      C[i] += np.dot(grad_ave,v_non_orth)/2.0/self.mesh.get_elem_volume(i)#*kappa
      C[j] -= np.dot(grad_ave,v_non_orth)/2.0/self.mesh.get_elem_volume(j)#*kappa


    return C, self.gradT# [-self.elem_kappa_map[n]*tmp for n,tmp in enumerate(self.gradT)]


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
     print('Elements:          ' + str(self.n_el))
     print('Azimuthal angles:  ' + str(self.n_theta))
     #print('Polar angles:      ' + str(self.n_phi))
     print('Reduced Mean-free-paths:   ' + str(self.n_rmfp))
     #print('Degree-of-freedom: ' + str(self.n_mfp*self.n_theta*self.n_phi*self.n_el))
     print('Degree-of-freedom: ' + str(self.n_rmfp*self.n_theta*self.n_phi*self.n_el))


      #Thermalize boundaries--------------------------
      #Hot = np.zeros(self.n_elems)
      #for side in self.mesh.side_list['Hot']:
   #    (coeff,elem) = self.mesh.get_side_coeff(side)#
   #    tmp = np.dot(coeff,angle_factor)#
   #    if tmp < 0:
   #     Hot[elem] = -tmp*0.5#
   #    else:
   #     r.append(elem); c.append(elem); d.append(tmp)

    #  Cold = np.zeros(self.n_elems)
    #  for side in self.mesh.side_list['Cold']:#
       #(coeff,elem) = self.mesh.get_side_coeff(side)
       #tmp = np.dot(coeff,angle_factor)
       #if tmp < 0:
       # Cold[elem] = tmp*0.5
       #else:
       # r.append(elem); c.append(elem); d.append(tmp)

      #-----------------------------------------------
  #Hot.dump(open(self.cache + '/Hot_' + str(index) +r'.np','w+'))
  #Cold.dump(open(self.cache +'/Cold_' + str(index) +r'.np','w+'))
