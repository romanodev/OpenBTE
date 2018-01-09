from scipy.sparse.linalg import spsolve
import os,sys
import numpy as np 
from scipy.sparse import csc_matrix
from pyvtk import *
import numpy as np
from mpi4py import MPI
import shutil
from scipy.sparse.linalg import *
from scipy.sparse import diags
from scipy.io import *
import scipy.io
import h5py
from utils import *
import deepdish as dd
import sparse
import time

class BTE(object):
  
  def __init__(self,argv):

   self.mesh = argv['geometry']
   self.x = argv.setdefault('density',np.ones(len(self.mesh.elems)))

   #Get material properties-----
   mat = argv['material']
   self.B0 = mat.state['B0']
   self.B1 = mat.state['B1']
   self.B2 = mat.state['B2']
   self.kappa_bulk = mat.state['kappa_bulk_tot']

   mat = argv.setdefault('secondary_material',argv['material'])
   self.B0_2 = mat.state['B0']
   self.B1_2 = mat.state['B1']
   self.B2_2 = mat.state['B2']
  
   self.compute_gradient = argv.setdefault('compute_gradient',False)
   self.lu = {}


   self.mfp = np.array(mat.state['mfp_sampled'])/1e-9 #In nm
   self.n_mfp = len(self.B0)

   #INITIALIZATION-------------------------
   self.n_el = len(self.mesh.elems)
   data = self.mesh.compute_boundary_condition_data('x')
   self.side_periodic_value = data['side_periodic_value']
   self.flux_sides = data['flux_sides']
   self.area_flux = data['area_flux']
   #self.compute_matrix = argv['compute_matrix']
   self.compute_matrix = True
   self.sequence = []
   self.kappa = []
   self.dom = mat.state['dom']
   self.n_theta = self.dom['n_theta']
   self.n_phi = self.dom['n_phi']

   self.n_elems = len(self.mesh.elems)

   self.compute_connection_matrix()
   if self.compute_gradient:
    self.compute_gamma_derivative()

   if self.compute_matrix:
    if MPI.COMM_WORLD.Get_rank() == 0:
     directory = 'tmp'
     if os.path.exists(directory):
      shutil.rmtree(directory)
     os.makedirs(directory)
  
    #compute matrix--------
    output = compute_sum(self.compute_directional_connections,self.n_phi,{},{})
   
   #solve the BTE 
   if MPI.COMM_WORLD.Get_rank() == 0:
    print('Solving BTE... started.')
    print('')
   self.compute_function(**argv)

   if MPI.COMM_WORLD.Get_rank() == 0:
    print('')
    print('Solving BTE....done.')
    print('')
    print('Effective Thermal Conductivity (BTE):   ' + str(round(self.state['bte_kappa'],4)) + r' W/m/K')
    print('')

   #if argv['compute_gradient']:
   # self.compute_gradient()


   if MPI.COMM_WORLD.Get_rank() == 0:
    if os.path.isdir('tmp'):
     shutil.rmtree('tmp')


   #SAVE FILE--------------------
   if argv.setdefault('save',True):
    if MPI.COMM_WORLD.Get_rank() == 0:
     dd.io.save('solver.hdf5', self.state)


  def get_transmission(self,i,j):

   a = 2.0*self.x[i]*self.x[j]/(self.x[i]+self.x[j])
   #return 1.0

   return a

  def compute_directional_connections(self,p,options):
    
   Diff = []
   Bplus = []
   rplus = []
   cplus = []
   Bminus = []
   rminus = []
   cminus = []
   Fminus = []
   Fplus = []
   
   for i,j in zip(*self.Gamma.nonzero()):
   
    side = self.mesh.get_side_between_two_elements(i,j)  
    phi_factor = self.dom['phi_dir'][p]/self.dom['d_phi_vec'][p]
    coeff = np.dot(phi_factor,self.mesh.get_coeff(i,j))
    d_coeff = np.dot(self.dom['phi_dir'][p],self.mesh.get_normal_between_elems(i,j))/np.pi
    if coeff > 0:
     Bplus.append(coeff)
     Diff.append(d_coeff)
     rplus.append(i)
     cplus.append(j)
     v = self.mesh.get_side_periodic_value(side,i,self.side_periodic_value)
     Fplus.append(v*coeff)

    if coeff < 0 :
     Bminus.append(coeff)
     rminus.append(i)
     cminus.append(j)
     v = self.mesh.get_side_periodic_value(side,j,self.side_periodic_value)
     Fminus.append(-v*coeff)


   #--------------------------WRITE FILES---------------------
   Bp = csc_matrix( (Bplus,(rplus,cplus)), shape=(self.n_elems,self.n_elems) )
   scipy.io.mmwrite('tmp/BPLUS_' + str(p) + r'.mtx',Bp) 
   Bm = csc_matrix( (Bminus,(rminus,cminus)), shape=(self.n_elems,self.n_elems) )
   scipy.io.mmwrite('tmp/BMINUS_'+ str(p) + r'.mtx',Bm) 
   Fm = csc_matrix( (Fminus,(rminus,cminus)), shape=(self.n_elems,self.n_elems) )
   scipy.io.mmwrite('tmp/FMINUS_' + str(p) + r'.mtx',Fm) 
   Fp = csc_matrix( (Fplus,(rplus,cplus)), shape=(self.n_elems,self.n_elems) )
   scipy.io.mmwrite('tmp/FPLUS_' + str(p) + r'.mtx',Fp) 
   Dp = csc_matrix( (Diff,(rplus,cplus)), shape=(self.n_elems,self.n_elems) )
   scipy.io.mmwrite('tmp/D_' + str(p) + r'.mtx',Dp) 
   #np.save(file('tmp/F_' + str(p),'w+'),F)
   #----------------------------------------------------------
   #return Bm,Bp,Fm,Fp

  def compute_gamma_derivative(self) :
   
   nc = len(self.mesh.elems)
   row_tmp = []
   col_tmp = []
   data_tmp = [] 
   data_tmp_b = [] 

   for ll in self.mesh.side_list['active'] :
    if not ll in self.mesh.side_list['Boundary']:
     area = self.mesh.get_side_area(ll)
     elems = self.mesh.get_elems_from_side(ll)
     kc1 = elems[0]
     c1 = self.mesh.get_elem_centroid(kc1)
     normal = self.mesh.get_side_normal(0,ll)
     Af = area*normal  
     nodes = self.mesh.side_node_map[ll]
     c2 = self.mesh.get_next_elem_centroid(kc1,ll)
     kc2 = elems[1]  
     a = self.get_transmission_derivative(kc1,kc2)
     row_tmp.append(kc1)
     col_tmp.append(kc2)
     data_tmp.append(a)
     row_tmp.append(kc2)
     col_tmp.append(kc1)
     a = self.get_transmission_derivative(kc2,kc1)
     data_tmp.append(a)
  
   self.Gamma_der = csc_matrix( (np.array(data_tmp),(np.array(row_tmp),np.array(col_tmp))), shape=(nc,nc) )


  def compute_connection_matrix(self) :
   
   nc = len(self.mesh.elems)
   row_tmp = []
   col_tmp = []
   data_tmp = [] 
   data_tmp_b = [] 

   for ll in self.mesh.side_list['active'] :
    if not ll in self.mesh.side_list['Boundary']:
     area = self.mesh.get_side_area(ll)
     elems = self.mesh.get_elems_from_side(ll)
     kc1 = elems[0]
     c1 = self.mesh.get_elem_centroid(kc1)
     normal = self.mesh.get_side_normal(0,ll)
     Af = area*normal  
     nodes = self.mesh.side_node_map[ll]
     c2 = self.mesh.get_next_elem_centroid(kc1,ll)
     kc2 = elems[1]  
     a = self.get_transmission(kc1,kc2)
     row_tmp.append(kc1)
     col_tmp.append(kc2)
     data_tmp.append(a+1e-20)
     row_tmp.append(kc2)
     col_tmp.append(kc1)
     a = self.get_transmission(kc2,kc1)
     data_tmp.append(a+1e-20)
  
   self.Gamma = csc_matrix( (np.array(data_tmp),(np.array(row_tmp),np.array(col_tmp))), shape=(nc,nc) )

  def get_transmission_derivative(self,i,j):

   x1 = self.x[i]
   x2 = self.x[j]
   #f = 2.0*x1*x2/(x1+ x2)
   #return 0.5*f*f/x2/x2
   return 2.0*x2*x2/(x1+x2)/(x1+x2)



  def solve(self,p,options):

   kappa = 0.0
   

   TB = options['boundary_temp']
   Bminus=scipy.io.mmread('tmp/BMINUS_' + str(p) + '.mtx').tocsc() 
   Bplus=scipy.io.mmread('tmp/BPLUS_' + str(p) + '.mtx').tocsc()
   Fminus=scipy.io.mmread('tmp/FMINUS_' + str(p) + '.mtx').tocsc() 
   Fplus=scipy.io.mmread('tmp/FPLUS_' + str(p) + '.mtx').tocsc() 
   Diff=scipy.io.mmread('tmp/D_' + str(p) + '.mtx').tocsc() 
   phi_factor = self.dom['phi_dir'][p]/self.dom['d_phi_vec'][p]
   #-------------------------------------------------------------------------
 
    
   DS = csc_matrix(options['DS'])
   DS_new = csc_matrix((self.n_el,self.n_el))
   gradient_TL = csc_matrix(options['gradient_TL'])
   gradient_TL_new = csc_matrix(np.zeros((self.n_el,self.n_el),dtype=np.float64) )
   TL = options['temp']
   TL_new = np.zeros(self.n_el) 
   gradient_DS = options['gradient_DS']
   gradient_DS_new = np.zeros((self.n_elems,self.n_elems,self.n_elems))
   


   flux = np.zeros((self.n_el,3)) 
   new_boundary_temp = np.zeros(self.n_el) 
   suppression = np.zeros((self.n_mfp,self.n_theta,self.n_phi))
   #rd = [];cd = [];dd = [] #initialize for diffuse scattering
   
   coords = []

   symmetry = 2
   kappa = 0


   tmp = Bminus.multiply(DS-DS.multiply(self.Gamma))
   D =  -np.einsum('ij->i',tmp.todense())
   P =  np.einsum('ij->i',(Fminus.multiply(self.Gamma)).todense())
   #Building matrixes----------------------------------
   #if self.current_iter == 0:
   Am = Bminus.multiply(self.Gamma)
   b = [tmp[0,0] for tmp in Bplus.sum(axis=1)]
   Ap  = diags([b],[0]).tocsc() 


   #collect data
   A = Ap + Am
   B = P + D
   #-------------------------
   

   #----------------------------------------------------

   gradient = np.zeros(self.n_elems)
   #gradient_TB = np.zeros(self.n_elems)
   kappa_factor = 3.0/4.0/np.pi*self.mesh.size[0]/self.area_flux
   for t in range(int(self.n_theta/2.0)): #We take into account the symmetry
    theta_factor = self.dom['at'][t] / self.dom['d_theta_vec'][t]
    for m in range(self.n_mfp):

     #----------------------------------------------------------------------------------
     #index = m*self.dom['n_theta']*self.dom['n_phi'] +t*self.dom['n_phi']+p
     #if self.current_iter == 0:
     F = scipy.sparse.eye(self.n_elems) + theta_factor * self.mfp[m] * A
     lu = splu(F.tocsc())
     #self.lu.update({m*self.dom['n_theta']*self.dom['n_phi'] +t*self.dom['n_phi']+p:lu})
     #else:
      #lu = self.lu[index] #read previous lu
     #-------------------------------------------------------------------------------
     RHS = self.mfp[m]*theta_factor * B + TL

     temp = lu.solve(RHS)
    

     TL_new += self.B2[m] * temp * self.dom['d_omega'][t][p]/4.0/np.pi * symmetry
     for i in range(self.n_el) :
      flux[i] += self.B1[m]*temp[i]*self.dom['S'][t][p]
     #temp = spsolve(F.tocsc(),RHS,use_umfpack = True)
     #suppression[m,t,p] = power
     #NEW------------------------------------------------------------------------------
     pre_factor = 0.5*theta_factor/self.mfp[m]*self.dom['d_theta_vec'][t]*self.dom['d_phi_vec'][p]
     for i,j in zip(*Fminus.nonzero()):
      vol = self.mesh.get_elem_volume(i)
      suppression[m,t,p] += Fminus[i,j]*temp[i]*vol*pre_factor

     for i,j in zip(*Fplus.nonzero()):
      vol = self.mesh.get_elem_volume(i)
      suppression[m,t,p] += Fplus[i,j]*temp[i]*vol*pre_factor
     #-------------------------------------------------


     
     #for i,j in zip(*Diff.nonzero()):
     # rd.append(i); cd.append(j)
     # value =  self.dom['at'][t]*self.B1[m]*symmetry*Diff[i,j]*temp[i]
     # dd.append(value)

     temp_mat = csc_matrix(np.tile(temp,[self.n_el,1]).T)
     DS_new += self.dom['at'][t]*self.B1[m]*symmetry*Diff.multiply(temp_mat)
     


     #For diffuse scattering--------------
     #for i,j in zip(*Bplus.nonzero()):
     # rd.append(i); cd.append(j) 
     # value = self.B1[m]*symmetry*temp[i]*np.dot(self.dom['S'][t][p],self.mesh.get_normal_between_elems(i,j))/np.pi
     # dd.append(value)  
     #----------------------------------
     
     #Compute TB----------------------------------------------------------------------
     #for side in self.mesh.side_list['Boundary']:
     # elem = self.mesh.side_elem_map[side][0]
     # normal = self.mesh.compute_side_normal(elem,side)
     # if np.dot(self.dom['phonon_dir'][t][p],normal) > 0:
     #  value = self.B1[m]*symmetry*temp[elem]*np.dot(self.dom['S'][t][p],normal)/np.pi
     #  new_boundary_temp[elem] += value
      #------------------------------------------------------------------------------

      #-----------------------   
     #for n,i in enumerate(temp):
     # B2 = self.x[n]*self.B2[m] + (1.0-self.x[n])*self.B2[m]
     # new_temp += B2 * temp * self.dom['d_omega'][t][p]/4.0/np.pi * symmetry

     #new_temp += self.B2[m] * np.multiply(temp,self.x) * self.dom['d_omega'][t][p]/4.0/np.pi * symmetry
  


     #----------------------------------------------------------------------------------

     #Compute gradient---------------------
     if self.compute_gradient:
  

      T1 = Bminus.multiply(temp_mat.T.multiply(self.Gamma_der.T))
      tmp = Bminus.multiply(temp_mat.T.multiply(self.Gamma_der))
      b = [tmp[0,0] for tmp in tmp.sum(axis=1)]
      T2 = diags([b],[0]).tocsc()
      H = T1 + T2 

      T1 = Fminus.multiply(self.Gamma_der.T)
      tmp = Fminus.multiply(self.Gamma_der)
      b = [tmp[0,0] for tmp in tmp.sum(axis=1)]
      T2 = diags([b],[0]).tocsc()
      B2 = T1 + T2

      T1 = DS.multiply(Bminus.multiply(self.Gamma_der.T))
      tmp = DS.multiply(Bminus.multiply(self.Gamma_der))
      b = [tmp[0,0] for tmp in tmp.sum(axis=1)]
      T2 = diags([b],[0]).tocsc()
      B3 = T1 + T2

      T1 = -(Bminus-Bminus.multiply(self.Gamma)).todense()
      B4 =  np.einsum('kij,ij->ik',gradient_DS,T1)
      B4 =  csc_matrix(B4)

      S = B2 + B3 + B4
      
      #if self.current_iter == 1:
      # print(np.sum(B4)/np.sum(B3))
      #GX-------------------
      #NEW------------------------------------------------------------------------------
      GX = np.zeros(self.n_elems)
      pre_factor = theta_factor*self.B0[m]/self.mfp[m]*self.dom['d_theta_vec'][t]*self.dom['d_phi_vec'][p]*kappa_factor
      for i,j in zip(*Fminus.nonzero()):
       vol = self.mesh.get_elem_volume(i)
       GX[i] += Fminus[i,j]*vol*pre_factor

      for i,j in zip(*Fplus.nonzero()):
       vol = self.mesh.get_elem_volume(i)
       GX[i] += Fplus[i,j]*vol*pre_factor
     #----------------------------------------------------------------------------------

     #Compute gradient---------------------
      #Compute gradient
      Finv = inv(F.T.tocsc())
      L = Finv*GX

      #L = spsolve(F.T,GX,use_umfpack = False)
      #gradient -= L.T*(H-S-gradient_TL)
      #T_der = Finv*(H-S-gradient_TL)

      fp = (H-S)*self.mfp[m]*theta_factor-gradient_TL

      gradient -= L.T*fp
      #T_der = Finv*fp #i,k

      T_der = -inv(F.tocsc())*fp
 
      #For lattice temp----
      gradient_TL_new += T_der * self.B2[m] * self.dom['d_omega'][t][p]/4.0/np.pi * symmetry

      #if new:
      # for i,k in zip(*T_der.nonzero()):
      #  for dummy,j in zip(*Diff[i].nonzero()):
      #   gradient_DS_new[k,i,j] += self.dom['at'][t] * self.B1[m] * symmetry * T_der[i,k] * Diff[i,j]

      #else:
      gradient_DS_new += self.dom['at'][t]*self.B1[m]*symmetry*np.einsum('ij,ik->kij',Diff.todense(),T_der.todense())
         

   #---------------------------------------------
   suppression *= kappa_factor

   #DS_new = csc_matrix( (np.array(dd),(np.array(rd),np.array(cd))), shape=(self.n_el,self.n_el) )
   DS = np.array(DS_new.todense())
   kappa = np.array([kappa])
   tmp = np.zeros((self.n_el,self.n_el))
   for i,j in zip(*gradient_TL_new.nonzero()):
    tmp[i,j] = gradient_TL_new[i,j]
   #--------------------------------------------


   output = {'gradient_DS':gradient_DS_new,'gradient_TL':tmp,'kappa':kappa,'temp':TL_new,'DS':DS,'boundary_temp':new_boundary_temp,'suppression':suppression,'flux':flux,'gradient':gradient}

   return output



  #def compute_gradient(self):
  # if MPI.COMM_WORLD.Get_rank() == 0:
  #  print('Gradient Calculation started....')
  # gradient = np.zeros(self.n_elems)
  # output = compute_sum(self.solve_gradient,self.n_phi,{'output',np.zeros(self.n_elems)},{})
  # return gradient
  #def solve_gradient(self):




  def compute_function(self,**argv):
 
   fourier_temp = argv['fourier_temperature']
   max_iter = argv.setdefault('max_bte_iter',10)
   max_error = argv.setdefault('max_bte_error',1e-2)
   error = 2.0 * max_error
   previous_temp = fourier_temp
   previous_boundary_temp = fourier_temp
   #previous_gradient_T = np.zeros((self.n_el,self.n_el),dtype=np.float64)
   previous_gradient_T = argv['T_der']
   #previous_gradient_DS = np.zeros((self.n_el,self.n_el,self.n_el),dtype=np.float64)
  
   previous_kappa = 0.0
   self.current_iter = 0
   symmetry = 2 

   #Initialize DF to Fourier modeling---
   #rd = [];cd = [];dd = []
   
   #for i,j in zip(*self.Gamma.nonzero()):
   # rd.append(i);cd.append(j)
   # dd.append(fourier_temp[i])
   #DS = csc_matrix( (np.array(dd),(np.array(rd),np.array(cd))),shape=(self.n_elems,self.n_elems))
   #DS = DS.todense()

   DS = np.einsum('i,ij->ij',fourier_temp,self.mesh.A)
   #Initialize previous gradient on DS
   previous_gradient_DS = np.einsum('ij,ik->kij',self.mesh.A,previous_gradient_T)
   #------------------------
   #---------------------------------
   #-----------------------------------
   if MPI.COMM_WORLD.Get_rank() == 0:
    print('    Iter    Thermal Conductivity [W/m/K]      Error')
    print('   ---------------------------------------------------')

   while error > max_error and self.current_iter < max_iter:

    options = {'DS':np.array(DS),'temp':np.array(previous_temp),'boundary_temp':np.array(previous_boundary_temp),'gradient_TL':np.array(previous_gradient_T),'gradient_DS':np.array(previous_gradient_DS)}
    output = {'DS':np.zeros((self.n_el,self.n_el)),'temp':np.zeros(self.n_el),'kappa':np.array([0],dtype=np.float64),'boundary_temp':np.zeros(self.n_el),'suppression':np.zeros((self.n_mfp,self.n_theta,self.n_phi)),'flux':np.zeros((self.n_el,3)),'gradient':np.zeros(self.n_el),'gradient_TL':np.zeros((self.n_el,self.n_el),dtype=np.float64),'gradient_DS':np.zeros((self.n_el,self.n_el,self.n_el),dtype=np.float64)}

    #solve---
    output = compute_sum(self.solve,self.n_phi,output,options)
    #----------------
    kappa = output['kappa']
    flux = output['flux']
    previous_boundary_temp = output['boundary_temp']
    previous_temp = output['temp']
    previous_gradient_T = output['gradient_TL']
    previous_gradient_DS = output['gradient_DS']
    gradient = output['gradient']
    directional_suppression = output['suppression']
    DS = output['DS']

    #Compute kappa---------------------------
    kappa = 0
    for m in range(self.n_mfp):
     for t in range(self.n_theta):
      for p in range(self.n_phi):
       kappa += self.B0[m]*directional_suppression[m,t,p]*symmetry
    #----------------------------------------
 
    error = abs((kappa-previous_kappa))/kappa
    if MPI.COMM_WORLD.Get_rank() == 0:
     #print(np.linalg.norm(gradient))
     print('{0:7d} {1:20.4E} {2:25.4E}'.format(self.current_iter,kappa*self.kappa_bulk, error))
    
    previous_kappa = kappa
    self.current_iter +=1

   #Apply symmetry-----------------
   Nt = int(self.n_theta/2.0)
   for p in range(self.n_phi):
    for t in range(Nt):
     for m in range(self.n_mfp):
      directional_suppression[m][self.n_theta-t-1][p] = directional_suppression[m][t][p]
   #-----------------------------
   #Compute suppression function
   suppression = np.zeros(self.n_mfp) 
   for m in range(self.n_mfp):
    for t in range(self.n_theta):
     for p in range(self.n_phi):
      suppression[m] += directional_suppression[m,t,p]

   self.state = {'bte_kappa':kappa*self.kappa_bulk,\
            'directional_suppression':directional_suppression,\
            'suppression_function':suppression,\
            'gradient':gradient,\
            'dom':self.dom,\
            'mfp':self.mfp*1e-9,\
            'bte_temperature':previous_temp,\
            'bte_flux':flux}

    

   if MPI.COMM_WORLD.Get_rank() == 0:
    print('   ---------------------------------------------------')



