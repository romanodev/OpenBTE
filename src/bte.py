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

   self.lu = {}

   self.mfp = np.array(mat.state['mfp_sampled'])/1e-9 #In nm
   self.n_mfp = len(self.B0)

   #INITIALIZATION-------------------------
   self.n_el = len(self.mesh.elems)
   data = self.mesh.compute_boundary_condition_data('x')
   self.side_periodic_value = data['side_periodic_value']
   self.flux_sides = data['flux_sides']
   self.area_flux = data['area_flux']
   self.dom = mat.state['dom']
   self.n_theta = self.dom['n_theta']
   self.n_phi = self.dom['n_phi']
   self.n_elems = len(self.mesh.elems)

   #if self.compute_matrix:
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
    print('Effective Thermal Conductivity (BTE):   ' + str(round(self.state['kappa_bte'],4)) + r' W/m/K')
    print('')

   if MPI.COMM_WORLD.Get_rank() == 0:
    if os.path.isdir('tmp'):
     shutil.rmtree('tmp')


   #SAVE FILE--------------------
   if argv.setdefault('save',True):
    if MPI.COMM_WORLD.Get_rank() == 0:
     dd.io.save('solver.hdf5', self.state)



  def compute_directional_connections(self,p,options):

    
   Diff = []
   Fminus = []
   Fplus = []
   r=[];c=[];d=[]
   rk=[]; ck=[]; dk=[]

   P = np.zeros(self.n_elems)
   phi_factor = self.dom['phi_dir'][p]/self.dom['d_phi_vec'][p]
   
   for i,j in zip(*self.mesh.A.nonzero()):
   
    side = self.mesh.get_side_between_two_elements(i,j)  
    coeff = np.dot(phi_factor,self.mesh.get_coeff(i,j))
    if coeff > 0:
     r.append(i); c.append(i); d.append(coeff)
     v = self.mesh.get_side_periodic_value(side,i,self.side_periodic_value)
     rk.append(i); ck.append(j);      
     dk.append(v*coeff*self.mesh.get_elem_volume(i))

    if coeff < 0 :
     r.append(i); c.append(j); d.append(coeff)
     v = self.mesh.get_side_periodic_value(side,j,self.side_periodic_value)
     P[i] -= v*coeff
     rk.append(i); ck.append(j);      
     dk.append(-v*coeff*self.mesh.get_elem_volume(i))

   #Write the boundaries------
   HW_minus = np.zeros(self.n_elems)
   HW_plus = np.zeros(self.n_elems)
   for side in self.mesh.side_list['Boundary']:
    (coeff,elem) = self.mesh.get_side_coeff(side)
    tmp = np.dot(coeff,phi_factor)
    if tmp > 0:
     r.append(elem); c.append(elem); d.append(tmp)
     HW_plus[elem] = np.dot(self.dom['phi_dir'][p],self.mesh.get_side_normal(0,side))/np.pi
    else :
     HW_minus[elem] = -tmp
   #-------------

   #--------------------------WRITE FILES---------------------
   A = csc_matrix( (d,(r,c)), shape=(self.n_elems,self.n_elems) )
   K = csc_matrix( (dk,(rk,ck)), shape=(self.n_elems,self.n_elems) )

   #data = {'A':A,'K':K,'P':P,'HW_minus':HW_minus,'HW_plus':HW_plus}
   #dd.io.save('tmp/save_' + str(p) + '.hdf5', data)

   scipy.io.mmwrite('tmp/A_' + str(p) + r'.mtx',A) 
   scipy.io.mmwrite('tmp/K_' + str(p) + r'.mtx',K) 
   #scipy.sparse.save_npz('tmp/A_' + str(p) + '.npz',A)
   #scipy.sparse.save_npz('tmp/K_' + str(p) + '.npz',K)

   P.dump(file('tmp/P_' + str(p) +r'.np','wb+'))
   HW_minus.dump(file('tmp/HW_MINUS_' + str(p) +r'.np','w+'))
   HW_plus.dump(file('tmp/HW_PLUS_' + str(p) +r'.np','w+'))

  def solve(self,p,options):


   if self.current_iter == 0:
    A = scipy.io.mmread('tmp/A_' + str(p) + '.mtx').tocsc()
   # A = scipy.sparse.load_npz('tmp/A_' + str(p) + '.npz')
   #K = scipy.sparse.load_npz('tmp/K_' + str(p) + '.npz')
   P = np.load(file('tmp/P_' + str(p) +r'.np','rb'))
   HW_MINUS = np.load(file('tmp/HW_MINUS_' + str(p) +r'.np','r'))
   HW_PLUS = np.load(file('tmp/HW_PLUS_' + str(p) +r'.np','r'))
   K = scipy.io.mmread('tmp/K_' + str(p) + '.mtx').tocsc()
   
   #data = dd.io.load('tmp/save_' + str(p) + '.hdf5')
   #A = data['A']
   #K = data['K']
   #P = data['P']
   #HW_MINUS = data['HW_minus']
   #HW_PLUS = data['HW_plus']

    
   TB = options['boundary_temp']
   TL = options['temp']
   TL_new = np.zeros(self.n_el) 
   TB_new = np.zeros(self.n_el) 
   phi_factor = self.dom['phi_dir'][p]/self.dom['d_phi_vec'][p]
   D = np.multiply(TB,HW_MINUS)

   flux = np.zeros((self.n_el,3)) 
   suppression = np.zeros((self.n_mfp,self.n_theta,self.n_phi))
   
   symmetry = 2
   kappa_factor = 3.0/4.0/np.pi*self.mesh.size[0]/self.area_flux
   for t in range(int(self.n_theta/2.0)): #We take into account the symmetry
    theta_factor = self.dom['at'][t] / self.dom['d_theta_vec'][t]
    for m in range(self.n_mfp):
     #----------------------------------------------------------------------------------
     index = m*self.dom['n_theta']*self.dom['n_phi'] +t*self.dom['n_phi']+p
     if self.current_iter == 0:
      F = scipy.sparse.eye(self.n_elems) + theta_factor * self.mfp[m] * A
      lu = splu(F.tocsc())
      self.lu.update({index:lu})
     else:
      lu = self.lu[index] #read previous lu
     #-------------------------------------------------------------------------------

     #CORE--------------------------------------------------------------------------------
     RHS = self.mfp[m]*theta_factor * (P + D) + TL
     temp = lu.solve(RHS)
     TL_new += self.B2[m] * temp * self.dom['d_omega'][t][p]/4.0/np.pi * symmetry
     flux += self.B1[m]*np.outer(temp,self.dom['S'][t][p])
     pre_factor = 0.5*theta_factor/self.mfp[m]*self.dom['d_theta_vec'][t]*self.dom['d_phi_vec'][p]
     suppression[m,t,p] += pre_factor * K.dot(temp).sum()
     if symmetry == 2.0:
      suppression[m,self.n_theta -t -1,p] += suppression[m,t,p]
     

     TB_new += self.B1[m]*self.dom['at'][t]*np.multiply(temp,HW_PLUS)*symmetry
     #------------------------------------------------------------------------------------
   

   suppression *= kappa_factor

   output = {'temp':TL_new,'boundary_temp':TB_new,'suppression':suppression,'flux':flux}

   return output



  def compute_function(self,**argv):
 
   fourier_temp = argv['fourier_temperature']
   max_iter = argv.setdefault('max_bte_iter',10)
   max_error = argv.setdefault('max_bte_error',1e-2)
   error = 2.0 * max_error
   previous_temp = fourier_temp
   previous_boundary_temp = fourier_temp
   previous_gradient_T = argv['T_der']
  
   previous_kappa = 0.0
   self.current_iter = 0
   symmetry = 2 

   #------------------------
   #---------------------------------
   #-----------------------------------
   if MPI.COMM_WORLD.Get_rank() == 0:
    print('    Iter    Thermal Conductivity [W/m/K]      Error')
    print('   ---------------------------------------------------')

   while error > max_error and self.current_iter < max_iter:

    options = {'temp':np.array(previous_temp),'boundary_temp':np.array(previous_boundary_temp)}
    output = {'temp':np.zeros(self.n_el),'boundary_temp':np.zeros(self.n_el),'suppression':np.zeros((self.n_mfp,self.n_theta,self.n_phi)),'flux':np.zeros((self.n_el,3))}

    #solve---
    output = compute_sum(self.solve,self.n_phi,output,options)
    #----------------
    flux = output['flux']
    previous_boundary_temp = output['boundary_temp']
    previous_temp = output['temp']
    directional_suppression = output['suppression']

    kappa = sum([self.B0[m]*directional_suppression[m,:,:].sum() for m in range(self.n_mfp)])
 
    error = abs((kappa-previous_kappa))/kappa
    if MPI.COMM_WORLD.Get_rank() == 0:
     print('{0:7d} {1:20.4E} {2:25.4E}'.format(self.current_iter,kappa*self.kappa_bulk, error))
    
    previous_kappa = kappa
    self.current_iter +=1

   suppression = [directional_suppression[m,:,:].sum() for m in range(self.n_mfp)]

   self.state = {'kappa_bte':kappa*self.kappa_bulk,\
            'directional_suppression':directional_suppression,\
            'suppression_function':suppression,\
            'dom':self.dom,\
            'mfp':self.mfp*1e-9,\
            'bte_temperature':previous_temp,\
            'bte_flux':flux}

    

   if MPI.COMM_WORLD.Get_rank() == 0:
    print('   ---------------------------------------------------')



