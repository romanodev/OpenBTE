from __future__ import absolute_import
#import os
import numpy as np
import scipy.sparse
from scipy.sparse import csc_matrix
from mpi4py import MPI
#import shutil
from scipy.sparse.linalg import splu
from os import listdir
from scipy.sparse import *
from os.path import isfile, join
import time
import sparse
from copy import deepcopy
from collections import namedtuple
from scipy.sparse.linalg import spilu
from scipy.sparse import spdiags
import scipy.io
import deepdish as dd
from .geometry2 import Geometry
from scipy import interpolate
import sparse
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve_triangular
import time
from numpy.testing import assert_array_equal
import pickle
import os
import shutil
import matplotlib.pylab as plt
import opt_einsum as einsum

#levi civita
eijk = np.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

def is_in(myarr, list_arrays):
    return next((True for elem in list_arrays if np.allclose(elem, myarr,rtol=1e-4)), False)

def occurrance(myarr, list_arrays):
    return sum([ int(np.allclose(elem,myarr,rtol=1e-5)) for elem in list_arrays])


def unique_versors(list_arrays):

    a = np.ma.array(list_arrays, mask=False)

    versors_irr = []
    index_list = []
    index_tot = []
    for n,v in enumerate(list_arrays):
     if not n in np.array(index_tot):
      pick = [ int(np.allclose(elem,v,rtol=1e-3)) for elem in list_arrays]   
      indexes = np.nonzero(pick)[0]
    
      if len(indexes) > 0:
       index_tot = np.append(index_tot,indexes)
       index_list.append(indexes)
       versors_irr.append(v)



    return sum([ int(np.allclose(elem,myarr,rtol=1e-5)) for elem in list_arrays])



def log_interp1d(xx, y, kind='linear'):

     if len(y) > 1:
      yy = y.copy()
      scale = min(yy)
      yy -= min(yy)
      yy+= 1e-12
      logx = np.log10(xx)
      logy = np.log10(yy)
      lin_interp = interpolate.interp1d(logx,logy,kind=kind,fill_value='extrapolate')
      log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))  +scale -1e-12
     else:
      log_interp = lambda zz: y[0]
     
     return log_interp

class Solver(object):

  def __init__(self,**argv):

   if 'geometry' in argv.keys():   
     self.mesh = argv['geometry']
     self.mesh._update_data()
   else:  
    #self.mesh = Geometry(model='load',filename = argv.setdefault('geometry_filename','geometry.p'))
    self.mesh = Geometry(model='load',filename = argv.setdefault('geometry_filename','geometry.p'))
   

   self.old_index = -1
   self.n_side_per_elem = len(self.mesh.elems[0])
   self.dim = self.mesh.dim
   self.TT = 0.5*self.mesh.B * np.array([self.mesh.elem_volumes]).T    
   self.ms_error = argv.setdefault('ms_error',0.1)
   self.verbose = argv.setdefault('verbose',True)
   self.n_elems = len(self.mesh.elems)
   self.cache = os.getcwd() + '/.cache'
   self.keep_lu  = argv.setdefault('keep_lu',False) 
   self.alpha = argv.setdefault('alpha',1.0)
 
   if MPI.COMM_WORLD.rank == 0 and argv.setdefault('save_data',True):
    if os.path.exists(self.cache):
        shutil.rmtree(self.cache)
    os.mkdir('.cache')
   MPI.COMM_WORLD.Barrier() 
    
   self.argv = argv
   self.multiscale = argv.setdefault('multiscale',False)
   

   if 'material' in argv.keys():
     self.mat = argv['material'].output
   else:  
     self.mat = pickle.load(open('material.p','rb'))

   self.control_angle =  self.rotate(np.array(self.mat['control_angle']),**argv)
   #self.control_angle_not_int =  self.rotate(np.array(self.mat['control_angle_not_int']),**argv)
   self.kappa_directional =  self.rotate(np.array(self.mat['kappa_directional']),**argv)
   #self.kappa_directional_not_int =  self.rotate(np.array(self.mat['kappa_directional_not_int']),**argv)
   #self.control_angle =  np.array(self.mat['control_angle'])
   #self.kappa_directional =  np.array(self.mat['kappa_directional'])
   #self.angle_plus =  self.rotate(np.array(self.mat['angle_plus']),**argv)
   #self.angle_minus =  self.rotate(np.array(self.mat['angle_minus']),**argv)
   #self.control_angle =  self.rotate(np.array(self.mat['control_angle']),**argv)
   #self.angle_plus =  self.rotate(np.array(self.mat['angle_plus']),**argv)
   #self.angle_minus =  self.rotate(np.array(self.mat['angle_minus']),**argv)
   #self.control_angle = self.mat['control_angle']
   #self.angle_plus = self.mat['angle_plus']
   #self.angle_minus = self.mat['angle_minus']

   

   self.multiscale = argv.setdefault('multiscale',False)


   self.n_parallel = self.mat['n_parallel'] #total indexes
   self.n_serial = self.mat['n_serial'] #total indexes
 


   self.lu = {}
   self.lu_fourier = {}
   self.last_index = -1

   self.kappa_factor = self.mesh.kappa_factor
   if self.verbose: 
    self.print_logo()
    self.print_dof()



   if self.verbose: self.print_bulk_kappa()
   
   self.assemble_fourier()
   
   #---------------------------
   
   #solve the BTE
   self.solve_bte(**argv)
   


   #if MPI.COMM_WORLD.Get_rank() == 0:
   # if os.path.isdir(self.cache):
   #  shutil.rmtree(self.cache)
   #MPI.COMM_WORLD.Barrier() 
    

  def get_material_from_element(self,elem):

   return self.mat_map[self.mesh.elem_region_map[elem]]
  
  def rotate(self,data,**argv):

   dphi = argv.setdefault('rotate_phi',0)
   if not dphi == 0:
     angle_rot = dphi*np.pi/180.0   
     M = np.array([[np.cos(angle_rot),-np.sin(angle_rot)],[np.sin(angle_rot),np.cos(angle_rot)]])
     drot = data.copy()
     for n in range(len(data)):
      drot[n,0:2] = np.dot(M,data[n,0:2])
     return drot
   return data
    


  def get_multiscale_diffusive(self,index,n,SDIFF,TDIFF,TDIFFGrad):

          angle = self.control_angle[index]
          aa = sparse.COO([0,1,2],angle,shape=(3))
          HW_PLUS = sparse.tensordot(self.mesh.CP,aa,axes=1).clip(min=0)
          s = SDIFF[n] * self.mat['mfp'][n]
          temp = TDIFF[n] - self.mat['mfp'][n]*np.dot(self.control_angle[index],TDIFFGrad[n].T)      
          t = temp*self.mat['domega'][index]
          j = np.multiply(temp,HW_PLUS)*self.mat['domega'][index]

          return t,s,j

  def get_antialiasing(self,matrix,versor,AM,AP):

      h = np.linalg.norm(versor)

      #Get delta------------------------------------------------------------------------
      dim = np.shape(matrix)
      angle_versor = -np.arctan2(versor[1],versor[0]) +np.pi/2
      angle_versor = np.round(angle_versor,5)
      if (angle_versor < 0) : angle_versor += 2 * np.pi
      if (angle_versor > 2*np.pi) : angle_versor -= 2 * np.pi

      angle_minus = angle_versor - self.mat['dphi']/2
      angle_plus  = angle_versor + self.mat['dphi']/2
      if (angle_minus < 0) : angle_minus += 2 * np.pi
      if (angle_plus > 2*np.pi) : angle_plus -= 2 * np.pi

      for i,j in zip(matrix.nonzero()[0],matrix.nonzero()[1]):
         normal = matrix[i,j].todense() 
         angle_normal = -np.arctan2(normal[1],normal[0]) +np.pi/2
         if (angle_normal < 0) : angle_normal += 2 * np.pi
         angle_normal = np.round(angle_normal,5)

         #surface tangential----
         angle = angle_normal + np.pi/2
         if (angle < 0) : angle += 2 * np.pi
         if (angle > 2*np.pi) : angle -= 2 * np.pi
         #----------------
         dangle = np.round(angle_versor - angle,5)

         if abs(dangle) < self.mat['dphi']/2:

           p1 = angle_minus; p2 = angle
           AP[i,j] = h*((np.cos(p1) - np.cos(p2))*normal[0] + (np.sin(p2) - np.sin(p1))*normal[1])

           p1 = angle; p2 = angle_plus
           AM[i,j] = h*((np.cos(p1) - np.cos(p2))*normal[0] + (np.sin(p2) - np.sin(p1))*normal[1])
 


  def get_boundary_matrices(self,global_index):


     if not os.path.exists(self.cache + '/Am_' + str(global_index) + '.p') :
      kdir  = self.kappa_directional[global_index]
      k_angle = sparse.COO([0,1,2],self.kappa_directional[global_index],shape=(3))
      
      tmp = sparse.tensordot(self.mesh.CP,k_angle,axes=1)
      AP = tmp.clip(min=0)#.to_scipy_sparse().todok()
      AM = (tmp-AP)#.tocsc().todok() #preserve the sign of AM
      if self.argv.setdefault('antialiasing',False):
       self.get_antialiasing(self.mesh.CP,self.kappa_directional_not_int[global_index],AM,AP)

      #Am = -tmp.clip(max=0)#.todense()
      AP = AP.todense()
      AM = -AM.todense()

      AM.dump(open(self.cache + '/Am_' + str(global_index) + '.p','wb'))
      AP.dump(open(self.cache + '/Ap_' + str(global_index) + '.p','wb'))
     else :
      AM = np.load(open(self.cache +'/Am_' + str(global_index) +'.p','rb'),allow_pickle=True)
      AP = np.load(open(self.cache +'/Ap_' + str(global_index) +'.p','rb'),allow_pickle=True)

     return AM,AP

  def get_bulk_data(self,global_index,TB,TL):

     index_irr = self.mat['temp_vec'][global_index]
     aa = sparse.COO([0,1,2],self.control_angle[global_index],shape=(3))
     mfp   = self.mat['mfp'][global_index]
     irr_angle = self.mat['angle_map'][global_index]

     if not irr_angle == self.old_index:
       self.old_index = irr_angle

       if not os.path.exists(self.cache + '/A_' + str(irr_angle) + '.npz'):

        test2  = sparse.tensordot(self.mesh.N,aa,axes=1)
        AP = test2.clip(min=0)
        AM = (test2 - AP)
        AP = spdiags(np.sum(AP,axis=1).todense(),0,self.n_elems,self.n_elems,format='csc')
        self.mesh.B = self.mesh.B.tocsc()
        self.P = np.sum(np.multiply(AM,self.mesh.B),axis=1).todense()
        tmp = sparse.tensordot(self.mesh.CM,aa,axes=1)
        HW_PLUS = tmp.clip(min=0)
        self.HW_MINUS = (HW_PLUS-tmp)
        BP = spdiags(np.sum(HW_PLUS,axis=1).todense(),0,self.n_elems,self.n_elems,format='csc')
        self.A =  AP + AM + BP
     
        self.P.dump(self.cache + '/P_' + str(irr_angle) + '.p')
        sparse.io.save_npz(self.cache + '/A_' + str(irr_angle) + '.npz',self.A)
        sparse.io.save_npz(self.cache + '/HW_MINUS_' + str(irr_angle) + '.npz',self.HW_MINUS)

       else:  
        self.P = np.load(open(self.cache +'/P_' + str(irr_angle) +'.p','rb'),allow_pickle=True)
        self.A = sparse.load_npz(self.cache + '/A_' + str(irr_angle) + '.npz')
        self.HW_MINUS = sparse.load_npz(self.cache + '/HW_MINUS_' + str(irr_angle) + '.npz')

     boundary = np.sum(np.multiply(TB,self.HW_MINUS),axis=1).todense()
     RHS = mfp * (self.P + boundary) + TL[index_irr]  


     #Add connection-----
     #RHS += self.TL_old[global_index] - self.temp_old[global_index] - self.delta_old[global_index]
     #-------------------
 
     #else:
     #  RHS = mfp * (boundary) + TL[index_irr]   

     F = scipy.sparse.eye(self.n_elems,format='csc') + self.A * mfp
     lu = splu(F.tocsc())
     return lu.solve(RHS)




      
  def solve_bte(self,**argv):

   comm = MPI.COMM_WORLD  
   rank = comm.rank 

   if rank == 0 and self.verbose:
     print('    Iter    Thermal Conductivity [W/m/K]      Error        Diffusive  -  BTE  -  Ballistic')
     print('   ---------------------------------------------------------------------------------------')

   n_mfp = self.mat['n_serial']
   temp_vec = self.mat['temp_vec']
   nT = np.shape(self.mat['CollisionMatrix'])[0]
   kdir = self.kappa_directional

   if self.argv.setdefault('load_state',False):
     data  = dd.io.load(self.argv.setdefault('filename','solver.hdf5'))
     error_vec = data['error_vec']
     kappa_vec = data['kappa_vec']
     ms_vec = data['ms_vec']
     TL = data['TL']
     TB = data['TB']
     temp_fourier = data['temp_fourier']
     temp_fourier_grad = data['temp_fourier_grad']
     if rank == 0: 
      for n in range(len(ms_vec)):
       print(' {0:7d} {1:20.4E} {2:25.4E} {3:10.2F} {4:10.2F} {5:10.2F}'.format(n+1,kappa_vec[n],error_vec[n],ms_vec[n][0],ms_vec[n][1],ms_vec[n][2]))

   else:          
     #fourier first guess----
     if rank == 0: 
      kappa_fourier,temp_fourier,temp_fourier_grad = self.get_diffusive_suppression_function(self.mat['kappa_bulk_tot'])
      kappa_fourier = np.array([kappa_fourier])
      data = {'kappa_fourier':kappa_fourier,'temp_fourier':temp_fourier,'temp_fourier_grad':temp_fourier_grad}
     else: data = None   
     data = comm.bcast(data,root=0)
     comm.Barrier()
     temp_fourier = data['temp_fourier']
     temp_fourier_grad = data['temp_fourier_grad']
     kappa_fourier = data['kappa_fourier']
     TL = np.tile(temp_fourier,(nT,1))
     TB = np.tile(temp_fourier,(self.n_side_per_elem,1)).T
     #----------------------------------------------------------------
     #kappa_vec = [float(kappa_fourier)]
     kappa_vec = [0]
     error_vec = [1.0]
     ms_vec = [[1,0,0]]
     if rank == 0 and self.verbose:
       print(' {0:7d} {1:20.4E} {2:25.4E} {3:10.2F} {4:10.2F} {5:10.2F}'.format(1,float(kappa_fourier),1,1,0,0))
     #---------------------------

   #Save if only Fourier---
   if self.argv.setdefault('only_fourier',False):
     data_save = {'flux_fourier':-self.mat['kappa_bulk_tot']*temp_fourier_grad,'temp_fourier':temp_fourier,'kappa_fourier':kappa_fourier}
     dd.io.save(self.argv.setdefault('filename','solver.hdf5'),data_save)

   #Initialize data-----
   n_iter = len(kappa_vec)
   self.TL_old = TL.copy()
   self.delta_old = np.zeros_like(TL)
   TB_old = TB.copy()
 

   self.temp_old = self.TL_old.copy()
   #-------------------------------
   

   while n_iter < argv.setdefault('max_bte_iter',10) and \
          error_vec[-1] > argv.setdefault('max_bte_error',1e-2):

    self.n_iter = n_iter
    #Compute diffusive approximation---------------------------          
    if self.multiscale:

      TDIFF,TDIFFp = np.zeros((2,n_mfp,self.n_elems))          
      TDIFFGrad = np.zeros((n_mfp,self.n_elems,3))          
      TDIFFGradp = np.zeros((n_mfp,self.n_elems,3))          
      Jx,Jxp = np.zeros((2,n_mfp,self.n_elems))          
      Jy,Jyp = np.zeros((2,n_mfp,self.n_elems))          
      Jz,Jzp = np.zeros((2,n_mfp,self.n_elems))          
      block = self.mat['n_serial'] // comm.size + 1

      for kk in range(block):

       n = rank*block + kk   
       if n < n_mfp :
           
        kappa_value = self.mat['mfp'][n] * self.mat['mfp'][n]/3.0 #The first index is MFP anyway
        G = self.mat['mfp'][n]/2.0*np.ones(self.n_side_per_elem) #we have to define this for each side
        dummy,TDIFFp[n],TDIFFGradp[n] = self.get_diffusive_suppression_function(kappa_value,TL=TL[0],TB=TB,G = G)
        
      comm.Allreduce([TDIFFp,MPI.DOUBLE],[TDIFF,MPI.DOUBLE],op=MPI.SUM)
      comm.Allreduce([TDIFFGradp,MPI.DOUBLE],[TDIFFGrad,MPI.DOUBLE],op=MPI.SUM)

    #Print diffusive Kappa---------
    TL2p,TL2 = np.zeros((2,nT,self.n_elems))
    Tp,T = np.zeros((2,self.n_elems))
    Vp,V = np.zeros((2,self.n_elems,3))
    Jp,J = np.zeros((2,self.n_elems,3))
    TBp_minus,TB_minus = np.zeros((2,self.n_elems,self.n_side_per_elem))
    TBp_plus,TB_plus = np.zeros((2,self.n_elems,self.n_side_per_elem))
    ndifp = np.zeros(1)
    ndif = np.zeros(1)
    nbal = np.zeros(1)
    nbalp = np.zeros(1)
    (etavp,etav) = np.zeros((2,nT))
    (etadp,etad) = np.zeros((2,nT))
    #experimental---
    delta = np.zeros_like(TL)
    deltap = np.zeros_like(TL)
    temp_vec = np.zeros((self.n_parallel * self.n_serial,self.n_elems))
    tempp_vec = np.zeros((self.n_parallel * self.n_serial,self.n_elems))
    #--------------

    K2p = np.zeros(1)
    K2 = np.zeros(1)
    block = self.n_parallel // comm.size + 1

    for kk in range(block):
     index = rank*block + kk  
     if index < self.n_parallel :
      #print(index)
      mfp_plot = []
      #-------------------------------------------------------   
      idx = [self.n_serial]
      if self.multiscale:
       #Compute ballistic regime---
       a = time.time()
       temp_bal = self.get_bulk_data(index * self.n_serial + self.n_serial -1,TB,TL)
       eta_bal = np.ones(self.n_serial) * self.mesh.B_with_area_old.dot(temp_bal).sum()
       #-------------------------------------

       #Compute diffusive regime----------------------------------
       eta_diff= []
       for n in range(n_mfp):
        m = self.mat['mfp'][n]   
        a = np.array([m])
        vv = TDIFF[n] -  self.mat['mfp'][n]*np.einsum('ci,i->c',TDIFFGrad[n],self.control_angle[index*self.n_serial + 1])
        eta_diff.append(self.mesh.B_with_area_old.dot(vv).sum())
       #----------------------------------------------------------

       #Compute the intersection index-----
       idx   = np.argwhere(np.diff(np.sign(eta_diff - eta_bal))).flatten()
       if len(idx) == 0: idx = [self.n_serial-1]
      #------------------------------------------------------

      fourier = False
      eta_vec = np.zeros(self.n_serial)
      for n in range(self.n_serial)[idx[0]::-1]:
          
        global_index = index * self.n_serial + n 
        if fourier == True:
         ndifp[0] += 1   
         temp = TDIFF[n]
         eta = eta_diff[n]
        else:
         temp = self.get_bulk_data(global_index,TB,TL)
       
         #-new-----------
         #delta_tmp = self.get_bulk_data(global_index,TB,DeltaTL)
         tempp_vec[global_index] = temp
         #temp = self.temp[global_index]
         #------------------------------------- 

         #print(time.time()-a)
         eta = self.mesh.B_with_area_old.dot(temp).sum()
         if self.multiscale:
           if abs(eta_diff[n] - eta)/abs(eta) < 1e-2:
            fourier = True  
         
        eta_vec[n] = eta
        (Am,Ap) = self.get_boundary_matrices(global_index)
        TBp_minus += Am 
        TBp_plus  += np.einsum('es,e->es',Ap,temp)
        TL2p += np.outer(self.mat['CollisionMatrix'][:,global_index],temp)    
        Tp+= temp*self.mat['TCOEFF'][global_index]
        kdir = self.mat['kappa_directional'][global_index]
        K2p += np.array([eta*np.dot(kdir,self.mesh.applied_grad)])
        Jp += np.outer(temp,kdir)*1e9

        tempp_vec[global_index] = temp
        #correction--

        deltap -= np.outer(self.mat['invH'][:,global_index],np.einsum('j,cj->c',self.mat['FRTA'][global_index],self.mesh.compute_grad(temp))) 
        #-----------

      #Ballistic component
      ballistic = False
      for n in range(self.n_serial)[idx[0]+1:]:

        global_index = index * self.n_serial + n 
        if ballistic == True:
         nbalp[0] += 1   
         temp = temp_bal
         eta = eta_bal[n]
        else:
         temp = self.get_bulk_data(global_index,TB,TL)
         self.temp[global_index] = temp
         eta = self.mesh.B_with_area_old.dot(temp).sum()
         if self.multiscale:
          if abs(eta_bal[n] - eta)/abs(eta) <1e-2 :
            ballistic = True  

        (Am,Ap) = self.get_boundary_matrices(global_index)
        TBp_minus += Am 
        TBp_plus  += np.einsum('es,e->es',Ap,temp)
        TL2p += np.outer(self.mat['CollisionMatrix'][:,global_index],temp)    
        Tp+= temp*self.mat['TCOEFF'][global_index]
        kdir = self.mat['kappa_directional'][global_index]
        K2p += np.array([eta*np.dot(kdir,self.mesh.applied_grad)])
        Jp += np.outer(temp,kdir)*1e9

    comm.Allreduce([K2p,MPI.DOUBLE],[K2,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce([TL2p,MPI.DOUBLE],[TL2,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce([TBp_minus,MPI.DOUBLE],[TB_minus,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce([TBp_plus,MPI.DOUBLE],[TB_plus,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce([Tp,MPI.DOUBLE],[T,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce([Jp,MPI.DOUBLE],[J,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce([ndifp,MPI.DOUBLE],[ndif,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce([nbalp,MPI.DOUBLE],[nbal,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce([deltap,MPI.DOUBLE],[delta,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce([tempp_vec,MPI.DOUBLE],[temp_vec,MPI.DOUBLE],op=MPI.SUM)

    kappa_vec.append(abs(K2[0] * self.kappa_factor))
    error_vec.append(abs(kappa_vec[-1]-kappa_vec[-2])/abs(max([kappa_vec[-1],kappa_vec[-2]])))
    #----------------

    #Upate lattice and boundary temperature
    
    #if rank == 0:
    TB_new = TB_plus.copy()
    for i in range(3):
      for n in range(len(TB_plus.T[i])):
       if not (TB_plus[n,i] == 0):
         TB_new[n,i] /= TB_minus[n,i]

    TB = self.alpha * TB_new + (1-self.alpha)*TB_old; TB_old = TB.copy()

    
    #T experimental----
    #if self.argv.setdefault('synthetic',False):
    # if rank==0:
    #   JNonFourier = -J*1e-9 + self.mat['kappa_bulk_tot'] * self.mesh.compute_grad(TL_old[0])
    #   heat_source = self.mesh.compute_divergence(JNonFourier,add_jump=False)
    #   dummy,t,j = self.get_diffusive_suppression_function(kappa = self.mat['kappa_bulk_tot'],heat_source= heat_source)
    #   data = {'T':t}
    # else: data = None
    # data = comm.bcast(data,root=0)
    # TL = [data['T']]
    #else: 
    #print(min(TL2[0]),max(TL2[0])) 
    TL = self.alpha * TL2.copy() + (1-self.alpha)*self.TL_old; 
    #DeltaTL = TL-TL_old; 
    self.TL_old = TL.copy()

    #experimental--
    self.temp_old = temp_vec.copy()
    self.delta_old = delta.copy()
     #---

    n_iter +=1   
    #-----



    #Thermal conductivity   
    if rank==0 and self.verbose:
      ndif = ndif[0]/(self.n_serial*self.n_parallel)
      nbal = nbal[0]/(self.n_serial*self.n_parallel)
      nbte = 1- ndif - nbal
      ms_vec.append([ndif,nbte,nbal])
      kappa_current = kappa_vec[-1]
      #kappa_current = np.sum(kappa_vec)
      print(' {0:7d} {1:20.4E} {2:25.4E} {3:10.2F} {4:10.2F} {5:10.2F}'.format(n_iter,kappa_current,error_vec[-1],ndif,nbte,nbal))
      data = {'kappa_vec':kappa_vec,'temperature':T,'pseudogradient':self.mesh.compute_grad(T),'flux':J,'temp_fourier':temp_fourier,'flux_fourier':-self.mat['kappa_bulk_tot']*temp_fourier_grad}
      if self.argv.setdefault('save_state',False):
          data.update({'TB':TB,'TL':TL,'error_vec':error_vec,'ms_vec':ms_vec,'temp_fourier_grad':temp_fourier_grad})  

    if rank == 0:
     dd.io.save(self.argv.setdefault('filename','solver.hdf5'),data)
    else: data = None
    self.state =  MPI.COMM_WORLD.bcast(data,root=0)


   if rank==0 and self.verbose:
     print('   ---------------------------------------------------------------------------------------')
     print(' ')
    




  def print_bulk_kappa(self):
      
    if MPI.COMM_WORLD.Get_rank() == 0:
     if self.verbose:
      #print('  ')
      #for region in self.region_kappa_map.keys():
       print('Kappa bulk: ')
       print('{:8.2f}'.format(self.mat['kappa_bulk_tot'])+ ' W/m/K')    
       print(' ')
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
    FF = np.zeros((self.n_elems,self.n_side_per_elem))
    boundary_mask = np.zeros(self.n_elems)
    for side in self.mesh.side_list['Boundary']:
     elem = self.mesh.side_elem_map[side][0]
     side_index = np.where(np.array(self.mesh.elem_side_map[elem])==side)[0][0]
     vol = self.mesh.get_elem_volume(elem)
     area = self.mesh.get_side_area(side)
     FF[elem,side_index] = area/vol
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

    #if self.argv.setdefault('compute_vorticity',False):
    #     grad = self.mesh.compute_grad(temp).T
    #     Vp += opt_einsum('ijk,jc,k->ci',eijk,grad,kdir)


    #if self.argv.setdefault('compute_vorticity',False):
    #   J = -self.mat['kappa_bulk_tot']*flux_fourier
    #   Jx = self.mesh.compute_grad(J.T[0]).T
    #   Jy = self.mesh.compute_grad(J.T[1]).T
    #   Jz = self.mesh.compute_grad(J.T[2]).T
    #   Hessian = [Jx,Jy,Jz]
    #   Vf = einsum('ijk,kjc->ci',eijk,Hessian)
    #else:
    #   Vf = np.zeros((self.n_elems,3))

  def get_diffusive_suppression_function(self,kappa,**argv):



    A  = kappa * self.F 
    B  = kappa * self.B

    if 'G' in argv.keys(): 
     G = argv['G']   
     TB = argv['TB']   
     tmp = np.dot(self.FF,G) #total flux from boundaries
     A += spdiags(tmp,0,self.n_elems,self.n_elems) 
     B += np.einsum('ci,ci,i->c',self.FF,TB,G)
    if 'heat_source' in argv.keys(): 
      B += argv['heat_source']


    if 'TL' in argv.keys():
      A +=  csc_matrix(scipy.sparse.eye(self.n_elems)) 
      B += argv['TL']
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
        if not 'TL' in argv.keys():
         RHS[10] = 0.0      
        temp = SU.solve(RHS)
        temp = temp - (max(temp)+min(temp))/2.0
        (C,grad) = self.compute_non_orth_contribution(temp)
        
        kappa_eff = self.compute_diffusive_thermal_conductivity(temp)*kappa
        error = abs((kappa_eff - kappa_old)/kappa_eff)
        kappa_old = kappa_eff
        n_iter +=1

    s = kappa_eff#/kappa
    
    #ind = self.mesh.B_with_area_old.dot(temp).sum()

    return s,temp.copy(),grad



  '''
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
        
      FLUXp[index] = np.array([-self.mat['kappa_bulk_tot']*tmp for k,tmp in enumerate(flux)])
      #FLUXp[index] = flux.T
      TLp[index] = temp
     
        
    comm.Allreduce([KAPPAp,MPI.DOUBLE],[KAPPA,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce([TLp,MPI.DOUBLE],[TLtot,MPI.DOUBLE],op=MPI.SUM) #to be changed
    comm.Allreduce([FLUXp,MPI.DOUBLE],[FLUX,MPI.DOUBLE],op=MPI.SUM)  
    
    
    #if argv.setdefault('verbose',True) and rank==0:

     #print('  ')
     #print('Thermal Conductivity: '.ljust(20) +  '{:8.2f}'.format(KAPPAp.sum())+ ' W/m/K')
     #print('  ')
    


    return {'kappa_fourier':KAPPA,'temperature_fourier_gradient':FLUX,'temperature_fourier':TLtot}
 '''
      
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
     print('Space DOFs:      ' + str(self.n_elems))
     #print('Azimuthal angles:  ' + str(self.mat['n_theta']))
     #print('Polar angles:      ' + str(self.mat['n_phi']))
     print('Momentum DOFs:   ' + str(self.mat['n_serial']*self.mat['n_parallel']))
     #print('Bulk Thermal Conductivity:   ' + str(round(self.mat['kappa_bulk_tot'],4)) +' W/m/K')
     print(' ')
     
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
