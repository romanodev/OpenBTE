from __future__ import absolute_import
#import os
import numpy as np
#from scipy.sparse import *
import scipy.sparse as sp
#from GBQsparse import MSparse
from .GPUSolver import *
from scipy import *
#import scipy.sparse
from scipy.sparse import csc_matrix
#import sparseqr
from mpi4py import MPI
from scipy.sparse.linalg import splu
#import pycuda.gpuarray as gpuarray
#import pycuda.driver as cuda
from itertools import permutations
#import pycuda.autoinit
from os import listdir
#from scipy.sparse import *
from os.path import isfile, join
import time
from copy import deepcopy
from collections import namedtuple
from scipy.sparse import spdiags
import sparse
import scipy.io
import deepdish as dd
from .geometry_gpu import GeometryGPU as Geometry
from scipy import interpolate
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve_triangular
import time
from numpy.testing import assert_array_equal
import pickle
import os
import shutil
import matplotlib.pylab as plt
#import torch
#import skcuda.linalg as skla
#import skcuda.misc as misc
#import cupy as cp
from numba import jit


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


#def get_specular_direction(self,i):


@jit
def matmul_jit(A, B):
    """Perform square matrix multiplication of out = A * B
    """
    out = np.empty((A.shape[0],B.shape[1]), A.dtype)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            tmp = 0.
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]
            out[i, j] = tmp
    return out

'''
@jit(nopython=True)
def get_next_data(X,S,Sx,BB,Gbm,Gtbp,Gtbm,eb,sb):

      kappa = np.sum(np.multiply(BB,X))

      #TL = np.matmul(S,X)#*alpha + (1-alpha)*TL_old

      #matrix multiplication---
      TL = np.empty((S.shape[0],X.shape[1]), S.dtype)
      Bm = np.zeros_like(TL)
      for i in range(TL.shape[0]):
        for n,(l,_) in enumerate(zip(eb,sb)):
         Bm[i,l]  += Gbm[i,n] * X[i,l] * Gtbp[i,n] / Gtbm[n] #the sum is for each side
         for j in range(TL.shape[1]):
            tmp = 0.
            for k in range(S.shape[1]):
                tmp += S[i, k] * X[k, j]
            TL[i, j] = tmp


      #TL_old = TL.copy()
      #Compute TB----
      #for n,(i,j) in enumerate(zip(eb,sb)):
      #  Bm[:,i]  += Gbm[:,n] * np.dot(X[:,i],Gtbp[:,n])/Gtbm[n] #the sum is for each side
      return TL,Bm,kappa

'''

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

class SolverGPU(object):

  def __init__(self,**argv):

   if 'geometry' in argv.keys():  
     self.mesh = argv['geometry']
     #self.mesh._update_data()
   else:  
    self.mesh = Geometry(model='load',filename_geometry = argv.setdefault('filename_geometry','geometry.p'))
   


   if len(self.mesh.elems[0]) == 4:
         self.structured=True
   else:    
         self.structured=False

   self.old_index = -1
   self.n_side_per_elem = len(self.mesh.elems[0])
   self.dim = self.mesh.dim
   #self.TT = 0.5*self.mesh.B * np.array([self.mesh.elem_volumes]).T    
   self.ms_error = argv.setdefault('ms_error',0.1)
   self.verbose = argv.setdefault('verbose',True)
   self.n_elems = len(self.mesh.elems)
   self.cache = os.getcwd() + '/.cache'
   self.keep_lu  = argv.setdefault('keep_lu',False) 
   self.alpha = argv.setdefault('alpha',1.0)
   self.save_data = argv.setdefault('save_data',True)
   self.save_state = argv.setdefault('save_state',True)
 
   if MPI.COMM_WORLD.rank == 0 and self.save_data:
    if os.path.exists(self.cache):
        shutil.rmtree(self.cache)
    os.mkdir('.cache')
   MPI.COMM_WORLD.Barrier() 
   
   
   self.argv = argv
   self.multiscale = argv.setdefault('multiscale',False)
  

   if 'material' in argv.keys():
     self.mat = argv['material'].state
   else: 

    self.mat = []
    if 'matfiles' in argv.keys():
     for mm in argv['matfiles']:
       self.mat.append(pickle.load(open(mm,'rb')))
    else:     
      self.mat = [pickle.load(open('material.p','rb'))]


   #compute elem kappa map
   self.elem_mat_map = self.mesh.elem_mat_map

   #create_elem_kappa_map
   self.elem_kappa_map = {}
   for e in self.elem_mat_map.keys():
     self.elem_kappa_map.update({e:self.mat[self.mesh.elem_mat_map[e]]['kappa_bulk_tot']})

   if argv.setdefault('kappa_from_patterning',False):
    self.elem_kappa_map = self.mesh.elem_kappa_map
   else :   
    self.compute_kappa_map()

   self.control_angle =  self.rotate(np.array(self.mat[0]['control_angle']),**argv)
   self.kappa_directional =  self.rotate(np.array(self.mat[0]['kappa_directional']),**argv)
   self.i = self.mesh.i
   self.j = self.mesh.j
   self.k = self.mesh.k
   self.ip = self.mesh.ip
   self.jp = self.mesh.jp
   self.dp = self.mesh.dp
   self.pv = self.mesh.pv
   self.eb = self.mesh.eb
   self.sb = self.mesh.sb
   self.db = self.mesh.db
   self.dbp = self.mesh.dbp
   self.eft = self.mesh.eft
   self.sft = self.mesh.sft
   self.dft = self.mesh.dft
   #self.mfp  = self.mat[0]['mfp']
 
   #(a,b) = np.shape(self.kappa_directional)
   #kappa = 0
   #for i in range(a):
   #  kappa += self.kappa_directional[i,0] * self.control_angle[i,0] * self.mat[0]['mfp'][i]
   #print(kappa)
   #quit()

   self.lu = {} #keep li

   #self.MATRIX = {} #keep MATRIX
   #self.RHS = {} #keep MATRIX

   self.multiscale = argv.setdefault('multiscale',False)
   self.keep_lu = argv.setdefault('keep_lu',False)

   self.n_parallel = self.mat[0]['n_parallel'] #total indexes
   self.n_serial = self.mat[0]['n_serial'] #total indexes
   self.n_index = self.n_parallel * self.n_serial
    
   self.lu_fourier = {}
   self.last_index = -1

   self.kappa_factor = self.mesh.kappa_factor
   if self.verbose: 
    self.print_logo()
    self.print_dof()
   
   if len(self.mesh.side_list['Interface']) > 0:
    self.compute_transmission_coefficients() 

   self.build_kappa_mask()
   #if self.verbose: self.print_bulk_kappa()
   
   self.assemble_fourier_new()

   if self.multiscale:
    self.assemble_modified_fourier()

   #---------------------------
  
 
   #solve the BTE
   #self.solve_bte(**argv)
   #self.solve_bte_cg(**argv)
   self.solve_bte_working_3(**argv)
   #self.solve_bte_working2(**argv)

   if MPI.COMM_WORLD.Get_rank() == 0:
    if os.path.isdir(self.cache):
     shutil.rmtree(self.cache)
   MPI.COMM_WORLD.Barrier() 
  
  

  def compute_kappa_map(self):

   kappa_matrix = self.mat[0]['kappa_bulk_tot']
   self.elem_kappa_map = {}
   for g in self.mesh.l2g:
     self.elem_kappa_map[g] = kappa_matrix*np.eye(3)

   #kappa_inclusion = self.mat['kappa_inclusion']

   #self.elem_kappa_map = {}
   #for elem in self.mesh.region_elem_map['Matrix']:
   #  self.elem_kappa_map[elem] = kappa_matrix*np.eye(3)

   #if 'Inclusion' in self.mesh.region_elem_map.keys():   
   #  for elem in self.mesh.region_elem_map['Inclusion']:
   #   self.elem_kappa_map[elem] = kappa_inclusion*np.eye(3)

  def rotate2D(self,theta):

   rotMatrix = np.array([[np.cos(theta), -np.sin(theta), 0], 
                         [np.sin(theta),  np.cos(theta),0],[0,0,0]])

   return rotMatrix

  #def get_reflected_index(self,normal,rangle)):



    #P = np.eye(3)-2*np.outer(normal,normal)
    #direction_specular = np.dot(P,angle)
    #print(direction,direction_specular,normal)
    #quit()


  def compute_transmission_coefficients(self):

   #compute all angles
   dirs = []
   for index in range(self.n_parallel):
     direction = self.control_angle[index*self.n_serial]
     direction /= np.linalg.norm(direction)
     dirs.append(direction)
  
   self.transmission = {  side:np.zeros((len(dirs),len(dirs)))  for side in self.mesh.side_list['Interface']}
   self.reflection = {  side:np.zeros((len(dirs),len(dirs)))  for side in self.mesh.side_list['Interface']}

   #-------
   for side in self.mesh.side_list['Interface']:
    (i,j) = self.mesh.side_elem_map[side]
    nij = self.mesh.get_normal_between_elems(i,j)
    vi = self.mat[self.elem_mat_map[i]]['other_properties']['v']
    vj = self.mat[self.elem_mat_map[j]]['other_properties']['v']
    rhoi = self.mat[self.elem_mat_map[i]]['other_properties']['rho']
    rhoj = self.mat[self.elem_mat_map[j]]['other_properties']['rho']
    zi = rhoi*vi
    zj = rhoj*vj

    for index_1,dir_1 in enumerate(dirs):

     if np.dot(dir_1,nij) > 0:    

      theta_c = np.arcsin(min([vj,vi])/vi)
      theta_1 = self.get_angle(nij,dir_1) #refracted angle
      dir_2 = np.dot(self.rotate2D(2*theta_1 + np.pi),dir_1) #specular direction---
      index_2 = np.argmax([np.dot(dir_2,d) for d in dirs])
      if theta_1 > theta_c: #reflective index
        self.reflection[side][index_2,index_1] = 1
      else: #refracted index
        theta_3 = np.arcsin(vi/vj*np.sin(theta_1))        
        dir_3 = np.dot(self.rotate2D(-theta_1+np.pi+theta_3),dir_1) #specular direction---
        index_3 = np.argmax([np.dot(dir_3,d) for d in dirs])
        tij = 4*zi*zj*np.cos(theta_1)*np.cos(theta_3)/(pow(zi*np.cos(theta_3)+zj*np.cos(theta_1),2))
        self.transmission[side][index_3,index_1] = tij
        
        #this is the component partially reflected back 
        self.reflection[side][index_2,index_1] = 1-tij



  def get_angle(self,normal,direction):     

      angle_1 = np.arctan2(normal[1], normal[0])
      angle_2 = np.arctan2(direction[1], direction[0])

      angle = np.arctan2(normal[1], normal[0]) - np.arctan2(direction[1], direction[0])

      if (angle > np.pi) :
       angle -= 2 * np.pi
      elif angle <= -np.pi :
       angle += 2 * np.pi
      return angle


  def get_material_from_element(self,elem):

   return self.elem_kappa_map[elem]
 

 
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
      #if self.argv.setdefault('antialiasing',False):
      # self.get_antialiasing(self.mesh.CP,self.kappa_directional_not_int[global_index],AM,AP)
      AP = AP.todense()
      AM = -AM.todense()

      #Compute matrices for the fixed-temperature boundaries---- To compute thermal conductivities
      #tmp = sparse.tensordot(self.mesh.BN,k_angle,axes=1)
      #BNP = tmp.clip(min=0)
      #BNM = BNP-tmp
      #BNP = BNP.todense()
      #BNM = -BNM.todense()
      #---------------------------------------------------------

      if self.save_data:
       AM.dump(open(self.cache + '/Am_' + str(global_index) + '.p','wb'))
       AP.dump(open(self.cache + '/Ap_' + str(global_index) + '.p','wb'))
       #BNP.dump(open(self.cache + '/BNP_' + str(global_index) + '.p','wb'))
       #BNM.dump(open(self.cache + '/BNM_' + str(global_index) + '.p','wb'))
     else :
      AM = np.load(open(self.cache +'/Am_' + str(global_index) +'.p','rb'),allow_pickle=True)
      AP = np.load(open(self.cache +'/Ap_' + str(global_index) +'.p','rb'),allow_pickle=True)
     # BNM = np.load(open(self.cache +'/BNM_' + str(global_index) +'.p','rb'),allow_pickle=True)
     # BNP = np.load(open(self.cache +'/BNP_' + str(global_index) +'.p','rb'),allow_pickle=True)

     return AM,AP#,BNM,BNP



  def solve_bte_working2(self,**argv):

     nT = np.shape(self.mat[0]['B'])[0]
     kappa_fourier,temp_fourier,temp_fourier_grad,temp_fourier_int,flux_fourier_int = self.get_diffusive_suppression_function()
     kappa_fourier = np.array([kappa_fourier])
     Tnew = temp_fourier.copy()
     TB = np.tile(temp_fourier,(self.n_side_per_elem,1)).T
     TB_new = np.array([temp_fourier[i] for n,(i,j) in enumerate(zip(self.eb,self.sb))])

     TL = np.zeros((nT,self.mesh.nle))

     W = np.load('W.dat.npy')
     a = np.diag(W)
     s = np.sqrt(a)
     Wod = W-np.diag(a)
     g  = W.sum(axis=0)
     S = np.outer(g,g)/np.sum(g) - Wod

     A = np.outer(g,g)/np.sum(g) - W

     sigma = np.load(open('sigma.dat','rb'),allow_pickle=True)*1e9
     F = (sigma.T/a).T
     Sx = sigma[:,0]

     #Solve Bulk
     G = np.einsum('qj,jn->qn',sigma,self.k,optimize=True)
     Gp = G.clip(min=0)
     Gm = G.clip(max=0)
     D = np.zeros((self.n_index,self.mesh.nle),dtype=float64)
     for n,i in enumerate(self.i): 
       D[:,i] += Gp[:,n]
  
     D += np.tile(a,(self.mesh.nle,1)).T

     #Compute boundary---
     if len(self.eb) > 0:
      Gb = np.einsum('qj,jn->qn',sigma,self.db,optimize=True)
      Gbp = Gb.clip(min=0)
      Gbm = Gb.clip(max=0)
      Bp = np.zeros((self.n_index,self.mesh.nle),dtype=float64)
      for n,i in enumerate(self.eb): 
        Bp[:,i] += Gbp[:,n]
      D += Bp

     scale = np.max(D)
     #Compute super lu---
     lu = get_lu(self.i,self.j,Gm/scale,D/scale,self.mesh.nle)   

     #Boundary----------
     Bm = np.zeros((self.n_index,self.mesh.nle))
     if len(self.eb) > 0:
      for n,(i,j) in enumerate(zip(self.eb,self.sb)):
       Bm[:,i] -= TB[i,j]*Gbm[:,n]

     #Periodic------------------
     P = np.zeros((self.n_index,self.mesh.nle))
     for n,(i,j) in enumerate(zip(self.i,self.j)):
       P[:,i] += Gm[:,n]*self.mesh.B[i,j]

     #Initial lattice temperature----
     TL = np.einsum('q,c->qc',a,Tnew)
     #TL = np.zeros_like(Tnew)

     #TB--------------------------------
     if len(self.eb) > 0:
      Gtb  = np.einsum('qj,jn->qn',sigma,self.dbp,optimize=True)
      Gtbp = Gtb.clip(min=0)
      Gtbm = Gtb.clip(max=0).sum(axis=0)

     As = np.einsum('ij,i,j->ij',A,1/s,1/s)
     Ss = np.einsum('ij,i,j->ij',S,1/s,1/s)

     #---------------------------------------------
     DeltaTot = np.zeros_like(TL)
     for kk in range(5):
      #Sub CG loop-
      if kk > 0:
       X1 = X0 + (DeltaTot.T/a).T
       kappa = np.einsum('cd,qd,q->',self.mesh.B_with_area_old.todense(),X1,sigma[:,0],optimize=True)*self.kappa_factor*1e-18
       print('Ana:', kappa)

      X = solve_from_lu(lu, (P + TL + Bm + DeltaTot)/scale)
      if kk == 0:
       X0 = X.copy()

      kappa = np.einsum('cd,qd,q->',self.mesh.B_with_area_old.todense(),X,sigma[:,0],optimize=True)*self.kappa_factor*1e-18
      print('num', kappa)
      Xs = (X.T*s).T
      TLs = ((Bm.T + TL.copy().T + DeltaTot.T)/s).T

      if kk == 0:
       #scaling
       r = np.dot(Ss,Xs)-TLs
       p = -r
       r_k_norm = np.array([ np.dot(r[:,i],r[:,i]) for i in range(self.mesh.nle)])
      else:
       p = p_restart
       r_k_norm = r_k_norm_restart

      print(' ')
      print('Entering CG loop')
      kappa_old = 0
      DeltaT = np.zeros_like(TL)
      for i in range(6):
       Ap = np.array([np.dot(As,p[:,i]) for i in range(self.mesh.nle)]).T
       pAp =np.array([ np.dot(p[:,i],Ap[:,i]) for i in range(self.mesh.nle)])
       alpha = r_k_norm / pAp
       DeltaT += alpha*p
       Xs += alpha * p
       kappa = np.einsum('cd,qd,q->',self.mesh.B_with_area_old.todense(),((Xs.T)/s).T,sigma[:,0],optimize=True)*self.kappa_factor*1e-18
       error = abs((kappa-kappa_old))/abs(kappa)
       print(kappa,error)
       kappa_old = kappa
       #if error < 0.01:
         #X = ((Xs.T)/s).T 
         #DeltaT = ((DeltaT.T)*s).T 
         #DeltaT = ((DeltaT.T)*a).T 
       #  break  
       r += alpha * Ap
       r_kplus1_norm = np.array([ np.dot(r[:,i],r[:,i])           for i in range(self.mesh.nle)])
       beta = r_kplus1_norm / r_k_norm
       r_k_norm = r_kplus1_norm
       p = beta * p - r
      print('Exit CG loop')
      print(' ')
      #-----------

      #kappa = np.einsum('cd,qd,q->',self.mesh.B_with_area_old.todense(),X,sigma[:,0],optimize=True)*self.kappa_factor*1e-18
      #TL = np.matmul(S,X)
      TL = np.matmul(S,X)
     
      #DeltaTot += ((DeltaT.T)*s).T 
      DeltaTot += ((DeltaT.T)*s).T 
      #DeltaT = ((DeltaT.T)*a).T 
      #X = ((Xs.T)/s).T
      #TL = np.dot(S,X)
      #kappa = np.einsum('cd,qd,q->',self.mesh.B_with_area_old.todense(),X,sigma[:,0],optimize=True)*self.kappa_factor*1e-18
      #print(kappa)
      #error = abs((kappa-kappa_old))/abs(kappa)
      r_k_norm_restart = r_k_norm
      p_restart = p
      #X1 = X0  + (DeltaTot.T/a).T
      #kappa = np.einsum('cd,qd,q->',self.mesh.B_with_area_old.todense(),X1,sigma[:,0],optimize=True)*self.kappa_factor*1e-18
      #print(kappa)


      if len(self.eb) > 0:
       #Compute TB----
       Bm = np.zeros((self.n_index,self.mesh.nle))
       for n,(i,j) in enumerate(zip(self.eb,self.sb)):
         Bm[:,i]  += Gbm[:,n] * np.dot(X[:,i],Gtbp[:,n])/Gtbm[n] #the sum is for each side

  def solve_bte_working_3(self,**argv):

     nT = np.shape(self.mat[0]['B'])[0]
     kappa_fourier,temp_fourier,temp_fourier_grad,temp_fourier_int,flux_fourier_int = self.get_diffusive_suppression_function()
     print(kappa_fourier)

     if argv.setdefault('only_fourier'):
       data = {'kappa_vec':np.array([kappa_fourier]),'temperature_fourier':temp_fourier,'flux_fourier':-self.mat[0]['kappa_bulk_tot']*temp_fourier_grad}
       pickle.dump(data,open(argv.setdefault('filename_solver','solver.p'),'wb'),protocol=pickle.HIGHEST_PROTOCOL)

       quit()

     rta = argv.setdefault('rta',False)

     kappa_fourier = np.array([kappa_fourier])
     Tnew = temp_fourier.copy()
     TB = np.tile(temp_fourier,(self.n_side_per_elem,1)).T
     
     tc = np.load('tc')
     W = np.load('W')
     invW = np.load('invW')
     a = np.diag(W)
     sigma = np.load('sigma')*1e9
     F = (sigma.T/a).T
     coeff = sigma
     Wod = np.diag(a) - W
     #scale = np.ones_like(a)
     scale = a
     #S = np.einsum('u,uv->uv',a,invW)

     #H = np.eye(self.n_index) - invW
     H = -(invW - np.diag(np.diag(1/a)))

     #Solve Bulk
     G = np.einsum('qj,jn->qn',coeff,self.k,optimize=True)
     Gp = G.clip(min=0)
     Gm = G.clip(max=0)
     D = np.zeros((self.n_index,self.mesh.nle),dtype=float64)
     for n,i in enumerate(self.i): 
       D[:,i] += Gp[:,n]

     D2 = np.zeros((self.n_index,self.mesh.nle),dtype=float64)
     D2[:,self.i] = Gp

     D += np.tile(scale,(self.mesh.nle,1)).T

     #Compute boundary------------------------------------------
     Bm = np.zeros((self.n_index,self.mesh.nle))
     if len(self.db) > 0:
      Gb = np.einsum('qj,jn->qn',coeff,self.db,optimize=True)
      Gbp = Gb.clip(min=0)
      Gbm = Gb.clip(max=0)
      Bp = np.zeros((self.n_index,self.mesh.nle),dtype=float64)
    
      for n,(i,j) in enumerate(zip(self.eb,self.sb)):
        Bp[:,i] += Gbp[:,n]
        Bm[:,i] -= TB[i,j]*Gbm[:,n]
      SS = np.einsum('qn,n->nq',Gbm,1/Gbm.sum(axis=0))
      D += Bp
      #------------------------------------------------------

     #Thermostatting------------------------------------------
     Bt = np.zeros((self.n_index,self.mesh.nle))
     KP = np.zeros((self.n_index,self.mesh.nle))
     KM = 0
     if len(self.dft) > 0:
      Gb = np.einsum('qj,jn->qn',coeff,self.dft,optimize=True)
      Gftp = Gb.clip(min=0)
      Gftm = Gb.clip(max=0)

      Bftp = np.zeros((self.n_index,self.mesh.nle),dtype=float64)
      for n,(i,j) in enumerate(zip(self.eft,self.sft)):
       Bftp[:,i] += Gftp[:,n]
       Bt[:,i]   -= self.mesh.IB[i,j]*Gftm[:,n]
       

      KP =  np.einsum('uc,c->uc', Bftp,self.mesh.B_area,optimize=True)*self.kappa_factor*1e-18 #Power in
      KM = -np.einsum('uc,c->',Bt,self.mesh.B_area,optimize=True)*self.kappa_factor*1e-18 #Power in
      D += Bftp

     ms = MultiSolver(self.i,self.j,Gm,D,self.mesh.nle)

     #Periodic------------------
     P = np.zeros((self.n_index,self.mesh.nle))
     for n,(i,j) in enumerate(zip(self.i,self.j)): 
      P[:,i] += Gm[:,n]*self.mesh.B[i,j]

     #inte = -np.sum(self.mesh.B_with_area_old.todense(),axis=0)
     BB = -np.outer(sigma[:,0],np.sum(self.mesh.B_with_area_old.todense(),axis=0))*self.kappa_factor*1e-18
     #Boundary---
     #Bm = np.zeros((self.n_index,self.mesh.nle))
     #if len(self.db) > 0:
      #Gtb  = np.einsum('qj,jn->qn',sigma,self.dbp,optimize=True)
      #Gtb  = np.einsum('qj,jn->qn',sigma,self.db,optimize=True)
      #Gtbp = Gtb.clip(min=0)
      #Gtbm = Gtb.clip(max=0).sum(axis=0)
      #SS = np.einsum('qn,n->nq',Gbm,1/Gtbm)


     #---------------------------------------------
     kappa_vec = [kappa_fourier]
     miter = argv.setdefault('max_bte_iter',100)
     merror = argv.setdefault('max_bte_error',1e-3)
     alpha = argv.setdefault('alpha',1)

     error = 1
     kk = 0

     #----Constant gradient----
     #LL = self.mesh.size[0]
     #Tnew = np.zeros_like(Tnew)
     #for i in range(self.mesh.nle):
     # c = self.mesh.compute_elem_centroid(i)[0]
     # delta = (c+LL/2)/LL
     # t = -0.5 + delta
     # Tnew[i] = t
     
     #---------------------
     X = np.tile(Tnew,(self.n_index,1))
     #X = (X.T*scale).T
     DeltaT = X
     #X = np.zeros((self.n_index,self.mesh.nle))
     X_old = X.copy()
     #DeltaT = -X.copy()
     #DeltaT = np.matmul(Wod,alpha*X+(1-alpha)*X_old) 
     #DeltaT = np.matmul(Wod,X) 
     #T = np.matmul(Wod,alpha*X+(1-alpha)*X_old) 
     #DeltaT = np.zeros((self.n_index,self.mesh.nle))
     #B = np.tile(sigma[:,0],(self.mesh.nle,1)).T/100.0
     alpha = 0.75
     Bm_old = Bm.copy()
     #B = np.einsum('uj,cj->uc',sigma,self.mesh.compute_grad(Tnew)) 
     kappa_old = kappa_fourier
     while kk < miter and error > merror:
      #DeltaT = np.matmul(H,B)
      #DeltaT = np.einsum('uk,kc->uc',H,B)
      if not rta:
       DeltaT = np.matmul(Wod,alpha*X+(1-alpha)*X_old) 
      else:
       DeltaT = np.einsum('qc,q,u->uc',X,tc,a)

      X_old = X.copy()
      X = ms.solve(P  + Bt  + Bm  + DeltaT)
      kappa = np.sum(np.multiply(BB,X))
      print('BTE:', kappa)

      #B = -((X.T)*a).T+DeltaT
      #X = ms.solve(P + Delta)
      #if kk > 0:
      # X = ms.solve(DeltaT+P)
      #else:    
      #X = ms.solve(DeltaT)
      kk +=1
      #print(X)
      #quit()
      #B = np.zeros_like(DeltaT)
      #for n in range(self.n_index):
      # B[n] = np.einsum('j,cj->c',sigma[n],self.mesh.compute_grad(X[n]))
      #B = -(X.T*a).T + DeltaT 
      #B = ((-X + DeltaT).T*a).T


      ##DeltaT = -np.matmul(H,B)
      #print(DeltaT)
      #quit()
      
     #DeltaT = (DeltaT.T*a).T


      #B = (X.T*a).T - T
      #B = np.tile(sigma[:,0],(self.mesh.nle,1)).T/100.0

      #X = self.cg(W,-(X.T/a).T+DeltaT,BB,x=X)

      #DeltaX = self.cg(W,B,BB,x=X)
      #DeltaT = ((DeltaT.T)*a).T
      #X = self.cg(W,B,BB,x=X)
      #quit()

      if len(self.db) > 0: 
         Bm[:,self.eb] = np.einsum('un,un,nq->qn',X[:,self.eb],Gbp,SS,optimize=True)
         #m = alpha*Bm + (alpha-1)*Bm_old
         #Bm =
         #Bm_old = Bm.copy()

      #Due to periodicity
      #kappa = np.sum(np.multiply(BB,X))
      #print('CG: ', kappa)
      #DeltaT = ((DeltaX.T)*a).T
      #Due to thermostatting
      #kappa += np.sum(np.multiply(KP,X)) + KM

      #print(np.sum(np.multiply(KP,X)))
      #print(KM)
      error = abs(kappa_old-kappa)/abs(kappa)
      kappa_old = kappa
      kappa_vec.append(kappa)
    
      #print(self.mesh.nle)

      #B = np.tile(sigma[:,0],(self.mesh.nle,1)).T
      #DeltaT = np.matmul(S,B)
      #print(B)
      #for n in range(self.n_index):
      # B[n] = np.einsum('j,cj->c',sigma[n],self.mesh.compute_grad(X[n]))
     
      #print(B)

      #B = X + (DeltaT.T/a).T 

      #quit()

      #DeltaT = np.matmul(S,(X.T*a).T+DeltaT+P)
      #X = np.matmul(invW,B)
      #quit()
      #DeltaT = np.matmul(Wod,X)
      #kappa = -np.sum(np.multiply(BB,X))/100

     #quit()
     
     T = np.einsum('qc,q->c',X,tc)
     #DeltaT = X-np.tile(T,(self.n_index,1))
     #J = np.einsum('qj,qc->cj',sigma,DeltaT)*1e-18
     J = np.einsum('qj,qc->cj',sigma,X)*1e-18
     data = {'kappa_vec':kappa_vec,'temperature':T,'flux':J,'temperature_fourier':temp_fourier,'flux_fourier':-self.mat[0]['kappa_bulk_tot']*temp_fourier_grad}
     pickle.dump(data,open(argv.setdefault('filename_solver','solver.p'),'wb'),protocol=pickle.HIGHEST_PROTOCOL)



  def cg(self,A, b,BB,precondition=True,x=[]):

    if len(x) == 0:
     x = b.copy()/np.diag(A)

    kappa = np.sum(np.multiply(BB,x)) 
    print('Guess ', kappa)

    if precondition:
     s = np.sqrt(np.diag(A))   
     A = np.einsum('ij,i,j->ij',A.copy(),1/s,1/s)
     b = (b.copy().T/s).T
     x = (x.copy().T*s).T
    

    n,_ = np.shape(b)

    r = np.dot(A,x)-b
    #r = np.dot(A, x) - b
    p = - r
    r_k_norm = np.array([ np.dot(r[:,i],r[:,i]) for i in range(self.mesh.nle)])

    #r_k_norm = np.dot(r, r)
    error = 1
    kappa_old=0
    xn = 0
    N = 2*n
    N = 14
    delta = np.zeros((self.n_index,self.mesh.nle))
    for i in range(N):

         
        Ap = np.array([np.dot(A,p[:,i]) for i in range(self.mesh.nle)]).T
        pAp =np.array([ np.dot(p[:,i],Ap[:,i]) for i in range(self.mesh.nle)])
        alpha = r_k_norm / pAp
        x += alpha * p
        delta += alpha*p
        kappa = np.sum(np.multiply(BB,((x.T)/s).T))

        error = abs((kappa-kappa_old))/abs(kappa)
        print(i,kappa,error)
        if error < 1e-4:
         break
        kappa_old = kappa


        r += alpha * Ap
        r_kplus1_norm = np.array([ np.dot(r[:,i],r[:,i])           for i in range(self.mesh.nle)])
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        p = beta * p - r
        
    return ((delta.T)/s).T



  def solve_bte_working(self,**argv):

     nT = np.shape(self.mat[0]['B'])[0]
     kappa_fourier,temp_fourier,temp_fourier_grad,temp_fourier_int,flux_fourier_int = self.get_diffusive_suppression_function()
     print(kappa_fourier)

     if argv.setdefault('only_fourier'):
        quit()

     kappa_fourier = np.array([kappa_fourier])
     Tnew = temp_fourier.copy()
     TB = np.tile(temp_fourier,(self.n_side_per_elem,1)).T
     #TB_new = np.array([temp_fourier[i] for n,(i,j) in enumerate(zip(self.eb,self.sb))])

     #TL = np.zeros((nT,self.mesh.nle))

     #W = np.load('W.dat.npy')
     #a = np.diag(W)
     #if argv.setdefault('rta',False):
     # W = np.diag(a) 

     #Winv = np.linalg.inv(W)
     #Wod = W-np.diag(a)
     #g  = W.sum(axis=1)

     
     tc = np.load('tc')
     W = np.load('W')
     a = np.diag(W)
     sigma = np.load('sigma')*1e9
     Wod = W-np.diag(a)
     #tc = W.sum(axis=0)/sum(g)
     
     #S = np.outer(g,tc) - Wod
     #S = - Wod
     #S = np.load('S.dat')
     #a = np.load('a.dat')
     #S = np.outer(g,g) - Wod

     #sigma = np.load(open('sigma.dat','rb'),allow_pickle=True)

     #sigma = self.rotate(sigma,**{'rotate_phi':.0})
     #sigma.dump('sigma_30.dat')

     #F = (sigma.T/a).T
     #Sx = sigma[:,0]
     #------------------

     #Solve Bulk
     G = np.einsum('qj,jn->qn',sigma,self.k,optimize=True)
     Gp = G.clip(min=0)
     Gm = G.clip(max=0)
     D = np.zeros((self.n_index,self.mesh.nle),dtype=float64)
     for n,i in enumerate(self.i): 
       D[:,i] += Gp[:,n]

     D2 = np.zeros((self.n_index,self.mesh.nle),dtype=float64)
     D2[:,self.i] = Gp

     D += np.tile(a,(self.mesh.nle,1)).T

     #Compute boundary---
     if len(self.db) > 0:
      Gb = np.einsum('qj,jn->qn',sigma,self.db,optimize=True)
      Gbp = Gb.clip(min=0)
      Gbm = Gb.clip(max=0)
      Bp = np.zeros((self.n_index,self.mesh.nle),dtype=float64)
      for n,i in enumerate(self.eb): 
        Bp[:,i] += Gbp[:,n]

      D += Bp

     #Compute super lu---
     #lu = get_lu(self.i,self.j,Gm,D,self.mesh.nle)   

     ms = MultiSolver(self.i,self.j,Gm,D,self.mesh.nle)


     #Periodic------------------
     P = np.zeros((self.n_index,self.mesh.nle))
     for n,(i,j) in enumerate(zip(self.i,self.j)): 
      P[:,i] += Gm[:,n]*self.mesh.B[i,j]

     #Initial lattice temperature----
     #T= np.einsum('q,c->qc',a,Tnew)
     
     #new---
     #tt = np.einsum('qj,qu,u->ju',sigma,Winv,a)*self.kappa_factor*1e-18

     #kappam = np.einsum('qj,qu,u->ju',sigma,Winv,a)*self.kappa_factor*1e-18
     #kappam.dump('kappam30.dat')
     #quit()
     #G = np.einsum('uq,qj->uj',Winv,sigma)
     #bulk = np.einsum('qi,qu,uj->ij',sigma,Winv,sigma)
    
     inte = -np.sum(self.mesh.B_with_area_old.todense(),axis=0)
     #-----

     BB = -np.outer(sigma[:,0],np.sum(self.mesh.B_with_area_old.todense(),axis=0))*self.kappa_factor*1e-18

     Bm = np.zeros((self.n_index,self.mesh.nle))
     if len(self.db) > 0:
      Gtb  = np.einsum('qj,jn->qn',sigma,self.dbp,optimize=True)
      Gtbp = Gtb.clip(min=0)
      Gtbm = Gtb.clip(max=0).sum(axis=0)
      SS = np.einsum('qn,n->nq',Gbm,1/Gtbm)
      for n,(i,j) in enumerate(zip(self.eb,self.sb)):
        Bm[:,i] -= TB[i,j]*Gbm[:,n]


     #---------------------------------------------
     #Conjugate gradient------------
     #TL_old = TL.copy()
     kappa_vec = [kappa_fourier]

     miter = argv.setdefault('max_bte_iter',100)
     merror = argv.setdefault('max_bte_error',1e-3)

     alpha = argv.setdefault('alpha',1)


     error = 1
     kk = 0
     DeltaT = np.zeros((self.n_index,self.mesh.nle))

     X = np.tile(Tnew,(self.n_index,1))


     kappa_old = kappa_fourier
     while kk < miter and error > merror:

      #X = ms.solve(P + TL + Bm)
      #X = ms.solve(P + Bm + T + DeltaT)


      DeltaT = np.matmul(-Wod,X)
      #print(np.allclose(np.outer(a,Tnew),DeltaT))
      X = ms.solve(P + Bm + DeltaT)

      #T = np.tile(T,(self.n_index,1))

      kappa = np.sum(np.multiply(BB,X)) 
      error = abs(kappa_old-kappa)/abs(kappa)
      kappa_old = kappa
      kappa_vec.append(kappa)
      print(kappa)

      #Xm = np.einsum('qc,c->q',X-((TL.T+Bm.T)/a).T,inte)
      #print(np.dot(Xm,tt[0]),kappa)
      #Xm = np.einsum('qc,c->q',X-T,inte)*self.kappa_factor
      #print(np.dot(sigma[:,0]*1e-18,Xm),kappa)
      #print(np.sum(np.multiply(BBt,X-((TL.T)/a).T)),kappa)
      #DeltaT = np.matmul(Wod,(T.T/a).T-X)
      #DeltaT = np.matmul(Wod,X)

      #T = np.einsum('qc,q->c',X,tc)
      #T = np.einsum('q,c->qc',a,T)


      #print(np.sum(np.matmul(A,(T.T/a).T))/np.sum(np.matmul(A,X)))

      

      #TL = np.matmul(S,X)*alpha + (1-alpha)*TL_old; TL_old = TL.copy()

      if len(self.db) > 0: Bm[:,self.eb] = np.einsum('un,un,nq->qn',X[:,self.eb],Gtbp,SS,optimize=True)

     quit()
     #compute grad---
     #ss = []
     #for n in range(self.n_index):
     # cc = -np.einsum('j,cj,c->',F[n],self.mesh.compute_grad(X[n]),inte)
     # ss.append(cc)
     #sup = np.array(ss)     

     #kappa_mode = np.multiply(sigma[:,0]*1e-18,G[:,0])
     #print(np.sum(kappa_mode))
     #print(np.dot(sup,kappa_mode))

      #print(np.dot(sigma[:,0]*1e-18,G[:,0]))
     #(Xm/G[:,0]).dump('sup.dat')
     #sup.dump('sup.dat')
     #F.dump('f.dat')
     Xm.dump('sup.dat')
     #sup.dump('sup_grad.dat')
     #tt.dump('bulk.dat')
     #G.dump('full.dat')
     T = np.einsum('qc,q->c',X,tc)
     DeltaT = X-np.tile(T,(self.n_index,1))
     J = np.einsum('qj,qc->cj',sigma,DeltaT)*1e-18
     data = {'kappa_vec':kappa_vec,'temperature':T,'flux':J,'temperature_fourier':temp_fourier,'flux_fourier':-self.mat[0]['kappa_bulk_tot']*temp_fourier_grad}
     pickle.dump(data,open(argv.setdefault('filename_solver','solver.p'),'wb'),protocol=pickle.HIGHEST_PROTOCOL)

     


  def solve_bte_cg(self,**argv):

     nT = np.shape(self.mat[0]['B'])[0]
     kappa_fourier,temp_fourier,temp_fourier_grad,temp_fourier_int,flux_fourier_int = self.get_diffusive_suppression_function()
     kappa_fourier = np.array([kappa_fourier])
     Tnew = temp_fourier.copy()
     TB = np.tile(temp_fourier,(self.n_side_per_elem,1)).T
     TB_new = np.array([temp_fourier[i] for n,(i,j) in enumerate(zip(self.eb,self.sb))])

     TL = np.zeros((nT,self.mesh.nle))

     W = np.load('W.dat.npy')
     a = np.diag(W)
     Wod = W-np.diag(a)
     g  = W.sum(axis=0)
     S = np.outer(g,g)/np.sum(g) - Wod
     A = W-np.outer(g,g)/np.sum(g) 
     s = np.sqrt(np.diag(A))
     TC = np.multiply(a,g/np.sum(g))
     Aod = Wod-np.outer(g,g)/np.sum(g) 
     sigma = np.load(open('sigma.dat','rb'),allow_pickle=True)*1e9
     F = (sigma.T/a).T
     Sx = sigma[:,0]
     ss = np.sqrt(a)

     #Solve Bulk
     #G = np.einsum('qj,jn->qn',sigma,self.k,optimize=True)
     G = np.einsum('qj,jn->qn',F,self.k,optimize=True)
     Gp = G.clip(min=0)
     Gm = G.clip(max=0)
     D = np.zeros((self.n_index,self.mesh.nle),dtype=float64)
     for n,i in enumerate(self.i): 
       D[:,i] += Gp[:,n]
  
     D += np.tile(a,(self.mesh.nle,1)).T
     #D += np.tile(np.ones(self.n_index),(self.mesh.nle,1)).T

     P = np.zeros((self.n_index,self.mesh.nle))
     for n,(i,j) in enumerate(zip(self.i,self.j)):
       P[:,i] += Gm[:,n]*self.mesh.B[i,j]


     lu = get_lu(self.i,self.j,Gm,D,self.mesh.nle,scaling=[])   
     #Fourier---
     X = np.einsum('q,c->qc',np.ones(self.n_index),Tnew)
     #RTA----
     X = solve_from_lu(lu,P+X)

     kappa = np.einsum('cd,qd,q->',self.mesh.B_with_area_old.todense(),X,sigma[:,0],optimize=True)*self.kappa_factor*1e-18
     print(kappa)


     quit()

     #lu = get_lu(self.i,self.j,Gm,D,self.mesh.nle,scaling=1/np.sqrt(a))   

     #Conjugate gradient------------
     TL = np.einsum('q,c->qc',a,Tnew)
     #TLw = np.einsum('q,c->qc',np.ones(self.n_index),Tnew)
     #TL = np.einsum('q,c->qc',np.ones(self.n_index),Tnew)


     #eta1 = np.einsum('cd,qd->q',self.mesh.B_with_area_old.todense(),X1,optimize=True)*self.kappa_factor*1e-18
     #plt.plot(eta,color='r')
     #r = np.dot(Aod,X) + TL
     
     #X1 = TLw -np.tile(Sx/a/self.mesh.size[0],(self.mesh.nle,1)).T 
     X = solve_from_lu(lu,P+TL)
  
     kappa = -np.einsum('cd,qd,q->',self.mesh.B_with_area_old.todense(),X,sigma[:,0],optimize=True)*self.kappa_factor*1e-18
     print(kappa)
     quit()
     #B = -np.tile(Sx,(self.mesh.nle,1)).T/self.mesh.size[0]

     B = (((X-TL).T)*a).T 
     #B = -X + TL
     #preconditioning---
     Aod = np.einsum('ij,i,j->ij',Aod,1/s,1/s)
     A = np.einsum('ij,i,j->ij',A,1/s,1/s)
     B = (B.T/s).T
     TL = (TL.T*s).T
     P = (P.T*s).T
     X = (X.T*s).T
     r = np.dot(A,X)-B

     #X *= a
     #-----------------

     #Xs = (X.T/s).T
     #kappa = np.einsum('cd,qd,q->',self.mesh.B_with_area_old.todense(),Xs,sigma[:,0],optimize=True)*self.kappa_factor*1e-18
     #print(kappa)

     #print(np.allclose(X2,X,rtol=1e-1,atol=1e-1))
     #print(X)
     #print(X2)
     #quit()
     #print(np.allclose(r,r2))
     #quit()
     #eta2 = np.einsum('cd,qd->q',self.mesh.B_with_area_old.todense(),X2,optimize=True)*self.kappa_factor*1e-18

     #print(np.allclose(X1,X2,atol=1e-2,rtol=1e-2))

     #quit()
     #plt.plot(eta,color='c')
     #plt.legend(['equation','analytical'])
     #plt.show()


     p =-r
     r_k_norm = np.array([ np.dot(r[:,i],r[:,i]) for i in range(self.mesh.nle)])

     RHS = np.zeros_like(TL)
     for i in range(100):
      Ap = np.array([np.dot(A,p[:,i]) for i in range(self.mesh.nle)]).T
      pAp =np.array([ np.dot(p[:,i],Ap[:,i]) for i in range(self.mesh.nle)])
      alpha = r_k_norm / pAp
     # RHS += np.multiply(np.outer(a,alpha),p)
      RHS += np.multiply(np.outer(np.ones(self.n_index),alpha),p)

      X = solve_from_lu(lu,P+RHS+TL) 

      #redefine this----
      #------
      #X += alpha * p
      
      Xs = (X.T/s).T
      kappa = -np.einsum('cd,qd,q->',self.mesh.B_with_area_old.todense(),Xs,sigma[:,0],optimize=True)*self.kappa_factor*1e-18
      print(kappa)

      #TL = np.einsum('qc,q->qc',X,TC) #already scales

      #B = ((X.T)*a).T - TL - RHS
      B = X-TL-RHS
      r = np.dot(A,X)-B
      #r += alpha * Ap
      #r = np.dot(A,X)-np.tile(Sx,(self.mesh.nle,1)).T/self.mesh.size[0]
      #r = np.dot(Aod,X) + TL + RHS


      r_kplus1_norm = np.array([ np.dot(r[:,i],r[:,i])           for i in range(self.mesh.nle)])
      beta = r_kplus1_norm / r_k_norm
      r_k_norm = r_kplus1_norm
      p = beta * p - r



     quit() 




     alpha = 0.75
     TL_old = TL.copy()
     for kk in range(5):
      X = solve_from_lu(lu,P+TL)
      kappa = np.einsum('cd,qd,q->',self.mesh.B_with_area_old.todense(),X,sigma[:,0],optimize=True)*self.kappa_factor*1e-18
      print(kappa)
      TL = np.matmul(S,X)
      TL = alpha*TL + (1-alpha)*TL_old




  def solve_bte(self,**argv):

     print('    Iter    Thermal Conductivity [W/m/K]      Error        Diffusive  -  BTE  -  Ballistic')
     print('   ---------------------------------------------------------------------------------------')

     #compute fourier-------
     nT = np.shape(self.mat[0]['B'])[0]
     kappa_fourier,temp_fourier,temp_fourier_grad,temp_fourier_int,flux_fourier_int = self.get_diffusive_suppression_function()
     kappa_fourier = np.array([kappa_fourier])
     Tnew = temp_fourier.copy()
     TB = np.tile(temp_fourier,(self.n_side_per_elem,1)).T
     #TB_new = np.zeros(len(self.eb))
     #for n,(i,j) in enumerate(zip(self.eb,self.sb)):
     TB_new = np.array([temp_fourier[i] for n,(i,j) in enumerate(zip(self.eb,self.sb))])

     TL = np.zeros((nT,self.mesh.nle))
     
     master_data = []
     master_b = []
     master_A = []

     n_iter = 0
     error_vec = [1]

     #this can be done in geometry--
     delta = 1e-16
     #delta = 0
     #Bulk----
     #skla.init()

     '''
     BB = np.zeros((len(self.i),self.mesh.nle))

     for n,i in enumerate(self.i):
      BB[n,i] = 1 

     C = np.zeros((len(self.eb),self.mesh.nle))
     for n,eb in enumerate(self.eb):
      C[n,eb] = 1 

     CC = np.zeros((len(self.ip),self.mesh.nle))
     for n,ip in enumerate(self.ip):
      CC[n,ip] = 1 

     DD = np.zeros((len(self.eb),self.mesh.nle))
     for n,eb in enumerate(self.eb):
      DD[n,eb] = 1 

     ##Get permutation matrix---
     #rows = np.append(self.i,np.arange(self.mesh.nle))
     #cols = np.append(self.j,np.arange(self.mesh.nle))
     rows = self.i
     cols = self.j
     '''

     #A = sp.coo_matrix((np.arange(len(rows)),(rows,cols)),dtype=np.int32)
     #_, _, E, rank = sparseqr.qr(A)
     #pp = []
     #E = list(E)
     #for i in range(self.mesh.nle):
     # pp.append(E.index(i))        
     #cols = np.array([pp[c] for c in cols])
     #cols = np.array([int(E[k]) for k in cols])
     #A = A.tocsr().sorted_indices()
     #-----------------------------


     #A = A.tocsr().sorted_indices()
     #create new matrix
     #cols = [int(E[k]) for k in cols]
     #print(cols) 
     #quit()
     #A = sp.csr_matrix((np.arange(len(rows)),(rows,cols)),dtype=np.int32) 
     #indices = A.indices.copy()
     #indptr  = A.indptr.copy()
     #rot = np.asarray(A.data,dtype=np.int)
     #-------------------------------------


     #W = np.load('W.dat.npy')
     #sigma = np.load(open('sigma.dat','rb'),allow_pickle=True)*1e9
     #a = np.diag(W)
     #Wod = W-np.diag(a)     


     W = np.load('W.dat.npy')
     a = np.diag(W)
     Wod = W-np.diag(a)
     g  = W.sum(axis=0)
     tc = g/np.sum(g)
     S = np.outer(g,g)/np.sum(g) - Wod
     sigma = np.load(open('sigma.dat','rb'),allow_pickle=True)*1e9
     F = (sigma.T/a).T
     Sx = sigma[:,0]

     #Solve Bulk

     G = np.einsum('qj,jn->qn',sigma,self.k,optimize=True)
     #G = np.einsum('qj,jn->qn',F,self.k,optimize=True)
     #G = np.einsum('q,qj,jn->qn',self.mfp,self.control_angle,self.k,optimize=True)
     Gp = G.clip(min=0)
     Gm = G.clip(max=0)
     D = np.zeros((self.n_index,self.mesh.nle),dtype=float64)
     for n,i in enumerate(self.i): 
       D[:,i] += Gp[:,n]
  
     #D += np.ones_like(D)
     D += np.tile(a,(self.mesh.nle,1)).T

     P = np.zeros((self.n_index,self.mesh.nle))
     for n,(i,j) in enumerate(zip(self.i,self.j)):
       P[:,i] += Gm[:,n]*self.mesh.B[i,j]

     lu = get_lu(self.i,self.j,Gm,D,self.mesh.nle)   

     #Conjugate gradient------------
     TL = np.einsum('q,c->qc',a,Tnew)
     #TL = np.einsum('q,c->qc',np.ones(self.n_index),Tnew)
     alpha = 0.75
     TL_old = TL.copy()
     for kk in range(5):
      X = solve_from_lu(lu,P+TL)
      kappa = np.einsum('cd,qd,q->',self.mesh.B_with_area_old.todense(),X,sigma[:,0],optimize=True)*self.kappa_factor*1e-18
      print(kappa)
      TL = np.matmul(S,X)
      TL = alpha*TL + (1-alpha)*TL_old


     #Vector

     #Theoretical solution
     X1 = np.tile(-Sx/np.diag(W)/self.mesh.size[0],(self.mesh.nle,1)).T #+ np.einsum('q,c->qc',np.ones(self.n_index),Tnew)
     #print(np.sum(X1))
     kappa = -np.einsum('cd,qd,q->q',self.mesh.B_with_area_old.todense(),X1,Sx,optimize=True)*self.kappa_factor

     plt.plot(kappa,color='r')
     
     #b = np.tile(-Sx,(self.mesh.nle,1)).T/self.mesh.size[0]

     X = solve_from_lu(lu,P+TL)

     #print(np.sum(X))
     #print(np.allclose(X1,X))
     kappa = -np.einsum('cd,qd,q->q',self.mesh.B_with_area_old.todense(),X - np.einsum('q,c->qc',np.ones(self.n_index),Tnew),sigma[:,0])*self.kappa_factor
     plt.plot(kappa,color='b')
     plt.show()
     print(kappa)
     quit()

     #print(X1)
     #quit()

     #kappa = np.sum(np.dot(X.T,Sx))
     #print(kappa)
     #quit()

     #b = TL
     kappa = -np.einsum('cd,qd,q->',self.mesh.B_with_area_old.todense(),X,Sx,optimize=True)*self.kappa_factor
     print(kappa)
     quit()


     r = np.dot(W,X)-b
     p =-r
     r_k_norm = np.array([ np.dot(r[:,i],r[:,i])           for i in range(self.mesh.nle)])
     for i in range(4):
      Ap = np.array([np.dot(W,p[:,i]) for i in range(self.mesh.nle)]).T
      pAp =np.array([ np.dot(p[:,i],Ap[:,i]) for i in range(self.mesh.nle)])
      alpha = r_k_norm / pAp
      #X = solve_from_lu(lu,P+TL+alpha*p)
      X += alpha * p
      kappa = -np.einsum('cd,qd,q->',self.mesh.B_with_area_old.todense(),X,sigma[:,0],optimize=True)*self.kappa_factor
      print(kappa)
      r += alpha * Ap
      r_kplus1_norm = np.array([ np.dot(r[:,i],r[:,i])           for i in range(self.mesh.nle)])
      beta = r_kplus1_norm / r_k_norm
      r_k_norm = r_kplus1_norm
      p = beta * p - r



     quit() 
     sigma *=1e9
     g = W.sum(axis=0)

     a = np.diag(W)
     A = -W + np.outer(g,g)/np.sum(g)
     S =  A + np.diag(a) 
     
     #Wod = W-np.diag(a)
     #tc = g/np.sum(g)
     #S = np.outer(g,g)/np.sum(g) - Wod
     #S = -A
     #A = -S - np.diag(a)
   

     #c = np.diag(A)
     F = (sigma.T/a).T
     Fx = F[:,0]
     Sx = sigma[:,0]

     #-----------------------
     t1 = time.time()
     self.Coll =self.mat[0]['B']
     self.M = self.Coll + np.eye(self.n_index)

     #CPU
     t1 = time.time() 
     #G = np.einsum('q,qj,jn->qn',self.mfp,self.control_angle,self.k,optimize=True)
     G = np.einsum('qj,jn->qn',sigma,self.k,optimize=True)
     Gp = G.clip(min=0)
     Gm = G.clip(max=0)
     D = np.zeros((self.n_index,self.mesh.nle),dtype=float64)
     for n,i in enumerate(self.i): 
       D[:,i] += Gp[:,n]
  
     #D += np.ones_like(D)

     D += np.tile(a,(self.mesh.nle,1)).T

     P = np.zeros((self.n_index,self.mesh.nle))
     for n,(i,j) in enumerate(zip(self.i,self.j)):
       P[:,i] += Gm[:,n]*self.mesh.B[i,j]

     lu = get_lu(self.i,self.j,Gm,D,self.mesh.nle)   

     #Conjugate gradient------------
     TL = np.einsum('q,c->qc',a,Tnew)

     X = np.tile(Fx,(self.mesh.nle,1)).T/100
     kappa = np.einsum('cd,qd,q->',self.mesh.B_with_area_old.todense(),X,sigma[:,0],optimize=True)*self.kappa_factor*1e-18
     #TL_new = np.matmul(S,X)
     print(kappa)

     B = np.tile(Sx,(self.mesh.nle,1)).T/100
     

     r = np.matmul(-W,X) - B
     p = -r
     r_k_norm = np.einsum('uc,vc->c',r,r)
     for kk in range(1):
      Ap  = np.einsum('uv,vc->uc',A,p)
      pAp = np.einsum('uc,vc->c',p,Ap)
      alpha = np.array([ r_k_norm[c]/pAp[c]  for c in range(self.mesh.nle) ])
      X += np.einsum('c,qc->qc',alpha,p)
      #X = solve_from_lu(lu,P-dx)

      kappa = np.einsum('cd,qd,q->',self.mesh.B_with_area_old.todense(),X,sigma[:,0],optimize=True)*self.kappa_factor*1e-18
      print(kappa)
      quit()

      r += np.einsum('c,qc->qc',alpha,Ap)
      r_kplus1_norm = np.einsum('uc,vc->c',r,r)
      beta = r_kplus1_norm / r_k_norm
      r_k_norm = r_kplus1_norm

      p = np.einsum('c,qc->qc',beta,p) - r

     quit()

     #r_k_norm = np.einsum('uc,vc->c',r,r)




     #----------------------------------
     quit()

     #r = TL

     alpha = 0.75
     TL_old = TL.copy()
     for kk in range(1000):
      X = solve_from_lu(lu,P+TL)
      #TL = np.matmul(self.Coll,X)
      TL = np.matmul(S,X)
      TL = alpha*TL + (1-alpha)*TL_old
      r = np.linalg.norm(TL - TL_old)
      TL_old = TL.copy() 
      #TL = skla.dot(gpuarray.to_gpu(self.Coll),gpuarray.to_gpu(X)).get()
      #kappa = np.einsum('cd,qd,q->',self.mesh.B_with_area_old.todense(),X,self.kappa_directional[:,0],optimize=True)*self.kappa_factor
      kappa = np.einsum('cd,qd,q->',self.mesh.B_with_area_old.todense(),X,sigma[:,0],optimize=True)*self.kappa_factor*1e-18
      print(kappa,r)
     print(time.time()-t1)
     quit()
     #Delta = X - np.tile(Tnew,(self.n_index,1))

     #DB = np.zeros((self.n_index,self.mesh.nle))
     #if len(self.db)> 0:
     # G = np.dot(self.control_angle,self.db)
     # G = np.einsum('i,ij->ij',self.mfp,G)
     # Gp  = G.clip(min=-delta)
     # Hm = G.clip(max=delta)
     # for n,eb in enumerate(self.eb): 
     #   DB[:,eb] += Gp[:,n]
     #Dtot = np.ones((self.n_index,self.mesh.nle),dtype=np.float64) + D + DB

     #Dtot = np.ones((self.n_index,self.mesh.nle),dtype=np.float64) + D 

     #F = np.concatenate((Gm,Dtot),axis=1)

     #This needs to be done only once---
     #lu = get_lu(rows,cols,A,self.mesh.nle)

     #Periodic--
     #G  = np.dot(self.control_angle,self.pv)
     #Gm = G.clip(max=delta)
     #P2 = np.einsum('i,ij,j->ij',self.mfp,Gm,self.dp)  

     #for n,i in enumerate(self.ip): 
     #  P[:,i] += 
       #P[:,i] += P2[:,n]
     #-----------------------------------

     #Given Tnew, TL and TB
     #Bn = np.zeros((self.n_index,self.mesh.nle))
     #if len(self.db) > 0:
     # for n,(i,j) in enumerate(zip(self.eb,self.sb)):
     #  Bn[:,i] += TB[i,j]*Hm[:,n]
     #RHS = P + Bn +  TL + np.tile(Tnew,(self.n_index,1))

     #RHS = P #+ np.tile(Tnew,(self.n_index,1))
     #B = RHS.copy()

     #A0 = Dtot.copy()
     #B0 = Tnew
     #X = solve_from_lu(lu,B)
     #X = SolveCPU(rows,cols,A,B,A0,B0)

     Delta = X - np.tile(Tnew,(self.n_index,1))

     #kappa = np.einsum('q,q,q',self.mfp,self.control_angle[:,0],self.kappa_directional[:,0])

     
     #quit()
     #print(kappa)

     kappa = np.einsum('cd,qd,q->q',self.mesh.B_with_area_old.todense(),X,self.kappa_directional[:,0],optimize=True)*self.kappa_factor
     kappa.dump(open('kappa.dat','wb'))
     print(sum(kappa))
     #kappa = np.einsum('cd,qd',self.mesh.B_with_area_old.todense(),Delta,optimize=True)

        #kdir = self.kappa_directional[global_index]

     #print('compute kappa')
     #    eta = self.mesh.B_with_area_old.dot(temp-Tnew).sum()
     #   K2p += np.array([eta*np.dot(kdir,self.mesh.applied_grad)])


     quit()
 
     #method 1
     #t1 = time.time()
     #S = MSparse(rows,cols,self.mesh.nle,self.n_index-3,reordering=False)
     #S.add_LHS(F[3:])
     #x1 = S.solve(RHS[3:])
     #quit()
     #S.free_memory()
     #t2 = time.time()
     #print(t2-t1)

     '''
     #Method 2
     t1 = time.time()
     S = MSparse(rows,colsp,self.mesh.nle,self.n_index-3,reordering=False)
     S.add_LHS(F[3:])
     x2 = S.solve(B[3:])
     x2 = np.array([x[E] for x in x2])
     S.free_memory()
     t2 = time.time()
     print(t2-t1)
     '''
     #Method 3
     #t1 = time.time()
     #F = np.array([ F[:,rot[n]] for n in range(len(rot))]).T #from coo to csr
     #F = F2[3:].copy()
 
     #data = np.zeros_like(F[3:])   
     #for n in np.arange(3,self.n_index):
     # Acsr = sp.coo_matrix((F[n],(rows,cols)),dtype=np.float64).tocsr()#.sorted_indices()
      #Acsr = sp.csr_matrix((F[n],(rows,cols)),dtype=np.float64).sorted_indices()
     # assert(len(Acsr.data)== len(indices))
     # assert(np.allclose(Acsr.indices,indices))
      #if new:
      # Acsr = Acsr*sparseqr.permutation_vector_to_matrix(E).tocsr().sorted_indices()
     # data[n-3] = Acsr.data
     

     #F = data.copy()
     #print(np.allclose(F3,F))
     #quit()


     #N = 1000
     t1 = time.time()
     X = solve_from_lu(lu,B)
     print(time.time()-t1)
     
     quit()
     #Xgpu = SolveGPU(rows,cols,A[:N,:],B[:N])
     #t2 =  time.time()
     Xcpu = SolveCPU(rows,cols,A,B)
     
    
     #t3 =  time.time()
     #print(t2-t1)
     #print(t3-t2)
     #print(np.allclose(Xgpu,Xcpu,atol=1e-1,rtol=1e-1))
     #m = MSparse(rows,cols,self.mesh.nle,self.n_index-3)
     #m.add_LHS(F)
     #X = m.solve(B)
     #m.free_memory()
     quit()

 
     S = MSparse(rows,cols.astype(np.int32),self.mesh.nle,2000,new=False)
     #data = np.tile(A.data,(1,1))
     #print(np.shape(data))
     #print(np.shape(B[3:]))
     #data /= np.max(data)
     #B /= np.max(data)
     #S.add_LHS(F[3:])
     S.add_LHS(F[:2000])
     tt = time.time()
     S.solve(B[:2000])
     print(time.time()-tt)
     S.free_memory()
     quit()    
 
     xs = np.array([x[E] for x in x3])
     t2 = time.time()
     quit()
     print(t2-t1)
     #print(np.allclose(x2,x3))
     #GPU------------------------------------
     t1 = time.time() 
     a1 = gpuarray.to_gpu(np.ascontiguousarray(self.control_angle,dtype=np.float32))

     #self.k = self.k[:,rot]
     a2 = gpuarray.to_gpu(np.ascontiguousarray(self.k,dtype=np.float32))
     #E = gpuarray.to_gpu(np.ascontiguousarray(E,dtype=np.int32))

     G = skla.dot(a1,a2)
                  
     skla.dot_diag(gpuarray.to_gpu(np.ascontiguousarray(self.mfp,dtype=np.float32)),G,overwrite=True)
     t4 = time.time()
     Gp = 0.5 * (G + (1.0-delta)*pycuda.cumath.fabs(G)) #get only the positive part
     #Gf = G.ravel()
     #Gf = gpuarray.if_positive(Gf,Gf,gpuarray.zeros(len(Gf),dtype=np.float32))
     #Gp = Gf.reshape(np.shape(G))

     Am2 = G - Gp

     D = skla.dot(Gp,gpuarray.to_gpu(np.ascontiguousarray(BB,dtype=np.float32)))

     #Coll = gpuarray.to_gpu(np.ascontiguousarray(self.mat[0]['B'],dtype=np.float32))
     #----------------------------------------------------
     
     #Boundary--
     control_angle = gpuarray.to_gpu(np.ascontiguousarray(self.control_angle,dtype=np.float32))
     G = skla.dot(control_angle,gpuarray.to_gpu(np.ascontiguousarray(self.db,dtype=np.float32)))

     skla.dot_diag(gpuarray.to_gpu(np.ascontiguousarray(self.mfp,dtype=np.float32)),G,overwrite=True)

     Gp = 0.5 * (G + (1.0-delta)*pycuda.cumath.fabs(G)) #get only the positive part

     #Hm = G - Gp
     
     DB = skla.dot(G,gpuarray.to_gpu(np.ascontiguousarray(C,dtype=np.float32)))
     #final steps--

     D0 = misc.ones((self.n_index,self.mesh.nle),dtype=np.float32)
     Dtot = D0 + D + DB

     #building final arrays---
     F2 = gpuarray.zeros((np.shape(Am2)[0],np.shape(Am2)[1]+np.shape(Dtot)[1]),dtype=np.float32)
     F2[:,0:np.shape(Am2)[1]] = Am2
     F2[:,np.shape(Am2)[1]:] = Dtot
     Sparse = F2.copy()

     rows = gpuarray.zeros(len(self.i) + self.mesh.nle,dtype=np.int32)
     rows[0:len(self.i)] = gpuarray.to_gpu(np.ascontiguousarray(self.i,dtype=np.int32))
     rows[len(self.i):]  = gpuarray.arange(0,self.mesh.nle,dtype=np.int32)

     cols = gpuarray.zeros(len(self.j) + self.mesh.nle,dtype=np.int32)
     cols[0:len(self.j)] = gpuarray.to_gpu(np.ascontiguousarray(self.j,dtype=np.int32))
     cols[len(self.j):]  = gpuarray.arange(0,self.mesh.nle,dtype=np.int32)
     #---------------------------------------------------

     #Periodic boundary conditions
     Gb = skla.dot(control_angle,gpuarray.to_gpu(np.ascontiguousarray(self.pv,dtype=np.float32)))
     Gbm = 0.5 * (Gb - (1.0-delta)*pycuda.cumath.fabs(Gb)) #get only the negative part

     skla.dot_diag(gpuarray.to_gpu(np.ascontiguousarray(self.dp,dtype=np.float32)),Gbm,trans='T',overwrite=True)

     P = skla.dot(Gbm,gpuarray.to_gpu(np.ascontiguousarray(CC,dtype=np.float32)))

     #Boundary-----------
     TB_gpu = gpuarray.to_gpu(np.ascontiguousarray(TB_new,dtype=np.float32))
     Tnew_gpu = gpuarray.to_gpu(np.ascontiguousarray(Tnew,dtype=np.float32))
     DD = gpuarray.to_gpu(np.ascontiguousarray(DD,dtype=np.float32))
     skla.dot_diag(TB_gpu,DD,overwrite=True)
     B2 = skla.dot(G-Gp,DD)

     #------------------
     TNEW = gpuarray.zeros((self.n_index,self.mesh.nle),dtype=np.float32)
     for n in range(self.n_index):
       TNEW[n] = Tnew_gpu
     #--------------------

     TL_gpu = gpuarray.to_gpu(np.ascontiguousarray(TL,dtype=np.float32))
     RHS2 = B2 + TL_gpu + P + TNEW

     F3 = gpuarray.zeros_like(F2)
     for k in range(self.mesh.nle):   
       F3[:,k] = F2[:,int(rot[k])]

     S = MSparse(indptr,indices,self.mesh.nle,self.n_index-3,new=True)
     S.add_LHS(F2[3:])
     dx = S.solve(RHS2[3:]).get()
     t2 = time.time()

     xf = np.array([x2[E] for x2 in dx])
     print(np.allclose(xs,xf))
 
     S.free_memory()
     quit()
     #for n in np.arange(3,self.n_index):
     # A = sp.csr_matrix( (F[n],(rows,cols)), shape=(self.mesh.nle,self.mesh.nle) )
     # S.add_csr_matrix(A,RHS[n])
     '''
     while n_iter < argv.setdefault('max_bte_iter',10) and \
          error_vec[-1] > argv.setdefault('max_bte_error',1e-2):


         #S = MSparse()
         yyy = time.time()
         for global_index in range(self.n_index):
          index_irr = self.mat[0]['temp_vec'][global_index]
          mfp   = self.mat[0]['mfp'][global_index]
          irr_angle = self.mat[0]['angle_map'][global_index]
          if np.linalg.norm(self.control_angle[global_index]) > 0.9:
           t1 = time.time()
           aa = sparse.COO([0,1,2],self.control_angle[global_index],shape=(3))

           #this has to be accellerated------
           test2  = sparse.tensordot(self.mesh.N,aa,axes=1)
           AP = test2.clip(min=-1e-9)
           AM = test2.clip(max=1e-9)
           #-------------
           tmp = sparse.tensordot(self.mesh.CM,aa,axes=1)
           HW_PLUS  = tmp.clip(min=-1e-9)
           HW_MINUS  = tmp.clip(max=1e-9)
           G = mfp*AP.sum(axis=1).todense() + mfp*HW_PLUS.sum(axis=1).todense() + np.ones(self.mesh.nle)
           AD = spdiags(G,0,self.mesh.nle,self.mesh.nle,format='csr')
           F = AM.tocsr() * mfp + AD
           #self.P = np.sum(np.multiply(AM,self.mesh.B),axis=1).todense()
           #boundary = np.sum(np.multiply(TB,HW_MINUS.todense()),axis=1)
           #RHS = mfp * (self.P + boundary) + Tnew + TL[index_irr]
           #-----------------------------------------------------

           #S.add_csr_matrix(F,RHS)

           #s = sp.linalg.spsolve(F,RHS)
         t2 = time.time()
         print('assembly:',t2-yyy)
         #T = S.solve() 
         #t3 = time.time()
         #print('Solving',t3-t2)
         quit()
         



           #create a list-----------------
           #master_data += list(F.data)
           #master_data = list(F.data)
           #a = F.nonzero()
           #nn = len(a[0])
           #g = [[global_index] * nn]
           #g = []
           #g.append(list(a[0]))
           #g.append(list(a[1]))
           #g = list(map(list, zip(*g)))
           #master_A += g
           #master_A = g
           #master_b.append(RHS)
           #master_b = [RHS]
           #-------------------------------

           #i = torch.LongTensor(master_A)
           #v = torch.FloatTensor(master_data)
           #A = torch.sparse.FloatTensor(i.t(), v, torch.Size([self.n_index,self.mesh.nle,self.mesh.nle])).to_dense() 
           #A = torch.sparse.FloatTensor(i.t(), v, torch.Size([self.mesh.nle,self.mesh.nle])).cuda().to_dense() 
           #b = torch.tensor(list(map(list, zip(*master_b)))).cuda()

           #t1 = time.time()
           #print(t1-yyy)
           #A_LU = A.lu()
           #x = torch.lu_solve(b,*A_LU)
           #t2 = time.time()
           #print(t2-t1)

'''



    
  def assemble_fourier_new(self):


   if  MPI.COMM_WORLD.Get_rank() == 0:
    F = sp.dok_matrix((self.mesh.nle,self.mesh.nle))
    B = np.zeros(self.mesh.nle)

    for ll in self.mesh.side_list['active']:
      (i,j) = self.mesh.side_elem_map[ll]
      vi = self.mesh.get_elem_volume(i)
      vj = self.mesh.get_elem_volume(j)
      kappa = self.get_kappa(i,j)
      (v_orth,dummy) = self.mesh.get_decomposed_directions(i,j,rot=kappa)
      li = self.mesh.g2l[i]; lj = self.mesh.g2l[j]

      if not i == j:
       F[li,li] += v_orth/vi#*kappa
       F[li,lj] -= v_orth/vi#*kappa

       F[lj,lj] += v_orth/vj#*kappa
       F[lj,li] -= v_orth/vj#*kappa

       B[li] += self.mesh.get_side_periodic_value(j,i)*v_orth/vi#*kappa
       B[lj] += self.mesh.get_side_periodic_value(i,j)*v_orth/vj#*kappa
      else:
       if ll in self.mesh.side_list['Hot']:
        F[li,li] += v_orth/vi
        B[li] += v_orth/vi*0.5#*kappa

       if ll in self.mesh.side_list['Cold']:
        F[li,li] += v_orth/vi
        B[li] -= v_orth/vi*0.5#*kappa

    data = {'F':F.tocsc(),'B':B}
   else: data = None
   data = MPI.COMM_WORLD.bcast(data,root=0)
   self.B = data['B']
   self.F = data['F']


  #def print_bulk_kappa(self):
      
    #if MPI.COMM_WORLD.Get_rank() == 0:
    # if self.verbose:
      #print('  ')
      #for region in self.region_kappa_map.keys():
       #print('Kappa bulk: ')
       #for mat in self.mat:
       #    print('{:8.2f}'.format(mat['kappa_bulk_tot'])+ ' W/m/K')   
       #    print(' ')
       #if 'Inclusion' in self.mesh.region_elem_map.keys():
       # print('Kappa inclusion: ')
       # print('{:8.2f}'.format(self.mat['kappa_inclusion'])+ ' W/m/K')   
       # print(' ')
       #print(region.ljust(15) +  '{:8.2f}'.format(self.region_kappa_map[region])+ ' W/m/K')    

  def get_kappa(self,i,j):

   if i ==j:
    return np.array(self.elem_kappa_map[i])*np.eye(3)

   side = self.mesh.get_side_between_two_elements(i,j)
   w = self.mesh.get_interpolation_weigths(side,i)
   normal = self.mesh.get_normal_between_elems(i,j)

   kappa_i = np.array(self.elem_kappa_map[i])*np.eye(3)
   kappa_j = np.array(self.elem_kappa_map[j])*np.eye(3)

   ki = np.dot(normal,np.dot(kappa_i,normal))
   kj = np.dot(normal,np.dot(kappa_j,normal))

   
   kappa = kj*kappa_i/(ki*(1-w) + kj*w)

 
   return kappa


  def build_kappa_mask(self):
    self.kappa_mask = []
    for kc1,kc2 in zip(*self.mesh.A.nonzero()):
      tt = self.mesh.get_side_periodic_value(self.mesh.l2g[kc2],self.mesh.l2g[kc1])
      if tt == 1.0:
       self.kappa_mask.append([kc1,kc2])

      ll = self.mesh.get_side_between_two_elements(self.mesh.l2g[kc1],self.mesh.l2g[kc2])

    for ll in self.mesh.side_list['Cold']:
       self.kappa_mask.append(self.mesh.side_elem_map[ll])
        
    

  def assemble_fourier(self) :

   if  MPI.COMM_WORLD.Get_rank() == 0:
    row_tmp = []
    col_tmp = []
    data_tmp = []
    #row_tmp_b = []
    #col_tmp_b = []
    #data_tmp_b = []
    #data_tmp_b2 = []
    B = np.zeros(self.n_elems)
    #kappa_mask = []
    for kc1,kc2 in zip(*self.mesh.A.nonzero()):
     kappa = self.get_kappa(kc1,kc2)
     #kappa = 1.0
     (v_orth,dummy) = self.mesh.get_decomposed_directions(kc1,kc2)
     vol1 = self.mesh.get_elem_volume(kc1)
     row_tmp.append(kc1)
     col_tmp.append(kc2)
     data_tmp.append(-v_orth/vol1*kappa)
     row_tmp.append(kc1)
     col_tmp.append(kc1)
     data_tmp.append(v_orth/vol1*kappa)
     #----------------------
     #row_tmp_b.append(kc1)
     #col_tmp_b.append(kc2)
     #tt = self.mesh.get_side_periodic_value(kc2,kc1)
     #data_tmp_b.append(tt*v_orth*kappa)
     #data_tmp_b2.append(tt)
     #if tt == 1.0:
     # kappa_mask.append([kc1,kc2])
     
     #---------------------
     B[kc1] += self.mesh.get_side_periodic_value(kc2,kc1)*v_orth/vol1*kappa

    #Boundary_elements
    FF = np.zeros((self.n_elems,self.n_side_per_elem))
    #boundary_mask = np.zeros(self.n_elems)
    for side in self.mesh.side_list['Boundary']:
     elem = self.mesh.side_elem_map[side][0]
     side_index = np.where(np.array(self.mesh.elem_side_map[elem])==side)[0][0]
     vol = self.mesh.get_elem_volume(elem)
     area = self.mesh.get_side_area(side)
     FF[elem,side_index] = area/vol
     #boundary_mask[elem] = 1.0

    F = csc_matrix((np.array(data_tmp),(np.array(row_tmp),np.array(col_tmp))), shape=(self.n_elems,self.n_elems) )
    #RHS = csc_matrix( (np.array(data_tmp_b),(np.array(row_tmp_b),np.array(col_tmp_b))), shape=(self.n_elems,self.n_elems) )
    #PER = csc_matrix( (np.array(data_tmp_b2),(np.array(row_tmp_b),np.array(col_tmp_b))), shape=(self.n_elems,self.n_elems) )

    #-------
    data = {'F':F,'B':B,'FF':FF}
   else: data = None
   data =  MPI.COMM_WORLD.bcast(data,root=0)
   self.F = data['F']
   #self.RHS = data['RHS']
   #self.PER = data['PER']
   #self.kappa_mask = data['kappa_mask']
   self.B = data['B']
   self.FF = data['FF']
   #self.boundary_mask = data['boundary_mask']


  def solve_modified_fourier_law(self,kappa,**argv):

    A  = kappa * self.Fm 
    B  = kappa * self.Bm
    G = argv['G']   
    TB = argv['TB']   
    tmp = np.dot(self.FF,G) #total flux from boundaries
    A += spdiags(tmp,0,self.n_elems,self.n_elems) 
    B += np.einsum('ci,ci,i->c',self.FF,TB,G)
    A +=  csc_matrix(scipy.sparse.eye(self.n_elems)) 
    B += argv['TL']

    
    SU = splu(A)
    C = np.zeros(self.n_elems)
    
    n_iter = 0
    kappa_old = 0
    error = 1        
    while error > self.argv.setdefault('max_fourier_error',1e-2) and \
                  n_iter < self.argv.setdefault('max_fourier_iter',10) :
                    
        RHS = B + C*kappa
        
        temp = SU.solve(RHS)
        
        temp = temp - (max(temp)+min(temp))/2.0
        (C,grad) = self.compute_non_orth_contribution(temp)
        
        kappa_eff = self.compute_diffusive_thermal_conductivity(temp)*kappa
        error = abs((kappa_eff - kappa_old)/kappa_eff)
        kappa_old = kappa_eff
        n_iter +=1

    s = kappa_eff
    

    return s,temp.copy(),grad




  def get_diffusive_suppression_function(self,**argv):

    A  = self.F
    #A.todense().dump('test.np')
    #D = csc_matrix(np.load('test.np',allow_pickle=True))
    #print(np.shape(D))

    B  = self.B

    if 'heat_source' in self.argv.keys(): 
     for elem in self.mesh.region_elem_map['Inclusion']: 
      B[elem] += self.argv['heat_source']


    #scaling
    #ss = np.array(A.max(axis=1).todense().T)[0]
    #for i in range(np.shape(A)[0]):
    #    A[i,:] /= ss[i]
        
    SU = splu(A)

    C = np.zeros(self.mesh.nle)
   
    n_iter = 0
    kappa_old = 0
    error = 1  
    if self.structured:
        self.argv['max_fourier_iter'] = 1
    while error > self.argv.setdefault('max_fourier_error',1e-2) and \
                  n_iter < self.argv.setdefault('max_fourier_iter',10) :
    
        RHS = B + C

        #scaling---
        #RHS = np.array([ RHS[i]/ss[i]  for i in range(len(RHS))])

        temp = SU.solve(RHS)
        
        temp = temp - (max(temp)+min(temp))/2.0

        kappa_eff = self.compute_diffusive_thermal_conductivity(temp)
        if kappa_eff == 0:
            error = 1
        else:    
         error = abs((kappa_eff - kappa_old)/kappa_eff)
        kappa_old = kappa_eff
        n_iter +=1

        temp_int = self.compute_interfacial_temperature(temp)
        (C,grad) = self.compute_non_orth_contribution(temp,**{'temp_interface':temp_int})
       

    s = kappa_eff#/kappa

    #flux_int = self.compute_interfacial_flux(grad,temp,temp_int)

    #quit()
    return s,temp.copy(),grad,temp_int,None

  def compute_interfacial_flux(self,grad,temp,temp_int):
     
     flux_int = {}
    
     #quit()
     for ll in self.mesh.side_list['Interface'] :

       (i,j) = self.mesh.side_elem_map[ll]
      
       (v_orth,v_non_orth) = self.mesh.get_decomposed_directions(i,j)
       area = self.mesh.get_side_area(ll)
       normal = self.mesh.compute_side_normal(i,ll)

       w = self.mesh.get_interpolation_weigths(ll,i)
       Tj = temp[j]
       Ti = temp[i]
       Tb = temp_int[ll]
       Cj = self.mesh.get_elem_centroid(j)
       Ci = self.mesh.get_elem_centroid(i)
      
       Fi = -grad[i]*self.elem_kappa_map[i]
       Fj = -grad[j]*self.elem_kappa_map[j]

       F0 = -(Tj-Ti)*v_orth/area*normal*self.get_kappa(i,j)
       #Fjn = -(Tj-Ti)*v_orth/area*normal*self.get_kappa(i,j)

       no = np.cross(normal,[0,0,1])
       #no = v_non_orth/np.linalg.norm(v_non_orth)

       Fino = np.dot(Fi,np.outer(no,no))
       Fjno = np.dot(Fj,np.outer(no,no))
       
       Fin = np.dot(Fi,np.outer(normal,normal))
       Fjn = np.dot(Fj,np.outer(normal,normal))

       #F0 = w*Fin + (1-w)*Fjn
       

       Ji = F0  + Fino
       Jj = F0  + Fjno
       


       flux_int[ll] = [Ji,Jj]

     return flux_int

  def compute_interfacial_temperature(self,temp):
     
     int_temp = {}
     
     for ll in self.mesh.side_list['Interface'] :

       (i,j) = self.mesh.side_elem_map[ll]
       w = self.mesh.get_interpolation_weigths(ll,i)
       Ti = temp[i]
       Tj = temp[j]

       normal = self.mesh.get_normal_between_elems(i,j)
       kappa_i = np.array(self.elem_kappa_map[i])
       kappa_j = np.array(self.elem_kappa_map[j])
       ki = np.dot(normal,np.dot(kappa_i,normal))
       kj = np.dot(normal,np.dot(kappa_j,normal))

       Tb = (kj*w*Tj + ki*(1-w)*Ti)/(kj*w + ki*(1-w))
      

       int_temp[ll] = [Tb,Tb]

     return int_temp


  def assemble_modified_fourier(self) :

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
     (v_orth,dummy) = self.mesh.get_decomposed_directions(kc1,kc2)
     vol1 = self.mesh.get_elem_volume(kc1)
     row_tmp.append(kc1)
     col_tmp.append(kc2)
     data_tmp.append(-v_orth/vol1)
     row_tmp.append(kc1)
     col_tmp.append(kc1)
     data_tmp.append(v_orth/vol1)
     #----------------------
     row_tmp_b.append(kc1)
     col_tmp_b.append(kc2)
     tt = self.mesh.get_side_periodic_value(kc2,kc1)
     data_tmp_b.append(tt*v_orth)
     data_tmp_b2.append(tt)
     if tt == 1.0:
      kappa_mask.append([kc1,kc2])
     
     #---------------------
     B[kc1] += self.mesh.get_side_periodic_value(kc2,kc1)*v_orth/vol1

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
   self.Fm = data['F']
   self.RHSm = data['RHS']
   #self.PERm = data['PER']
   self.kappa_maskm = data['kappa_mask']
   self.Bm = data['B']
   self.FFm = data['FF']



      
  def compute_non_orth_contribution(self,temp,**argv) :

    gradT = self.mesh.compute_grad(temp,interfacial_temperature = argv['temp_interface'])
    
    C = np.zeros(self.mesh.nle)
    for i,j in zip(*self.mesh.A.nonzero()):

     gi = self.mesh.l2g[i]     
     gj = self.mesh.l2g[j]     

     if not i==j:
      #Get agerage gradient----
      side = self.mesh.get_side_between_two_elements(gi,gj)

      w = self.mesh.get_interpolation_weigths(side,gi)
      #grad_ave = w*gradT[i] + (1.0-w)*gradT[j]
      #F_ave = w*gradT[i]*self.elem_kappa_map[gi] + (1.0-w)*gradT[j]*self.elem_kappa_map[gj]
      F_ave = w*np.dot(gradT[i],self.elem_kappa_map[gi]) + (1.0-w)*np.dot(gradT[j],self.elem_kappa_map[gj])
      #------------------------
      (dumm,v_non_orth) = self.mesh.get_decomposed_directions(gi,gj)

      if not 'kappa' in argv.keys():
       C[i] += np.dot(F_ave,v_non_orth)/2.0/self.mesh.get_elem_volume(gi)
       C[j] -= np.dot(F_ave,v_non_orth)/2.0/self.mesh.get_elem_volume(gj)
      else: 
       C[i] += np.dot(grad_ave,v_non_orth)/2.0/self.mesh.get_elem_volume(gi)*kappa
       C[j] -= np.dot(grad_ave,v_non_orth)/2.0/self.mesh.get_elem_volume(gj)*kappa

    return C, gradT


  #def compute_inho_diffusive_thermal_conductivity(self,temp,mat=np.eye(3)):

  # kappa = 0
  # for i,j in zip(*self.PER.nonzero()):

    #(v_orth,dummy) = self.mesh.get_decomposed_directions(i,j,rot=mat)
   
    
 #   kappa += 0.5*v_orth *self.PER[i,j]*(temp[j]+self.PER[i,j]-temp[i])#*kappa_b
#
   #return kappa*self.kappa_factor


  def compute_diffusive_thermal_conductivity(self,temp,mat=np.eye(3),kappa_bulk = -1):

    
   kappa = 0
   for (i,j) in self.kappa_mask:
    gi = self.mesh.l2g[i]   
    gj = self.mesh.l2g[j]   

    k = self.get_kappa(gi,gj)
    (v_orth,dummy) = self.mesh.get_decomposed_directions(gi,gj,rot=k)
    
    if kappa_bulk == -1:
     kk = 1
    else:
     kk = kappa_bulk 
     #normal = self.mesh.get_normal_between_elems(i,j)
     #k = np.dot(normal,np.dot(self.get_kappa(gi,gj),normal))
    
    if not i == j:
      Ti = temp[i]
      Tj = temp[j] + 1
    else:
      Ti = -temp[i]   
      side = self.mesh.get_side_between_two_elements(i,j)
      if side in self.mesh.side_list['Cold']:
         Tj = 0.5 #we do the opposite
      else:
         print('error in mask')
         quit()
    kappa += v_orth * (Tj-Ti)*kk
   
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
     print('Space DOFs:      ' + str(len(self.mesh.l2g)))
     #print('Azimuthal angles:  ' + str(self.mat['n_theta']))
     #print('Polar angles:      ' + str(self.mat['n_phi']))
     #print('Momentum DOFs:   ' + str(self.n_serial*self.n_parallel))
     #print('Bulk Thermal Conductivity:   ' + str(round(self.mat['kappa_bulk_tot'],4)) +' W/m/K')
     print(' ')
   
