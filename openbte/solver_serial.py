from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import time
from scipy.sparse import csc_matrix
from mpi4py import MPI
import shutil
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import lgmres
import matplotlib.pylab as plt
from scipy.sparse import diags
import scipy.io
import deepdish as dd
#from termcolor import colored
from .geometry import Geometry
from .material import Material


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
   path = '.'
   files = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and \
         'material' in i]

   self.materials = {}
   for filename in files:
    mat = Material(model='load',filename=filename)

    gamma = mat.state['kappa_bulk_tot']/mat.state['mfp_bulk'][0]/mat.state['mfp_bulk'][0]

    mat.state.update({'gamma':gamma})

    self.materials.update({mat.state['region']:mat.state})
    self.B0 = mat.state['B0']
    self.B1 = mat.state['B1']
    self.B2 = mat.state['B2']
    self.kappa_bulk = mat.state['kappa_bulk_tot']
   #-----------------------------------------------------

   
   self.lu = {}

   self.mfp = np.array(mat.state['mfp_sampled'])*1e9 #In nm
   self.n_mfp = len(self.mfp)

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

   #apply symmetry--------
   if self.dim == 2:
    self.n_theta_irr = int(self.n_theta/2)
    self.symmetry = 2.0
    self.n_index = self.n_phi
   else:
    self.n_theta_irr = 1
    self.symmetry = 1.0
    self.n_index = self.n_phi * self.n_theta
   #---------------------

   #if self.compute_matrix:
   directory = self.cache
   if os.path.exists(directory):
     shutil.rmtree(directory,ignore_errors=True)
   os.makedirs(directory)
   argv.setdefault('max_bte_iter',10)

   #if argv['max_bte_iter'] > 0:
       
   self.compute_directional_connections()
   
   

   
   self.compute_space_dependent_kappa_bulk()

   self.assemble_fourier()

   #
   #quit()

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

  def compute_space_dependent_kappa_bulk(self):

   if MPI.COMM_WORLD.Get_rank() == 0:

    side_kappa_map = {}
    elem_kappa_map = {}
    region_elem_map = {}
    region_kappa_map = {}
    for ll in self.mesh.side_list['active']:
     (elem1,elem2) = self.mesh.side_elem_map[ll]
     region_1 = self.mesh.get_region_from_elem(elem1)
     kappa_1 = self.materials[region_1]['kappa_bulk_tot']


     if not region_1 in region_kappa_map.keys():
      region_kappa_map.update({region_1:kappa_1})

     if not ll in self.mesh.side_list['Boundary'] :

      region_2 = self.mesh.get_region_from_elem(elem2)
      kappa_2 = self.materials[region_2]['kappa_bulk_tot']

      s =  self.mesh.get_interpolation_weigths(ll,elem1)
      L = self.mesh.get_distance_between_centroids_of_two_elements_from_side(ll)
      L_1 = s*L
      L_2 = (1.0 - s)*L
      if  kappa_1 == 0.0 or kappa_2 == 0.0:
       kappa = 1e-3
      else:
       kappa = L*kappa_1 * kappa_2/(kappa_2 * L_1 + L_2 * kappa_1)# +  kappa_1 * kappa_2/self.interface_conductance)

      #print(kappa_1,kappa_2,kappa)

      side_kappa_map.update({ll:kappa})

      if not elem1 in elem_kappa_map.keys():
       elem_kappa_map.update({elem1:kappa_1})
      if not elem2 in elem_kappa_map.keys():
       elem_kappa_map.update({elem2:kappa_2})

      if not region_2 in region_kappa_map.keys():
       region_kappa_map.update({region_2:kappa_2})

     else:
      (elem1,elem2) = self.mesh.side_elem_map[ll]
      if not elem1 in elem_kappa_map.keys():
        elem_kappa_map.update({elem1:kappa_1})
      side_kappa_map.update({ll:0.0})

    data = {'side_kappa_map':side_kappa_map,'elem_kappa_map':elem_kappa_map,\
              'region_elem_map':region_elem_map,'region_kappa_map':region_kappa_map}

   else: data = None

   data =  MPI.COMM_WORLD.bcast(data,root=0)

   self.side_kappa_map = data['side_kappa_map']
   self.elem_kappa_map = data['elem_kappa_map']
   self.region_elem_map = data['region_elem_map']
   self.region_kappa_map = data['region_kappa_map']




  def compute_directional_diffusion(self,index):

    t = int(index/self.n_phi)
    p = index%self.n_phi
    ss = self.dom['ss'][t][p]
    row_tmp = []
    col_tmp = []
    data_tmp = []
    for kc1,kc2 in zip(*self.mesh.A.nonzero()):
     ll = self.mesh.get_side_between_two_elements(kc1,kc2)
     (v_orth,dummy) = self.mesh.get_decomposed_directions(kc1,kc2,rot=ss)
     row_tmp.append(kc1)
     col_tmp.append(kc2)
     data_tmp.append(self.mesh.get_side_periodic_value(ll,kc2)*v_orth)
    RHS = csc_matrix( (np.array(data_tmp),(np.array(row_tmp),np.array(col_tmp))), shape=(self.n_elems,self.n_elems) )

    scipy.io.mmwrite(self.cache + '/RHS_DIFF_' + str(t) + '_' + str(p) + r'.mtx',RHS)


  def get_material_properties(self,elem):
    region = self.mesh.get_region_from_elem(elem)
    gamma = self.materials[region]['gamma']*1e-18
    mfp = self.materials[region]['mfp_bulk'][0]*1e9 # meter

    return gamma,mfp


  def compute_directional_connections(self):

    for index in range(self.n_index):

     r=[];c=[];d=[]
     rk=[]; ck=[]; dk=[]
     P = np.zeros(self.n_elems)
     HW_minus = np.zeros(self.n_elems)
     HW_plus = np.zeros(self.n_elems)
     for i,j in zip(*self.mesh.A.nonzero()):
      (cm,cp,cmb,cpb) = self.mesh.get_angular_coeff_old(i,j,index)
      r.append(i); c.append(i); d.append(cp) 
      v = self.mesh.get_side_periodic_value(i,j)
      rk.append(i); ck.append(j);
      dk.append(v*(cp+cm)*self.mesh.get_elem_volume(i))
      r.append(i); c.append(j);  d.append(cm)
      
      P[i] += v*cm
      HW_plus[i]  += cpb/np.pi
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

         
   #First Guess---------------------------  
   argv.update({'kappa':[self.kappa_bulk]})
   
   output = self.solve_fourier(**argv) #simple fourier----   
   temperature_fourier = np.tile(output['temperature_fourier'],(self.n_mfp,1))
   temperature_fourier_gradient = np.tile(output['temperature_fourier_gradient'],(self.n_mfp,1,1))
   ms = np.zeros((self.n_mfp,self.n_theta,self.n_phi))
   TB = temperature_fourier.copy()
   TL = temperature_fourier.copy()
   kappa_fourier =  np.tile(output['kappa_fourier'],(self.n_mfp))
  

   TLtot,TBtot  = np.zeros((2,self.n_mfp,self.n_elems))
   
   SUP = np.zeros((self.n_mfp,self.n_theta,self.n_phi))
   FLUX = np.zeros((self.n_el,3))
   kappa_old = 0.0
   n_iter = 0
   error = 1.0
   kappa_mfe = np.square(self.mfp)/3.0

   start = time.time()
   while n_iter < argv.setdefault('max_bte_iter',10): # and \
#          error > argv.setdefault('max_bte_error',1e-2):


    #solve_fourier
    #if n_iter ==  argv.setdefault('max_bte_iter',10)-1:
    # argv.update({'mfe_factor':1.0,\
    #               'verbose':False,\
    #               'lattice_temperature':TL.copy(),\
    #               'boundary_temperature':TB.copy(),\
    #               'kappa':kappa_mfe,\
    #               'interface_conductance': self.mfp/2.0})    
    # output = self.solve_fourier(**argv) 
    # kappa_fourier = output['kappa_fourier']
    # sup_zero = [self.compute_diffusive_thermal_conductivity(TL[n].copy(),mat = np.eye(3)) for n in range(self.n_mfp)]        
    #-


    SUP = np.zeros((self.n_mfp,self.n_theta,self.n_phi))
    #SUPms = np.zeros((self.n_mfp,self.n_theta,self.n_phi))
    SUPd = np.zeros((self.n_mfp,self.n_theta,self.n_phi))
    #SUPb = np.zeros((self.n_mfp,self.n_theta,self.n_phi))
    TLp,TBp  = np.zeros((2,self.n_mfp,self.n_elems))
    FLUXp = np.zeros((self.n_el,3))

    
    
    #solve_BTE
    for index in range(self.n_index):
     #Read directional information-------------------------------------------------- 
     A = scipy.io.mmread(self.cache + '/A_' + str(index) + '.mtx').tocsc()
     P = np.load(open(self.cache + '/P_' + str(index) +r'.np','rb'))
     HW_MINUS = np.load(open(self.cache + '/HW_MINUS_' + str(index) +r'.np','rb'))
     HW_PLUS = np.load(open(self.cache + '/HW_PLUS_' + str(index) +r'.np','rb'))
     K = scipy.io.mmread(self.cache + '/K_' + str(index) + '.mtx').tocsc()
      #-----------------------------------------------------------------------------

     #old_temp = TL[0].copy()
     for tt in range(self.n_theta_irr): #We take into account the symmetry
          
      if self.dim == 2:
       t = tt
       p = index
       theta_factor = self.dom['at'][t] / self.dom['d_theta_vec'][t] 
      else:
       t = int(index/self.n_phi)
       p = index%self.n_phi
       theta_factor = 1.0
        
      pre_factor = 3.0/4.0/np.pi*0.5*theta_factor * self.dom['d_omega'][t][p]  
         
      #dif = False   
      for m in range(self.n_mfp)[::-1]:
        
       D = np.multiply(TB[m],HW_MINUS)
       F = scipy.sparse.eye(self.n_elems) + theta_factor  * A * self.mfp[m]
       RHS = self.mfp[m]*theta_factor * (P + D) + TL[m]
       
       

       lu = splu(F.tocsc())   
       temp = lu.solve(RHS) 
 
       sup = pre_factor * K.dot(temp-TL[m]).sum()*self.kappa_factor/self.mfp[m]
       
       #if n_iter ==  argv.setdefault('max_bte_iter',10)-1:
       # sup_dif = self.compute_diffusive_thermal_conductivity(temp.copy(),mat = self.dom['ss2'][t][p])*3.0/4.0/np.pi 
       # SUPd[m,t,p] += sup_dif
       # SUPd[m,self.n_theta -t -1,p] += SUPd[m,t,p]
       #if not sup == 0:
       # if dif:
       #  sup = self.compute_diffusive_thermal_conductivity(temp.copy(),mat = self.dom['ss2'][t][p])*3.0/4.0/np.pi 
       # else:    
       #  er = abs((sup-sup_dif)/sup)   
       #  if abs((sup-sup_dif)/sup)< 5e-1:
       #   sup = sup_dif
       #   dif = True



       SUP[m,t,p] += sup

       TLp += np.outer(self.B2[:,m],temp)*self.dom['d_omega'][t][p]*self.symmetry/4.0/np.pi    
        
          
       #--------------------------
       if self.symmetry == 2.0:
         FLUXp += np.mean(self.B1[:,m])*np.outer(temp,np.multiply(self.dom['S'][t][p],[2.0,2.0,0.0]))
         SUP[m,self.n_theta -t -1,p] += SUP[m,t,p]
         TBp += self.dom['at'][t]*self.symmetry*np.outer(self.B1[:,m],np.multiply(temp,HW_PLUS))
       else:
         FLUXp += np.mean(self.B1[:,m])*np.outer(temp,self.dom['S'][t][p])
         TBp[m] += np.multiply(temp,HW_PLUS)*self.symmetry
 
     
        
    kappa = sum([self.B0[m]*SUP[m,:,:].sum() for m in range(np.shape(SUP)[0])])*self.kappa_bulk
    #sup_m =  [SUP[m,:,:].sum() for m in range(np.shape(SUP)[0])]
    n_iter +=1
    error = abs(kappa-kappa_old)/abs(kappa)
    kappa_old = kappa
    TL = TLp.copy()
    TB = TBp.copy()
    print(min(TB[0]),max(TB[0]))
    
    print(kappa)
    #check---
   #sup_d =  [SUPd[m,:,:].sum() for m in range(np.shape(SUPd)[0])]
   #print(time.time()-start)
   #plt.plot(self.mfp,sup_m,'g')  
   #plt.plot(self.mfp,sup_mb,'r')  
   #plt.plot(self.mfp,sup_d,'r')  
   #plt.plot(self.mfp,sup_md,'b')  
   #plt.plot(self.mfp,sup_ms,'r')  
   #plt.legend(['bal','dif'])
   #plt.plot(self.mfp,sup_zero,'k')  
   #plt.plot(self.mfp,kappa_fourier/kappa_mfe)
   #plt.xscale('log')
   #plt.ylim([0,1])
   #plt.show()
     
    
    

  def print_bulk_kappa(self):
      
    if MPI.COMM_WORLD.Get_rank() == 0:
     if self.verbose:
      print('  ')
      for region in self.region_kappa_map.keys():
       print(region.ljust(15) +  '{:8.2f}'.format(self.region_kappa_map[region])+ ' W/m/K')    

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
    self.kappa_mask = []
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
      self.kappa_mask.append([kc1,kc2])
     
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
    data = {'F':F,'RHS':RHS,'B':B,'PER':PER,'boundary_mask':boundary_mask,'FF':FF}
   else: data = None
   data =  MPI.COMM_WORLD.bcast(data,root=0)
   self.F = data['F']
   self.RHS = data['RHS']
   self.PER = data['PER']
   self.B = data['B']
   self.FF = data['FF']
   self.boundary_mask = data['boundary_mask']


  def solve_fourier(self,**argv):
       
        
    #Update matrices---------------------------------------------------
    G = argv.setdefault('interface_conductance',np.zeros(1))*self.dom['correction']
    TL = argv.setdefault('lattice_temperature',np.zeros((1,self.n_el)))
    TB = argv.setdefault('boundary_temperature',np.zeros((1,self.n_el)))
    mfe_factor = argv.setdefault('mfe_factor',0.0)
    kappa_bulk = argv.setdefault('kappa',[1.0])
    #print(kappa_bulk)

    #print(kappa_bulk)
    #G *=5.0
    n_index = len(kappa_bulk)
    #G = np.zeros(n_index)
    #-------------------------------
    KAPPAp,KAPPA = np.zeros((2,n_index))
    FLUX,FLUXp = np.zeros((2,n_index,3,self.n_elems))
    TLp,TLtot = np.zeros((2,n_index,self.n_elems))

    
    #set up MPI------------------
    for index in range(n_index):
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

      
      while error > argv.setdefault('max_fourier_error',1e-11) and \
                    n_iter < argv.setdefault('max_fourier_iter',10) :
        RHS = B + C*kappa_bulk[index]
        if mfe_factor == 0.0: RHS[10] = 0.0      
        temp = SU.solve(RHS)
        temp = temp - (max(temp)+min(temp))/2.0
        (C,flux) = self.compute_non_orth_contribution(temp)
        #KAPPAp[index] = self.compute_inho_diffusive_thermal_conductivity(temp)*kappa_bulk[index]
        #start = time.time()
        KAPPAp[index] = self.compute_diffusive_thermal_conductivity(temp)*kappa_bulk[index]
        #print(time.time()-start)
        error = abs((KAPPAp[index] - kappa_old)/KAPPAp[index])
        kappa_old = KAPPAp[index]
        n_iter +=1
        
      #FLUXp = np.array([-self.elem_kappa_map[k]*tmp for k,tmp in enumerate(flux)])
      FLUXp[index] = flux.T
      TLp[index] = temp
      
    if argv.setdefault('verbose',True):
     print('  ')
     print('Thermal Conductivity: '.ljust(20) +  '{:8.2f}'.format(KAPPAp.sum())+ ' W/m/K')
     print('  ')
    


    return {'kappa_fourier':KAPPAp,'temperature_fourier_gradient':FLUX,'temperature_fourier':TLp}

      
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

    ll = self.mesh.get_side_between_two_elements(i,j)
    kappa_b = self.side_kappa_map[ll]
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
     print('Polar angles:      ' + str(self.n_phi))
     print('Mean-free-paths:   ' + str(self.n_mfp))
     print('Degree-of-freedom: ' + str(self.n_mfp*self.n_theta*self.n_phi*self.n_el))



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
