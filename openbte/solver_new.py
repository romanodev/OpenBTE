from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
from scipy.sparse import csc_matrix
from mpi4py import MPI
import shutil
from scipy.sparse.linalg import splu
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
   if MPI.COMM_WORLD.Get_rank() == 0:
    directory = self.cache
    if os.path.exists(directory):
     shutil.rmtree(directory,ignore_errors=True)
    os.makedirs(directory)
   MPI.COMM_WORLD.Barrier()
    #compute matrix--------
   argv.setdefault('max_bte_iter',10)

   #if argv['max_bte_iter'] > 0:
       
   self.compute_directional_connections() 
    #tput = compute_sum(self.compute_directional_connections,self.n_index)
   
   

   
   self.compute_space_dependent_kappa_bulk()

   self.assemble_fourier()

   #
   #quit()

   #---------------------------

   #solve the BTE
   self.compute_function(**argv)


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


  #def compute_directional_connections(self,index,options):
      
  def compute_directional_connections(self):
      
      
   comm = MPI.COMM_WORLD   
   size = comm.size
   rank = comm.rank
   n_index = self.n_index
   block = n_index // size + 1
   for kk in range(block):
    index = rank*block + kk   
    if index < n_index:
  
     r=[];c=[];d=[]
     rk=[]; ck=[]; dk=[]
     P = np.zeros(self.n_elems)
     for i,j in zip(*self.mesh.A.nonzero()):

      #(gamma_i,mfp_i) = self.get_material_properties(i)
      #(gamma_j,mfp_j) = self.get_material_properties(j)

      (cm,cp,dummy,dummy) = self.mesh.get_angular_coeff(i,j,index)
      r.append(i); c.append(i); d.append(cp) #d.append(cp*mfp_i)
      v = self.mesh.get_side_periodic_value(i,j)
      rk.append(i); ck.append(j);
      dk.append(v*(cp+cm)*self.mesh.get_elem_volume(i))

      #dk.append(v*(cp*mfp_i*gamma_i+cm*mfp_j*gamma_j)*self.mesh.get_elem_volume(i))

      r.append(i); c.append(j);  d.append(cm)#d.append(cm*mfp_j*gamma_j/gamma_i)
      
      P[i] += v*cm#*mfp_j*gamma_j/gamma_i

     #------------------------------BOUNDARY-------------------------------
     HW_minus = np.zeros(self.n_elems)
     HW_plus = np.zeros(self.n_elems)
     for elem in self.mesh.boundary_elements:
      (dummy,dummy,cmb,cpb) = self.mesh.get_angular_coeff(elem,elem,index)
      (gamma,mfp) = self.get_material_properties(elem)
      HW_plus[elem] =  cpb/np.pi
      HW_minus[elem] = -cmb#*mfp
     #--------------------------------------------------------------

     #--------------------------WRITE FILES---------------------
     A = csc_matrix( (d,(r,c)), shape=(self.n_elems,self.n_elems) )
     K = csc_matrix( (dk,(rk,ck)), shape=(self.n_elems,self.n_elems) )
     scipy.io.mmwrite(self.cache + '/A_' + str(index) + r'.mtx',A)
     scipy.io.mmwrite(self.cache + '/K_' + str(index) + r'.mtx',K)
     P.dump(open(self.cache + '/P_' + str(index) +r'.np','wb+'))
     HW_minus.dump(open(self.cache +'/HW_MINUS_' + str(index) +r'.np','wb+'))
     HW_plus.dump(open(self.cache + '/HW_PLUS_' + str(index) +r'.np','wb+'))
  MPI.COMM_WORLD.Barrier()


  def solve_bte(self,**argv):

         
   output = self.solve_fourier(**argv) #simple fourier----   
   temperature_fourier = np.tile(output['temperature_fourier'],(self.n_mfp,1))
   temperature_fourier_gradient = np.tile(output['temperature_fourier_gradient'],(self.n_mfp,1,1))
   ms = np.zeros((self.n_mfp,self.n_theta,self.n_phi))
   TB = temperature_fourier.copy()
   TL = temperature_fourier.copy()
   
   TLtot,TBtot  = np.zeros((2,self.n_mfp,self.n_elems))

   kappa_fourier =  np.tile(output['kappa_fourier'],(self.n_mfp))
      
   
   SUP = np.zeros((self.n_mfp,self.n_theta,self.n_phi))
   FLUX = np.zeros((self.n_el,3))
   #ms_1 = 0
   kappa_old = 0.0
   n_iter = 0
   error = 1.0
   comm = MPI.COMM_WORLD  

   block = self.n_index // comm.size + 1
   while n_iter < argv.setdefault('max_bte_iter',10) and \
          error > argv.setdefault('max_bte_error',1e-2):
    
    #print('g')          
    #if comm.rank == 0:
    #   for n in range(self.n_mfp): 
    #    print('ff',min(TL[n]),max(TL[n])) 
    # tmp = self.compute_inho_diffusive_thermal_conductivity(TL[0].copy())
    # print('fff',tmp)
    #if comm.rank == 0:    
    #   fff = TL.copy() 
    #   print(max(fff[0]),max(TL[0]))
      
    
    #Solve fof Fourier-----         
    if n_iter ==  argv.setdefault('max_bte_iter',10)-1:
      #if comm.rank == 0:  
      # for n in range(self.n_mfp):
      #  tmp = self.compute_inho_diffusive_thermal_conductivity(TL[n])  
      #  print(n,tmp)
      
      
      
      #argv.update({'mfe_factor':1.0,\
      #             'verbose':False,\
      #             'lattice_temperature':TL,\
      #             'boundary_temperature':TB.copy(),\
      #             'kappa':np.square(self.mfp),\
      #             'interface_conductance': self.mfp/3.0})    
      #output = self.solve_fourier(**argv) 
      #kappa_fourier = output['kappa_fourier']
      #temperature_fourier = output['temperature_fourier']
      #temperature_fourier_gradient = output['temperature_fourier_gradient']
     #--------------------------------------
      sup_zero = [self.compute_diffusive_thermal_conductivity(TL[n].copy(),mat = np.eye(3)) for n in range(self.n_mfp)]        
      
    comm.Barrier()
    #Output variables---
    SUPp = np.zeros((self.n_mfp,self.n_theta,self.n_phi))
    TLp,TBp  = np.zeros((2,self.n_mfp,self.n_elems))
    FLUXp = np.zeros((self.n_el,3))
    
    rank = comm.rank
    for kk in range(block):
     index = rank*block + kk   
     if index < self.n_index:
      #Read directional information-------------------------------------------------- 
      A = scipy.io.mmread(self.cache + '/A_' + str(index) + '.mtx').tocsc()
      P = np.load(open(self.cache + '/P_' + str(index) +r'.np','rb'))
      HW_MINUS = np.load(open(self.cache + '/HW_MINUS_' + str(index) +r'.np','rb'))
      HW_PLUS = np.load(open(self.cache + '/HW_PLUS_' + str(index) +r'.np','rb'))
      K = scipy.io.mmread(self.cache + '/K_' + str(index) + '.mtx').tocsc()
      #-----------------------------------------------------------------------------

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
          
       fourier=False
       for m in range(self.n_mfp)[::-1]:
       #----------------------------------------------------------------------------------
        if not fourier:
        
         D = np.multiply(TB[m],HW_MINUS)
         F = scipy.sparse.eye(self.n_elems) + theta_factor  * A * self.mfp[m]
         lu = splu(F.tocsc())
         
         
         RHS = self.mfp[m]*theta_factor * (P + D) + TL[m]

         temp = lu.solve(RHS) 
    
         #gg = self.mesh.compute_grad(temp)
        
         
         #temp =  -np.dot(self.dom['S'][t][p],self.mesh.compute_grad(TL[0]).T)/self.dom['d_omega'][t][p]
         #sup = pre_factor * K.dot(temp).sum()*self.kappa_factor
        
         #sup = self.compute_inho_diffusive_thermal_conductivity(temp,mat=self.dom['ss2'][t][p])*3.0/4.0/np.pi
         #sup = 0.0
         #for i,j in zip(*self.mesh.A.nonzero()):
         #  v = self.mesh.get_side_periodic_value(i,j)            
         #  if v == -1:
         #   ll = self.mesh.get_side_between_two_elements(i,j)
         #   area = self.mesh.get_side_area(ll)   
         #   grad = (gg[i] + gg[j])/2.0
         #   sup -= np.dot(self.dom['ss2'][t][p][0],grad)*self.kappa_factor*3.0/4.0/np.pi*area
         
         sup = pre_factor * K.dot(temp-TL[m]).sum()*self.kappa_factor/self.mfp[m]
         #sup = self.compute_diffusive_thermal_conductivity(temp,mat = np.eye(3))     

         
        
             
         if sup == 0: 
            ms[m,t,p] = 1
         else:
          sup_fourier = kappa_fourier[m]*self.dom['ss2'][t][p][self.mesh.direction][self.mesh.direction]*3.0/4.0/np.pi/self.kappa_bulk
         
          if abs((sup_fourier - sup)/sup) < argv.setdefault('ms_error',1e-2): 
             ms[m,t,p] = 1
             if argv.setdefault('multiscale',False):
              fourier = True
              
        else:
         ms[m,t,p] = 1   
         sup = kappa_fourier[m]*self.dom['ss2'][t][p][self.mesh.direction][self.mesh.direction]*3.0/4.0/np.pi/self.kappa_bulk
         temp = temperature_fourier[m]-self.mfp[m]*np.dot(self.dom['S'][t][p],temperature_fourier_gradient[m])/self.dom['d_omega'][t][p]
         
        SUPp[m,t,p] += sup
        
        TLp += np.outer(self.B2[:,m],temp)*self.dom['d_omega'][t][p]*self.symmetry/4.0/np.pi    
        
        #for n in range(self.n_mfp):
        # TLp[n] += temp*self.dom['d_omega'][t][p]*self.symmetry/4.0/np.pi * self.B2[m,0]   
        
        
       #--------------------------
        if self.symmetry == 2.0:
         FLUXp += np.mean(self.B1[:,m])*np.outer(temp,np.multiply(self.dom['S'][t][p],[2.0,2.0,0.0]))
         SUPp[m,self.n_theta -t -1,p] += SUPp[m,t,p]
         
         #TBp += self.dom['at'][t]*self.symmetry*np.outer(self.B1[:,m],np.multiply(temp,HW_PLUS))
         TBp += self.dom['at'][t]*self.symmetry*np.outer(self.B1[:,m],np.multiply(temp,HW_PLUS))
        else:
         FLUXp += np.mean(self.B1[:,m])*np.outer(temp,self.dom['S'][t][p])
         TBp[m] += np.multiply(temp,HW_PLUS)*self.symmetry


   
    comm.Barrier()
    comm.Allreduce([SUPp,MPI.DOUBLE],[SUP,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce([TLp,MPI.DOUBLE],[TL,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce([TBp,MPI.DOUBLE],[TB,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce([FLUXp,MPI.DOUBLE],[FLUX,MPI.DOUBLE],op=MPI.SUM)
    #TL = TLtot.copy()
    #TB = TBtot.copy()
    #TL = TLtot.copy()
    #TB = TLtot.copy()    
    #  print('gg')  
    #  for n in range(self.n_mfp):  
          
    #   print(max(TL[n]))  
    #print('gg')
        
    kappa = sum([self.B0[m]*SUP[m,:,:].sum() for m in range(np.shape(SUP)[0])])*self.kappa_bulk
    sup_m = [SUP[m,:,:].sum() for m in range(np.shape(SUP)[0])]
    n_iter +=1
    error = abs(kappa-kappa_old)/abs(kappa)
    kappa_old = kappa
    if rank==0:
        
     print(kappa)
    #print(rank,np.linalg.norm(TL_old[0]-TL[0]))
    
    #check---
    
    
    
    #if rank == 0:
    # [print(max(TLtot[u])) for u in range(self.n_mfp)]
    #quit() 
    #Solve fourier---------------------------------------------------
    #print(max(TL[0]))
    #if n_iter < argv.setdefault('max_bte_iter',10) and \
    #      error > argv.setdefault('max_bte_error',1e-2) and \
    #      argv.setdefault('multiscale',False):
          
    # argv.update({'mfe_factor':1.0,\
    #              'verbose':False,\
    #              'lattice_temperature':TL,\
    #              'boundary_temperature':TB,\
    #              'kappa':1.0/3.0*np.square(self.mfp),\
    #              'interface_conductance': 1.0/2.0/self.mfp})    
    # output = self.solve_fourier(**argv) 
    # kappa_fourier = output['kappa_fourier']
    # temperature_fourier = output['temperature_fourier']
    # temperature_fourier_gradient = output['temperature_fourier_gradient']
    #----------------------------------------------------------------
    
    #if rank==0:
    #  print(kappa)  
    #  print(sum(sum(sum(ms)))/float(self.n_mfp*self.n_theta*self.n_phi))
      #print(error)
     #print(ms_1) 
     #print('{0:7d} {1:20.4E} {2:25.4E} {4:8.1E} {5:8.1E}'.format(n_iter,kappa,error,ms_1,1.0-ms_1))
  
   #argv.update({'mfe_factor':1.0,\
   #               'verbose':False,\
   #               'lattice_temperature':TL_old,\
   #               'boundary_temperature':TB_old,\
   #               'kappa':np.square(self.mfp)/3.0,\
   #               'interface_conductance': self.mfp/4.0})    
   #output = self.solve_fourier(**argv) 
   #kappa_fourier = output['kappa_fourier']
   #if rank==0:  
    #print(max(TL_old[0])) 
    #print(max(TL_old[-1])) 

    
   # for n in range(self.n_mfp):
   #  tmp = self.compute_inho_diffusive_thermal_conductivity(TL_old[n])
    # print(tmp)
  
    #print(tmp)
   if rank == 0:
     plt.plot(self.mfp,sup_m,'g')  
     plt.plot(self.mfp,sup_zero,'k')  

     #plt.plot(self.mfp,kappa_fourier/self.kappa_bulk,'r')
     plt.xscale('log')
     plt.ylim([0,1])
     plt.show()
     
    
    #sup_m = [SUP[m,:,:].sum() for m in range(np.shape(SUP)[0])]
    #output = {'suppression': SUP,\
    #        'suppression_mfp':sup_m,\
    #        'ThermalFlux_BTE': FLUX,\
    #        'Temperature_BTE': TL[0],\
    #        'kappa_bte':kappa}
  
    #return output
    

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
     data_tmp_b.append(self.mesh.get_side_periodic_value(kc2,kc1)*v_orth*kappa)
     data_tmp_b2.append(self.mesh.get_side_periodic_value(kc2,kc1))
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
    #G *=5.0
    n_index = len(kappa_bulk)
    G = np.zeros(n_index)
    #-------------------------------
    KAPPAp,KAPPA = np.zeros((2,n_index))
    FLUX,FLUXp = np.zeros((2,n_index,3,self.n_elems))
    TLp,TLtot = np.zeros((2,n_index,self.n_elems))

    
    #set up MPI------------------
    comm = MPI.COMM_WORLD  
    size = comm.size
    rank = comm.rank
    
    
    block = n_index // size + 1
    for kk in range(block):
     index = rank*block + kk   
     if index < n_index:
      #Aseemble matrix-------------       
     
      #diag_1 = diags([self.FF],[0]).tocsc()
      #A = (self.F + diag_1*G[index])   + mfe_factor * csc_matrix(scipy.sparse.eye(self.n_elems))/kappa_bulk[index]
      #B = (self.B + np.multiply(self.FF,TB[index])*G[index])  + mfe_factor * TL[index]/kappa_bulk[index] 
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
        KAPPAp[index] = self.compute_inho_diffusive_thermal_conductivity(temp)
        error = abs((KAPPAp[index] - kappa_old)/KAPPAp[index])
        kappa_old = KAPPAp[index]
        n_iter +=1
        
      #FLUXp = np.array([-self.elem_kappa_map[k]*tmp for k,tmp in enumerate(flux)])
      FLUXp[index] = flux.T
      TLp[index] = temp
      
    comm.Allreduce([KAPPAp,MPI.DOUBLE],[KAPPA,MPI.DOUBLE],op=MPI.SUM)
    comm.Allreduce([TLp,MPI.DOUBLE],[TLtot,MPI.DOUBLE],op=MPI.SUM) #to be changed
    comm.Allreduce([FLUXp,MPI.DOUBLE],[FLUX,MPI.DOUBLE],op=MPI.SUM) #
 
    if rank== 0 and argv.setdefault('verbose',True):
     print('  ')
     print('Thermal Conductivity: '.ljust(20) +  '{:8.2f}'.format(KAPPA[0])+ ' W/m/K')
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

    ll = self.mesh.get_side_between_two_elements(i,j)
    kappa_b = self.side_kappa_map[ll]
    (v_orth,dummy) = self.mesh.get_decomposed_directions(i,j,rot=mat)
    
    kappa += 0.5*v_orth *self.PER[i,j]*(temp[j]+self.PER[i,j]-temp[i])*kappa_b

   return kappa*self.kappa_factor


  def compute_diffusive_thermal_conductivity(self,temp,mat=np.eye(3)):

   kappa = 0
   for i,j in zip(*self.PER.nonzero()):
    (v_orth,dummy) = self.mesh.get_decomposed_directions(i,j,rot=mat)
    kappa += 0.5*v_orth *self.PER[i,j]*(temp[j]+self.PER[i,j]-temp[i])

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
