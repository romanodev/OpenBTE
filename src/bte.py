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
#from fourier import Fourier

class BTE(object):
  
  def __init__(self,argv):

   self.mesh = argv['geometry']
   #self.x = argv.setdefault('density',np.ones(len(self.mesh.elems)))


   self.multiscale = argv.setdefault('multiscale',False)
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
   #output = compute_sum(self.compute_directional_diffusion,self.n_phi*self.n_theta,{},{})

   self.assemble_fourier()  
 
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


  def compute_directional_diffusion(self,index,options):

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
     data_tmp.append(self.mesh.get_side_periodic_value(ll,kc2,self.side_periodic_value)*v_orth)
    RHS = csc_matrix( (np.array(data_tmp),(np.array(row_tmp),np.array(col_tmp))), shape=(self.n_elems,self.n_elems) )

    scipy.io.mmwrite('tmp/RHS_DIFF_' + str(t) + '_' + str(p) + r'.mtx',RHS) 


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
   scipy.io.mmwrite('tmp/A_' + str(p) + r'.mtx',A) 
   scipy.io.mmwrite('tmp/K_' + str(p) + r'.mtx',K) 
   P.dump(file('tmp/P_' + str(p) +r'.np','wb+'))
   HW_minus.dump(file('tmp/HW_MINUS_' + str(p) +r'.np','w+'))
   HW_plus.dump(file('tmp/HW_PLUS_' + str(p) +r'.np','w+'))

  def solve_bte(self,p,options):

   #if self.current_iter == 0:
   A = scipy.io.mmread('tmp/A_' + str(p) + '.mtx').tocsc()
   P = np.load(file('tmp/P_' + str(p) +r'.np','rb'))
   HW_MINUS = np.load(file('tmp/HW_MINUS_' + str(p) +r'.np','r'))
   HW_PLUS = np.load(file('tmp/HW_PLUS_' + str(p) +r'.np','r'))
   K = scipy.io.mmread('tmp/K_' + str(p) + '.mtx').tocsc()

   TB = options['boundary_temp']
   suppression_first = options['suppression_fourier']
   TL = options['temperature']
   temp_fourier = options['temperature_fourier']
   temp_fourier_gradient = options['temperature_fourier_gradient']
   TL_new = np.zeros(self.n_el) 
   TB_new = np.zeros(self.n_el) 
   phi_factor = self.dom['phi_dir'][p]/self.dom['d_phi_vec'][p]
   D = np.multiply(TB,HW_MINUS)

   flux = np.zeros((self.n_el,3)) 
   suppression = np.zeros((self.n_mfp,self.n_theta,self.n_phi))
   temperature_mfp = np.zeros((self.n_mfp,self.n_elems))
   ff = 0
   ff_0 = 0
   
   symmetry = 2
   kappa_factor = 3.0/4.0/np.pi*self.mesh.size[0]/self.area_flux
   for t in range(int(self.n_theta/2.0)): #We take into account the symmetry
    theta_factor = self.dom['at'][t] / self.dom['d_theta_vec'][t]
    ss = self.dom['ss'][t][p]
    fourier = False
    zeroth_order = False
    #RHS_DIFF = scipy.io.mmread('tmp/RHS_DIFF_' + str(t) + '_' + str(p) + '.mtx').tocsc()
    for m in range(self.n_mfp)[::-1]:
     pre_factor = 0.5*theta_factor*self.dom['d_theta_vec'][t]*self.dom['d_phi_vec'][p]
     #----------------------------------------------------------------------------------
     if not fourier:
      index = m*self.dom['n_theta']*self.dom['n_phi'] +t*self.dom['n_phi']+p
      if not index in self.lu.keys():
       F = scipy.sparse.eye(self.n_elems) + theta_factor * self.mfp[m] * A
       lu = splu(F.tocsc())
       self.lu.update({index:lu})
      else:
       lu = self.lu[index] #read previous lu
      RHS = self.mfp[m]*theta_factor * (P + D) + TL
      temp = lu.solve(RHS)
   
      sup = pre_factor * K.dot(temp-TL).sum()/self.mfp[m]
       #sup_app = pre_factor * K.dot(temp_app-TL).sum()/self.mfp[m]
       #a1 = abs(sup-sup_fourier)/abs(sup)
       #contr = self.compute_sup_approximation((TL-temp)/self.mfp[m],self.dom['S'][t][p])
       #sup_2 = sup_fourier - contr
       #a2 = abs(sup-sup_2)/abs(sup)
       #sup_3 = pre_factor * K.dot(-TL).sum()/self.mfp[m]
       
       #sup_1 = pre_factor * K.dot(temp_fourier[m]-TL).sum()/self.mfp[m]
       #sup_2 = suppression_first[m]*ss[0][0]
       #print(sup/sup_2,sup/(sup_2-sup_1))
      #+ self.dom['S'][t][p][0]*(TL-)

      if self.multiscale:
       if not sup == 0.0:
        sup_fourier = suppression_first[m]*ss[0][0] #- pre_factor * K.dot(temp_fourier[m]-TL).sum()/self.mfp[m]
       error = abs(sup-sup_fourier)/abs(sup)
       if error < 0.05 and self.multiscale:
        fourier = True
        ff +=1
        sup = sup_fourier
        sup_old = 0.0
        
     else:
      if not zeroth_order:
       temp = temp_fourier[m] - np.dot(self.mfp[m]*self.dom['S'][t][p],temp_fourier_gradient[m].T)
       sup = suppression_first[m]*ss[0][0] #- pre_factor * K.dot(temp_fourier[m]-TL).sum()/self.mfp[m]
     
       sup_zeroth = options['suppression_zeroth']*ss[0][0]
       if abs(sup - sup_zeroth)/abs(sup) < 0.05 and abs(sup-sup_old)/abs(sup) < 0.01 :
        zeroth_order = True
        ff_0 +=1
       else:
        ff +=1
        sup_old = sup
      else:
       sup = sup_zeroth
       ff_0 +=1
     #--------------------------
     TL_new += self.B2[m] * temp * self.dom['d_omega'][t][p]/4.0/np.pi * symmetry
     flux += self.B1[m]*np.outer(temp,self.dom['S'][t][p])
     suppression[m,t,p] += sup
     temperature_mfp[m] += temp*self.dom['d_omega'][t][p]*symmetry/4.0/np.pi

     if symmetry == 2.0:
      suppression[m,self.n_theta -t -1,p] += suppression[m,t,p]
     
     TB_new += self.B1[m]*self.dom['at'][t]*np.multiply(temp,HW_PLUS)*symmetry
     #------------------------------------------------------------------------------------
   suppression *= kappa_factor

   output = {'temperature':TL_new,'boundary_temp':TB_new,'suppression':suppression,'flux':flux,'temperature_mfp':temperature_mfp,'ms':np.array([ff]),'ms_0':np.array([ff_0])}

   return output



  def compute_function(self,**argv):
 
   #fourier_temp = argv['fourier_temperature']
   max_iter = argv.setdefault('max_bte_iter',10)
   max_error = argv.setdefault('max_bte_error',1e-2)
   error = 2.0 * max_error
  
   previous_kappa = 0.0
   self.current_iter = 0
   symmetry = 2 

   #First Step Fourier----------------- 
   output_fourier = {'suppression':np.array([0.0]),\
                     'temperature':np.zeros((1,self.n_el)),\
                     'temperature_gradient':np.zeros((1,self.n_el,3)),\
                     'flux':np.zeros((self.n_el,3))}

   output_fourier = compute_sum(self.solve_fourier,1,output_fourier,{'verbose':1,'kappa_bulk':[1.0],'mfe':False})

   suppression_fourier = self.n_mfp * [output_fourier['suppression'][0]]
   temperature_fourier= self.n_mfp * [output_fourier['temperature'][0]]
   temperature_fourier_gradient = self.n_mfp * [output_fourier['temperature_gradient'][0]]


   fourier_temp = output_fourier['temperature'][0]
   lattice_temperature = fourier_temp
   boundary_temperature = fourier_temp
   #previous_boundary_temp = fourier_temp
   #output = {'boundary_temp':fourier_temp,'temperature':fourier_temp}
   #------------------------------------ 

   #-----------------------------------
   if MPI.COMM_WORLD.Get_rank() == 0:
    print('    Iter    Thermal Conductivity [W/m/K]      Error       Zeroth - Fourier - BTE')
    print('   --------------------------------------------------------------------------------------------')

   while error > max_error and self.current_iter < max_iter:

    #zeroth order---------------------------

    if self.multiscale:
     suppression_zeroth = np.array([self.compute_generic_diffusive_thermal_conductivity(lattice_temperature,np.eye(3))])

    #first order------------------------------------------------------------------------
     output_fourier.update({'suppression':np.zeros(self.n_mfp),\
                           'temperature':np.zeros((self.n_mfp,self.n_elems)),\
                           'temperature_gradient':np.zeros((self.n_mfp,self.n_elems,3))})
 
     options_fourier = {'kappa_bulk':np.square(self.mfp)/3.0,\
                       'verbose':0,\
                       'temperature':lattice_temperature,\
                       'boundary_temperature':boundary_temperature,\
                       'mfe':True}

     output_fourier = compute_sum(self.solve_fourier,self.n_mfp,output_fourier,options_fourier)
     suppression_first = output_fourier['suppression']
     temperature_fourier = output_fourier['temperature']
     temperature_fourier_gradient = output_fourier['temperature_gradient']
    else:
     temperature_fourier_gradient = np.zeros((self.n_mfp,self.n_elems,3))
     temperature_fourier = np.zeros((self.n_mfp,self.n_elems))
     suppression_zeroth = np.zeros(1)
     suppression_first = np.zeros(self.n_mfp)
     
    #----------------------------------------------------------------------------------
    
    #BTE-------------------------------------------------------------------------------
    options_bte = {'temperature':np.array(lattice_temperature),\
                   'boundary_temp':np.array(boundary_temperature),\
                   'suppression_fourier':suppression_first,\
                   'suppression_zeroth':suppression_zeroth,\
                   'temperature_fourier':temperature_fourier,\
                   'temperature_fourier_gradient':temperature_fourier_gradient}

    output = {'temperature':np.zeros(self.n_el),\
              'boundary_temp':np.zeros(self.n_el),\
              'suppression':np.zeros((self.n_mfp,self.n_theta,self.n_phi)),\
              'flux':np.zeros((self.n_el,3)),\
              'temperature_mfp':np.zeros((self.n_mfp,self.n_elems)),\
              'ms':np.zeros(1,dtype=int),\
              'ms_0':np.zeros(1,dtype=int)}

    output = compute_sum(self.solve_bte,self.n_phi,output,options_bte)
    #--------------------------------------------------------------------------------

    #--------------------------
    directional_suppression = output['suppression']
    kappa = sum([self.B0[m]*directional_suppression[m,:,:].sum() for m in range(self.n_mfp)])
    error = abs((kappa-previous_kappa))/kappa
    if MPI.COMM_WORLD.Get_rank() == 0:
     ms = output['ms'][0]/float(self.n_mfp)/float(self.n_theta)/float(self.n_phi)*symmetry
     ms_0 = output['ms_0'][0]/float(self.n_mfp)/float(self.n_theta)/float(self.n_phi)*symmetry
     print('{0:7d} {1:20.4E} {2:25.4E} {3:10.2E} {4:8.1E} {5:8.1E}'.format(self.current_iter,kappa*self.kappa_bulk, error,ms_0,ms,1.0-ms-ms_0))
    previous_kappa = kappa
    self.current_iter +=1
    #------------------------

    boundary_temperature = output['boundary_temp']
    lattice_temperature = output['temperature']
    #-----------------------
   

   suppression = [directional_suppression[m,:,:].sum() for m in range(self.n_mfp)]
   flux = output['flux']

   #Compute zero suppression---------
   #zero_suppression = self.n_mfp * [self.compute_diffusive_thermal_conductivity(previous_temp)]
   #------------------------------
   #Iso suppression---------
   iso_suppression = [self.compute_diffusive_thermal_conductivity(output['temperature_mfp'][m]) for m in range(self.n_mfp)]


   self.state = {'kappa_bte':kappa*self.kappa_bulk,\
            'directional_suppression':directional_suppression,\
            'suppression_function':suppression,\
            'zero_suppression':suppression_zeroth,\
            'iso_suppression':iso_suppression,\
            'fourier_suppression':suppression_first,\
            'dom':self.dom,\
            'mfp':self.mfp*1e-9,\
            'bte_temperature':lattice_temperature,\
            'bte_flux':flux}

   if MPI.COMM_WORLD.Get_rank() == 0:
    print('   --------------------------------------------------------------------------------------------')



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
     ll = self.mesh.get_side_between_two_elements(kc1,kc2)  
     (v_orth,dummy) = self.mesh.get_decomposed_directions(kc1,kc2)
     vol1 = self.mesh.get_elem_volume(kc1)
     vol2 = self.mesh.get_elem_volume(kc2)
     row_tmp.append(kc1)
     col_tmp.append(kc2)
     data_tmp.append(-v_orth/vol1)
     row_tmp.append(kc1)
     col_tmp.append(kc1)
     data_tmp.append(v_orth/vol1)
     #----------------------
     row_tmp_b.append(kc1)
     col_tmp_b.append(kc2)
     data_tmp_b.append(self.mesh.get_side_periodic_value(ll,kc2,self.side_periodic_value)*v_orth)
     data_tmp_b2.append(self.mesh.get_side_periodic_value(ll,kc2,self.side_periodic_value))
     #---------------------
     B[kc1] += self.mesh.get_side_periodic_value(ll,kc2,self.side_periodic_value)*v_orth/vol1

    F = csc_matrix((np.array(data_tmp),(np.array(row_tmp),np.array(col_tmp))), shape=(self.n_elems,self.n_elems) )
    RHS = csc_matrix( (np.array(data_tmp_b),(np.array(row_tmp_b),np.array(col_tmp_b))), shape=(self.n_elems,self.n_elems) )
    PER = csc_matrix( (np.array(data_tmp_b2),(np.array(row_tmp_b),np.array(col_tmp_b))), shape=(self.n_elems,self.n_elems) )
    #Boundary_elements
    FF = np.zeros(self.n_elems)
    for side in self.mesh.side_list['Boundary']:
     elem = self.mesh.side_elem_map[side][0]
     vol = self.mesh.get_elem_volume(elem)
     area = self.mesh.get_side_area(side)
     FF[elem] = area/vol
    #-------    
    data = {'F':F,'RHS':RHS,'B':B,'FF':FF,'PER':PER}
   else: data = None
   data =  MPI.COMM_WORLD.bcast(data,root=0)
   self.F = data['F']
   self.RHS = data['RHS']
   self.PER = data['PER']
   self.B = data['B']
   self.FF = data['FF']


  def solve_fourier(self,n,options):

    #Update matrices-------
    kappa_bulk = options['kappa_bulk'][n]
    if options.setdefault('mfe',False):
     TB = options['boundary_temperature']     

     R = 3.0/2.0/self.mfp[n]
     #R= 0.0
     BR = diags([self.FF],[0]).tocsc()*R
     A = (self.F + BR)* kappa_bulk + csc_matrix(scipy.sparse.eye(self.n_elems)) 
     B = (self.B + np.multiply(self.FF,TB)*R) * kappa_bulk + options['temperature']
    else:
     A = self.F * kappa_bulk 
     B = self.B * kappa_bulk 
     A[10,10] = 1.0

    #------------------------
    SU = splu(A)
    C = np.zeros(self.n_el)
    #--------------------------------------
    min_err = 1e-3
    error = 2*min_err
    max_iter = 10
    n_iter = 0
    kappa_old = 0

    if options['verbose'] > 0:    
     print('  ')
     print('    Iter    Thermal Conductivity [W/m/K]      Error')
     print('   ---------------------------------------------------')
    
    temperature_mfp = np.zeros((len(options['kappa_bulk']),self.n_elems)) 
    gradient_temperature_mfp = np.zeros((len(options['kappa_bulk']),self.n_elems,3)) 
    while error > min_err and n_iter < max_iter :

     RHS = B + C*kappa_bulk
     if not 'temperature' in options.keys():
      RHS[10] = 0.0
     temp = SU.solve(RHS)
     temp = temp - (max(temp)+min(temp))/2.0
     (C,flux) = self.compute_non_orth_contribution(temp)

     kappa = self.compute_diffusive_thermal_conductivity(temp)
     error = abs((kappa - kappa_old)/kappa)
     kappa_old = kappa
     if options['verbose'] > 0:    
      print('{0:7d} {1:20.4E} {2:25.4E}'.format(n_iter,kappa*self.kappa_bulk, error))
     n_iter +=1 
    if options['verbose'] > 0:    
     print('   ---------------------------------------------------')
     print('  ')
    suppression = np.zeros(len(options['kappa_bulk']))
    suppression[n] = kappa
    temperature_mfp[n] = temp

    gradient_temperature_mfp[n] = self.mesh.compute_grad(temp,self.side_periodic_value)

    output = {'suppression':suppression,'temperature':temperature_mfp,'flux':flux,'temperature_gradient':gradient_temperature_mfp}

    return output

  def compute_non_orth_contribution(self,temp) :

    self.gradT = self.mesh.compute_grad(temp,self.side_periodic_value)
    C = np.zeros(self.n_el)
    for i,j in zip(*self.mesh.A.nonzero()):
     #Get agerage gradient----
     side = self.mesh.get_side_between_two_elements(i,j)
     w = self.mesh.get_interpolation_weigths(side)
     grad_ave = w*self.gradT[i] + (1.0-w)*self.gradT[j]
     #------------------------
     (dumm,v_non_orth) = self.mesh.get_decomposed_directions(i,j)
     
     C[i] += np.dot(grad_ave,v_non_orth)/2.0/self.mesh.get_elem_volume(i)
     C[j] -= np.dot(grad_ave,v_non_orth)/2.0/self.mesh.get_elem_volume(j)

    return C,-self.gradT


  def compute_diffusive_thermal_conductivity(self,temp):

   kappa = 0
   for i,j in zip(*self.RHS.nonzero()):
    kappa += 0.5*self.RHS[i,j]*(temp[j]+self.RHS[i,j]/abs(self.RHS[i,j])-temp[i])*self.mesh.size[0]/self.area_flux

   return kappa


  def compute_generic_diffusive_thermal_conductivity(self,temp,mat):

   kappa = 0
   for i,j in zip(*self.PER.nonzero()):
    (v_orth,dummy) = self.mesh.get_decomposed_directions(i,j,rot=mat)
    kappa += 0.5*v_orth*self.PER[i,j]*(temp[j]+self.PER[i,j]-temp[i])*self.mesh.size[0]/self.area_flux

   return kappa


 # def compute_sup_approximation(self,temp,S) :

 #  contr = 0.0
 #  area_tot = 0.0
 #  for s in self.flux_sides:
 #   elems = self.mesh.side_elem_map[s] 
 #   normal = self.mesh.get_side_normal(0,s)  
 #   area = self.mesh.get_side_area(s)
 #   i = elems[0]
 #   j = elems[1]
 #   temp_ave = (temp[j]-self.PER[i,j]+temp[i])/2.0
 #   contr += temp_ave * np.dot(normal,S)*area
 #   area_tot += area

 #  return contr/area_tot




