from __future__ import print_function
from __future__ import absolute_import
from scipy.sparse.linalg import spsolve
import os,sys
import numpy as np
from scipy.sparse import csc_matrix
import numpy as np
from mpi4py import MPI
import shutil
from scipy.sparse.linalg import *
from scipy.sparse import diags
from scipy.io import *
import scipy.io
from .utils import *
import deepdish as dd
import time
from termcolor import colored
from .WriteVTK import *
from .geometry import *
from .material import *


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
   self.interface_conductance = argv.setdefault('interface_conductance',1e20)

   #Get material properties---------------------------------
   path = '.'
   files = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and \
         'material' in i]

   self.materials = {}
   for filename in files:
    mat = Material(model='load',filename=filename)

    self.materials.update({mat.state['region']:mat.state})
    self.B0 = mat.state['B0']
    self.B1 = mat.state['B1']
    self.B2 = mat.state['B2']
    self.kappa_bulk = mat.state['kappa_bulk_tot']
   #-----------------------------------------------------


   self.lu = {}

   self.mfp = np.array(mat.state['mfp_sampled'])/1e-9 #In nm
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
     shutil.rmtree(directory)
    os.makedirs(directory)
    #compute matrix--------
   argv.setdefault('max_bte_iter',10)

   if argv['max_bte_iter'] > 0:
    output = compute_sum(self.compute_directional_connections,self.n_index)


   #quit()
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


  def compute_directional_connections(self,index,options):

   #------------------------------BULK----------------------------------
   Diff = []
   Fminus = []
   Fplus = []
   r=[];c=[];d=[]
   rk=[]; ck=[]; dk=[]
   P = np.zeros(self.n_elems)
   for i,j in zip(*self.mesh.A.nonzero()):
    (cm,cp,dummy,dummy) = self.mesh.get_angular_coeff(i,j,index)
    r.append(i); c.append(i); d.append(cp)
    v = self.mesh.get_side_periodic_value(i,j)
    rk.append(i); ck.append(j);
    dk.append(v*(cp+cm)*self.mesh.get_elem_volume(i))
    r.append(i); c.append(j); d.append(cm)
    P[i] += v*cm

   #------------------------------BOUNDARY-------------------------------
   HW_minus = np.zeros(self.n_elems)
   HW_plus = np.zeros(self.n_elems)
   for elem in self.mesh.boundary_elements:
    (dummy,dummy,cmb,cpb) = self.mesh.get_angular_coeff(elem,elem,index)
    HW_plus[elem] =  cpb/np.pi
    HW_minus[elem] = -cmb
   #--------------------------------------------------------------

   #--------------------------WRITE FILES---------------------
   A = csc_matrix( (d,(r,c)), shape=(self.n_elems,self.n_elems) )
   K = csc_matrix( (dk,(rk,ck)), shape=(self.n_elems,self.n_elems) )
   scipy.io.mmwrite(self.cache + '/A_' + str(index) + r'.mtx',A)
   scipy.io.mmwrite(self.cache + '/K_' + str(index) + r'.mtx',K)
   P.dump(open(self.cache + '/P_' + str(index) +r'.np','wb+'))
   HW_minus.dump(open(self.cache +'/HW_MINUS_' + str(index) +r'.np','wb+'))
   HW_plus.dump(open(self.cache + '/HW_PLUS_' + str(index) +r'.np','wb+'))


  def solve_bte(self,index,options):

   #if self.current_iter == 0:
   A = scipy.io.mmread(self.cache + '/A_' + str(index) + '.mtx').tocsc()
   P = np.load(open(self.cache + '/P_' + str(index) +r'.np','rb'))
   HW_MINUS = np.load(open(self.cache + '/HW_MINUS_' + str(index) +r'.np','rb'))
   HW_PLUS = np.load(open(self.cache + '/HW_PLUS_' + str(index) +r'.np','rb'))
   K = scipy.io.mmread(self.cache + '/K_' + str(index) + '.mtx').tocsc()
   #Hot = np.load(open(self.cache + '/Hot_' + str(index) +r'.np','r'))
   #Cold = np.load(open(self.cache + '/Cold_' + str(index) +r'.np','r'))


   TB = options['boundary_temperature']
   #D = np.multiply(TB,HW_MINUS)

   suppression_fourier = options['suppression_fourier']
   TL = options['lattice_temperature']
   #TL = temp_mfp[0]
   #print(max(TL))
   temp_fourier = options['temperature_fourier']
   temp_fourier_gradient = options['temperature_fourier_gradient']
   #TL_new = np.zeros(self.n_el)
   #TB_new = np.zeros(self.n_el)
   TB_new = np.zeros((self.n_mfp,self.n_el))
   #phi_factor = self.dom['phi_dir'][p]/self.dom['d_phi_vec'][p]

   flux = np.zeros((self.n_el,3))
   suppression = np.zeros((self.n_mfp,self.n_theta,self.n_phi))
   temperature_mfp = np.zeros((self.n_mfp,self.n_elems))
   ff_1 = 0
   ff_0 = 0


   #for t in range(int(self.n_theta/2.0)): #We take into account the symmetry

   for tt in range(self.n_theta_irr): #We take into account the symmetry
    if self.dim == 2:
     t = tt
     p = index
     theta_factor = self.dom['at'][t] / self.dom['d_theta_vec'][t]
    else:
     t = int(index/self.n_phi)
     p = index%self.n_phi
     theta_factor = 1.0
    pre_factor = 3.0/4.0/np.pi*0.5*theta_factor * self.dom['d_omega'][t][p]#  self.dom['d_theta_vec'][t]*self.dom['d_phi_vec'][p]


    #ss = self.dom['ss'][t][p]
    fourier = False
    zeroth_order = False
    #RHS_DIFF = scipy.io.mmread('tmp/RHS_DIFF_' + str(t) + '_' + str(p) + '.mtx').tocsc()


    for m in range(self.n_mfp)[::-1]:
     #----------------------------------------------------------------------------------
     if not fourier:

      D = np.multiply(TB[m],HW_MINUS)
      global_index = m*self.dom['n_theta']*self.dom['n_phi'] +t*self.dom['n_phi']+p
      F = scipy.sparse.eye(self.n_elems) + theta_factor * self.mfp[m] * A
      lu = splu(F.tocsc())
      RHS = self.mfp[m]*theta_factor * (P + D) + TL[m]
      temp = lu.solve(RHS)

      sup = pre_factor * K.dot(temp-TL[m]).sum()/self.mfp[m]*self.kappa_factor



      if self.multiscale:
       if not sup == 0.0:
        sup_fourier = suppression_fourier[m][t][p]
        error = abs(sup-sup_fourier)/abs(sup)

        #f t==6 and p==15:
         #print(sup,sup_fourier,error)
        if m == 0:
             print(m)

        if error < self.ms_error:

          fourier = True
          ff_1 +=1
        # sup = sup_fourier
        # sup_old = 0.0
      # else:
    #     fourier = True
#         sup_old = 0.0
     else:
      #if not zeroth_order:
       temp = temp_fourier[m] - self.mfp[m]*np.dot(self.mfp[m]*self.dom['S'][t][p],temp_fourier_gradient[m].T)

       #if not sup == 0.0 :
       sup = suppression_fourier[m][t][p]

       #sup_zeroth = options['suppression_zeroth']*ss[self.mesh.direction][self.mesh.direction]*3.0/4.0/np.pi

       #if abs(sup - sup_zeroth)/abs(sup) < 0.1 and abs(sup-sup_old)/abs(sup) < 0.01 :
       # zeroth_order = True
       # ff_0 +=1
       #else:
       ff_1 +=1
       #sup_old = sup
      #else:
      # sup = sup_zeroth
      # ff_0 +=1
     #--------------------------
     #TL_new += self.B2[m] * temp * self.dom['d_omega'][t][p]/4.0/np.pi * self.symmetry
     if self.symmetry == 2.0:
      flux += np.mean(self.B1[:,m])*np.outer(temp,np.multiply(self.dom['S'][t][p],[2.0,2.0,0.0]))
      #flux += self.B1[m]*np.outer(temp,np.multiply(self.dom['S'][t][p],[2.0,2.0,0.0]))
     else:
      flux += np.mean(self.B1[:,m])*np.outer(temp,self.dom['S'][t][p])


     suppression[m,t,p] += sup
     temperature_mfp[m] += temp*self.dom['d_omega'][t][p]*self.symmetry/4.0/np.pi

     if self.dim == 2:
      suppression[m,self.n_theta -t -1,p] += suppression[m,t,p]
      #TB_new += np.mean(self.B1[0,m])*self.dom['at'][t]*np.multiply(temp,HW_PLUS)*self.symmetry
      #TB_new += self.B1[m]*self.dom['at'][t]*np.multiply(temp,HW_PLUS)*self.symmetry
      TB_new[m] += self.dom['at'][t]*np.multiply(temp,HW_PLUS)*self.symmetry
     else:
      TB_new[m] += np.multiply(temp,HW_PLUS)*self.symmetry

     #------------------------------------------------------------------------------------
   output = {'boundary_temperature':TB_new,'suppression':suppression,'flux':flux,'temperature':temperature_mfp,'ms':np.array([float(ff_0),float(ff_1)])}


   return output



  def compute_function(self,**argv):

   max_iter = argv.setdefault('max_bte_iter',10)
   max_error = argv.setdefault('max_bte_error',1e-2)
   error = 2.0 * max_error

   previous_kappa = 0.0
   self.current_iter = 0

   if MPI.COMM_WORLD.Get_rank() == 0:
     if self.verbose:
      print('  ')
      for region in self.region_kappa_map.keys():
       print(region.ljust(15) +  '{:8.2f}'.format(self.region_kappa_map[region])+ ' W/m/K')




   #Solve standard Fourier---------------------------------------------------------

   #Initalization-----------------------------------------------------------
   suppression = np.zeros((self.n_mfp,self.n_theta,self.n_phi))
   #suppression_fourier = np.zeros((self.n_mfp,self.n_theta,self.n_phi))
   temperature = np.zeros((self.n_mfp,self.n_el))
   temperature_fourier = np.zeros((self.n_mfp,self.n_el))
   temperature_gradient = np.zeros((self.n_mfp,self.n_el,3))
   flux_fourier = np.zeros((self.n_el,3))
   #suppression_first = np.zeros(self.n_mfp)
   ms = np.zeros(2)
   #-----------------------------------------------------------------------
   suppression_fourier = np.zeros((1,self.n_theta,self.n_phi))

   temperature_fourier = np.zeros((1,self.n_el))
   compute_sum(self.solve_fourier,1,output =  {'suppression':suppression_fourier,\
                                                        'temperature':temperature_fourier,\
                                                        'temperature_gradient':np.zeros((1,self.n_el,3)),\
                                                        'flux':flux_fourier},\
                                             options = {'kappa_bulk':[1.0],'mfe_factor':0.0})
   #---------------------------------------------------------------------
   #---------------------------------------------------------------------


   kappa = suppression_fourier[0,:,:].sum()


   #quit()

   kappa_fourier = kappa
   suppression = np.array(self.n_mfp * [suppression_fourier[0]])
   fourier_temp = temperature_fourier[0]
   lattice_temperature = np.tile(temperature_fourier,(self.n_mfp,1))


   suppression_fourier = np.zeros((self.n_mfp,self.n_theta,self.n_phi))
   temperature_fourier = np.zeros((self.n_mfp,self.n_el))

   suppression_fourier = np.zeros((self.n_mfp,self.n_theta,self.n_phi))

   if MPI.COMM_WORLD.Get_rank() == 0:
      if argv['verbose']:
       print('  ')
       print('Thermal Conductivity: '.ljust(20) +  '{:8.2f}'.format(kappa)+ ' W/m/K')
       print('  ')


   #lattice_temperature = fourier_temp.copy()


   boundary_temperature = np.tile(np.multiply(self.boundary_mask,fourier_temp),(self.n_mfp,1))
   b_temperature = boundary_temperature.copy()
   #boundary_temperature = np.multiply(self.boundary_mask,fourier_temp)
   #------------------------------------


   #-----------------------------------
   if MPI.COMM_WORLD.Get_rank() == 0:
    if max_iter > 0:
     print('')
     print(colored('Solving BTE... started', 'green'))
     print('')
     print('    Iter    Thermal Conductivity [W/m/K]      Error       Zeroth - Fourier - BTE')
     print('   ------------------------------------------------------------------------------------')



   flux = np.zeros((self.n_elems,3))
   kappa = 0.0
   suppression_zeroth = np.zeros(self.n_mfp)
   iso_suppression = np.zeros(self.n_mfp)
   while error > max_error and self.current_iter < max_iter:

    #zeroth order---------------------------
    suppression_zeroth = np.array([self.compute_diffusive_thermal_conductivity(lattice_temperature[0],mat = np.eye(3))])

    if self.multiscale:
     compute_sum(self.solve_fourier,self.n_mfp,output =  {'suppression':suppression_fourier,\
                                                            'temperature':temperature_fourier,\
                                                            'temperature_gradient':temperature_gradient,\
                                                            'flux':flux},\
                                                 options = {'boundary_temperature':boundary_temperature.copy(),\
                                                            'lattice_temperature':lattice_temperature.copy(),\
                                                            'interface_conductance': 3.0/2.0/self.mfp,\
                                                            'kappa_bulk':np.square(self.mfp)/3.0,\
                                                            'mfe_factor':1.0})


    if not self.only_fourier:
     #BTE-------------------------------------------------------------------------------
     compute_sum(self.solve_bte,self.n_index, \
                          output = {'temperature':temperature,\
                                    'flux':flux,\
                                    'suppression':suppression,\
                                    'ms':ms,\
                                    'boundary_temperature':b_temperature},\
                          options = {'lattice_temperature':lattice_temperature.copy(),\
                                     'suppression_fourier':suppression_fourier.copy(),\
                                     'boundary_temperature':boundary_temperature.copy(),\
                                     'temperature_fourier':temperature_fourier.copy(),\
                                     'temperature_fourier_gradient':temperature_gradient.copy(),\
                                     'suppression_fourier':suppression_fourier.copy(),\
                                     'suppression_zeroth':suppression_zeroth.copy()})

     #--------------------------------------------------------------------------------
     #lattice_temperature = np.sum([self.B2[m]*temperature[m,:] for m in range(self.n_mfp)],axis=0)
     for n in range(self.n_mfp):
      lattice_temperature[n,:] = np.sum([self.B2[n,m]*temperature[m,:] for m in range(self.n_mfp)],axis=0)
      boundary_temperature[n,:] = np.sum([self.B1[n,m]*b_temperature[m,:] for m in range(self.n_mfp)],axis=0)
      if argv.setdefault('compute_iso',False):
       iso_suppression[n] = self.compute_diffusive_thermal_conductivity(temperature[n])
#Compute iso suppression------

 # suppression_iso = np.zeros(self.n_mfp)
 # for m in range(self.n_mfp):

#-----------------------------
     #lattice_temperature = np.dot(self.B2,temperature[m,:],axis=0)

    #--------------------------
    kappa = sum([self.B0[m]*suppression[m,:,:].sum() for m in range(self.n_mfp)])
    error = abs((kappa-previous_kappa))/abs(kappa)

    if MPI.COMM_WORLD.Get_rank() == 0:
     #print(kappa*self.kappa_bulk)
     ms_0 = ms[0]/float(self.n_mfp)/float(self.n_theta)/float(self.n_phi)*self.symmetry
     ms_1 = ms[1]/float(self.n_mfp)/float(self.n_theta)/float(self.n_phi)*self.symmetry
     #if max_iter > 0:
     print('{0:7d} {1:20.4E} {2:25.4E} {3:10.2E} {4:8.1E} {5:8.1E}'.format(self.current_iter,kappa*self.kappa_bulk, error,ms_0,ms_1,1.0-ms_0-ms_1))

    previous_kappa = kappa
    self.current_iter +=1
    #------------------------


   if not self.multiscale and max_iter > 0:
    compute_sum(self.solve_fourier,self.n_mfp,output =  {'suppression':suppression_fourier,\
                                                        'temperature':temperature_fourier,\
                                                        'temperature_gradient':temperature_gradient,\
                                                        'flux':flux},
                                                 options = {'boundary_temperature':boundary_temperature.copy(),\
                                                            'lattice_temperature':lattice_temperature.copy(),\
                                                            'interface_conductance': 1.0/2.0/self.mfp,\
                                                            #'interface_conductance': np.zeros(self.n_mfp),\
                                                            'kappa_bulk':np.square(self.mfp)/3.0,\
                                                            'mfe_factor':1.0})





   self.state = {'kappa_bte':kappa*self.kappa_bulk,\
            'suppression':suppression,\
            'zero_suppression':suppression_zeroth,\
            'fourier_suppression':suppression_fourier,\
            'iso_suppression':iso_suppression,\
            'dom':self.dom,\
            'kappa_fourier':kappa_fourier,\
            'mfp':self.mfp*1e-9,\
            'bte_temperature':lattice_temperature[0],\
            'fourier_temperature':fourier_temp,\
            'fourier_flux':flux_fourier,\
            'bte_flux':flux}

   if MPI.COMM_WORLD.Get_rank() == 0:
    if max_iter > 0 :
     print('   ------------------------------------------------------------------------------------')

   if max_iter > 0:
    if MPI.COMM_WORLD.Get_rank() == 0:
     print('')
     print(colored('Solving BTE... done', 'green'))
     print('')


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

     kappa = self.side_kappa_map[ll]
     #kappa = 100.0
     #print(kappa)

     (v_orth,dummy) = self.mesh.get_decomposed_directions(kc1,kc2)
     vol1 = self.mesh.get_elem_volume(kc1)
     vol2 = self.mesh.get_elem_volume(kc2)
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


  def solve_fourier(self,n,options):

    #Update matrices-------
    kappa_bulk = options['kappa_bulk'][n]
    G = options.setdefault('interface_conductance',np.zeros(self.n_mfp))[n]
    TL = options.setdefault('lattice_temperature',np.zeros((self.n_mfp,self.n_el)))
    TB = options.setdefault('boundary_temperature',np.zeros((self.n_mfp,self.n_el)))
    mfe_factor = options.setdefault('mfe_factor',0.0)

    #G = 0.0
    #mfe_factor = 0.0
    #print(G)
    #G = 0.0
    #G *= self.dom['correction']/2.0
    diag_1 = diags([self.FF],[0]).tocsc()
    #A = (self.F + diag_1*G)   + mfe_factor * csc_matrix(scipy.sparse.eye(self.n_elems))/kappa_bulk
    #B = (self.B + np.multiply(self.FF,TB[n])*G)  + mfe_factor * TL[n]/kappa_bulk
    A = self.F
    B = self.B
    if mfe_factor == 0.0:
     A[10,10] = 1.0
    #----------------------------------------------------

    #------------------------
    SU = splu(A)
    C = np.zeros(self.n_el)
    #--------------------------------------
    min_err = 1e-3
    error = 2*min_err
    max_iter = 10
    n_iter = 0
    kappa_old = 0


    n_kappa = len(options['kappa_bulk'])
    temperature_mfp = np.zeros((n_kappa,self.n_elems))
    gradient_temperature_mfp = np.zeros((len(options['kappa_bulk']),self.n_elems,3))
    #max_iter=1
    min_err = 1e-8
    while error > min_err and n_iter < max_iter :


     RHS = B + C#*kappa_bulk
     if mfe_factor == 0.0:
       RHS[10] = 0.0

     temp = SU.solve(RHS)
     temp = temp - (max(temp)+min(temp))/2.0

     (C,flux) = self.compute_non_orth_contribution(temp)

     #kappa = self.compute_diffusive_thermal_conductivity(temp)

     kappa = self.compute_inho_diffusive_thermal_conductivity(temp)

     error = abs((kappa - kappa_old)/kappa)


     kappa_old = kappa
     n_iter +=1
    suppression = np.zeros(len(options['kappa_bulk']))
    temperature_mfp[n] = temp

    gradient_temperature_mfp[n] = self.mesh.compute_grad(temp)
    gradient_temperature = self.mesh.compute_grad(temp)
    suppression = np.zeros((len(options['kappa_bulk']),self.n_theta,self.n_phi))
    for t in range(self.n_theta):
     for p in range(self.n_phi):
      suppression[n,t,p] = kappa*self.dom['ss2'][t][p][self.mesh.direction][self.mesh.direction]*3.0/4.0/np.pi


    fourier_flux = [-self.elem_kappa_map[k]*tmp for k,tmp in enumerate(flux)]
    fourier_flux = np.array(fourier_flux)

    output = {'suppression':suppression,'temperature':temperature_mfp,'flux':fourier_flux,'temperature_gradient':gradient_temperature_mfp}

    return output

  def compute_non_orth_contribution(self,temp) :

    self.gradT = self.mesh.compute_grad(temp)
    C = np.zeros(self.n_el)
    for i,j in zip(*self.mesh.A.nonzero()):

     if not i==j:
      ll = self.mesh.get_side_between_two_elements(i,j)
      kappa = self.side_kappa_map[ll]
      #Get agerage gradient----
      side = self.mesh.get_side_between_two_elements(i,j)


      w = self.mesh.get_interpolation_weigths(side,i)
      grad_ave = w*self.gradT[i] + (1.0-w)*self.gradT[j]
      #------------------------
      (dumm,v_non_orth) = self.mesh.get_decomposed_directions(i,j)

      C[i] += np.dot(grad_ave,v_non_orth)/2.0/self.mesh.get_elem_volume(i)*kappa
      C[j] -= np.dot(grad_ave,v_non_orth)/2.0/self.mesh.get_elem_volume(j)*kappa


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
