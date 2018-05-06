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
from termcolor import colored
from geometry import *
from material import *
from matplotlib.pylab import *

#from fourier import Fourier

class Solver(object):
  
  def __init__(self,**argv):

   data = dd.io.load(argv['filename'] + '.hdf5')
   self.FULL = data['A']
   self.MFP = data['mfp']*1e9
   self.COEFF = data['COEFF']
   self.TEMP = data['h']

   #self.FULL = argv['A'] 
   #self.MFP = argv['MFP']*1e9 #MFP in nanometers
   #self.COEFF = argv['COEFF'] 
   #self.TEMP = argv['TEMP_COEFF'] 

   self.n_mfp = np.shape(self.MFP)[0]


   self.mfp_abs = np.zeros(self.n_mfp)
   for n in range(self.n_mfp):
    self.mfp_abs[n] = np.linalg.norm(self.MFP[n])

   #if MPI.COMM_WORLD.Get_rank() == 0:
   # plot(self.mfp_abs)
   # show()
   #quit()

   self.temp = {}
   self.mesh = Geometry(model='load',filename = argv.setdefault('geometry_filename','geometry'))

   #self.mesh = argv['geometry']
   self.n_elems = len(self.mesh.elems)
   self.dim = self.mesh.dim
   self.mfe = False
   #self.x = argv.setdefault('density',np.ones(len(self.mesh.elems)))

   self.multiscale = argv.setdefault('multiscale',False)
   self.only_fourier = argv.setdefault('only_fourier',False)
   #Get material properties-----
  # mat = argv['material']

   self.lu = {}


   self.n_el = len(self.mesh.elems)
   #data = self.mesh.compute_boundary_condition_data()

   #self.side_periodic_value = data['side_periodic_value']
   #self.area_flux = data['area_flux']
   #self.kappa_factor = self.mesh.size[0]/self.area_flux
   self.kappa_factor = self.mesh.kappa_factor

   self.print_logo()  
   #quit() 


   #if self.compute_matrix:
   if MPI.COMM_WORLD.Get_rank() == 0:
    directory = 'tmp'
    if os.path.exists(directory):
     shutil.rmtree(directory)
    os.makedirs(directory)
   #compute matrix--------
   out_flux = np.zeros(self.n_elems)
   in_flux = np.zeros(self.n_elems)
   output = compute_sum(self.compute_directional_connections,self.n_mfp,output={'out_flux':out_flux,'in_flux':in_flux})

   self.out_flux = out_flux
   #if MPI.COMM_WORLD.Get_rank() == 0:
    #for side in self.mesh.side_list['Boundary']:
    # elem = self.mesh.side_elem_map[side][0]
    # print(out_flux[elem]/in_flux[elem])

   #quit()
   
   self.assemble_fourier()  
 
   #solve the BTE 
   self.compute_function(**argv)

   #if MPI.COMM_WORLD.Get_rank() == 0:
   # print('')
   # print('Solving BTE....done.')
   # print('')
    #print('Effective Thermal Conductivity (BTE):   ' + str(round(self.state['kappa_bte'],4)) + r' W/m/K')
    #print('')

   if MPI.COMM_WORLD.Get_rank() == 0:
    if os.path.isdir('tmp'):
     shutil.rmtree('tmp')


   #----------------

   #SAVE FILE--------------------
   #if argv.setdefault('save',True):
   # if MPI.COMM_WORLD.Get_rank() == 0:
   #  dd.io.save('solver.hdf5', self.state)


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
     #data_tmp.append(self.mesh.get_side_periodic_value(ll,kc2,self.side_periodic_value)*v_orth)
     data_tmp.append(self.mesh.get_side_periodic_value(ll,kc2)*v_orth)
    RHS = csc_matrix( (np.array(data_tmp),(np.array(row_tmp),np.array(col_tmp))), shape=(self.n_elems,self.n_elems) )

    scipy.io.mmwrite('tmp/RHS_DIFF_' + str(t) + '_' + str(p) + r'.mtx',RHS) 


  def compute_directional_connections(self,index,options):


   Diff = []
   Fminus = []
   Fplus = []
   r=[];c=[];d=[]
   rk=[]; ck=[]; dk=[]
   P = np.zeros(self.n_elems)

   for i,j in zip(*self.mesh.A.nonzero()):
   
    side = self.mesh.get_side_between_two_elements(i,j)  
    coeff = np.dot(self.MFP[index],self.mesh.get_coeff(i,j))

    if coeff > 0:
     r.append(i); c.append(i); d.append(coeff)
     v = self.mesh.get_side_periodic_value(side,i)
     rk.append(i); ck.append(j);      
     dk.append(v*coeff*self.mesh.get_elem_volume(i)*0.5)

    if coeff < 0 :
     r.append(i); c.append(j); d.append(coeff)
     v = self.mesh.get_side_periodic_value(side,j)
     P[i] -= v*coeff
     rk.append(i); ck.append(j);      
     dk.append(-v*coeff*self.mesh.get_elem_volume(i)*0.5)

   #Write the boundaries------
   HW_minus = np.zeros(self.n_elems)
   HW_minus_test = np.zeros(self.n_elems)
   HW_plus = np.zeros(self.n_elems)
   for side in self.mesh.side_list['Boundary']:
    (coeff,elem) = self.mesh.get_side_coeff(side)
    tmp = np.dot(coeff,self.MFP[index])
    if tmp < 0:
     HW_minus[elem] = -tmp
     HW_minus_test[elem] = np.dot(self.MFP[index],self.mesh.get_side_normal(0,side))*self.COEFF[index]
    else:
     r.append(elem); c.append(elem); d.append(tmp)
     HW_plus[elem] = np.dot(self.MFP[index],self.mesh.get_side_normal(0,side))*self.COEFF[index]
   #-----------------------------------------------

   #--------------------------WRITE FILES---------------------
   A = csc_matrix( (d,(r,c)), shape=(self.n_elems,self.n_elems) )
   K = csc_matrix( (dk,(rk,ck)), shape=(self.n_elems,self.n_elems) )
   scipy.io.mmwrite('tmp/A_' + str(index) + r'.mtx',A) 
   scipy.io.mmwrite('tmp/K_' + str(index) + r'.mtx',K) 
   P.dump(file('tmp/P_' + str(index) +r'.np','wb+'))
   HW_minus.dump(file('tmp/HW_MINUS_' + str(index) +r'.np','w+'))
   #HW_plus.dump(file('tmp/HW_PLUS_' + str(index) +r'.np','w+'))
  
   output = {'out_flux':HW_plus,'in_flux':HW_minus_test}
 
   return output

  def solve_bte(self,index,options):

   #----------------
   A = scipy.io.mmread('tmp/A_' + str(index) + '.mtx').tocsc()
   P = np.load(file('tmp/P_' + str(index) +r'.np','rb'))
   K = scipy.io.mmread('tmp/K_' + str(index) + '.mtx').tocsc()
   HW_MINUS = np.load(file('tmp/HW_MINUS_' + str(index) +r'.np','r'))
   #HW_PLUS = np.load(file('tmp/HW_PLUS_' + str(index) +r'.np','r'))
   #----------------

   temperature = options['temperature']
   boundary_temperature = options['boundary_temperature']

   TL_new = np.zeros((self.n_mfp,self.n_el))
   F = scipy.sparse.eye(self.n_elems) + A 
   lu = splu(F.tocsc())
   #print(np.multiply(boundary_temperature,HW_MINUS))
   RHS = P + temperature[index] + np.multiply(boundary_temperature,HW_MINUS)
   temp_new = lu.solve(RHS)

   #Store old temp-------
   if self.current_iter == 0:
    #temp_old = temperature[index]  
    temp_old = temp_new.copy()
   else:
    temp_old = self.temp[index]

   #-----------------------------

   alpha = 0.75

   temp = alpha*temp_new + (1.0 - alpha)*temp_old

   #--------------------------------------------------------
   boundary_temperature_new = np.zeros(self.n_elems)   
   for side in self.mesh.side_list['Boundary']:
    tmp = np.dot(self.MFP[index],self.mesh.get_side_normal(0,side))
    elem = self.mesh.side_elem_map[side][0]
    if tmp > 0.0:
     boundary_temperature_new[elem] += temp[elem]*tmp*self.COEFF[index]/self.out_flux[elem]
   #--------------------------------------------------------

   self.temp.update({index:temp})

   kappa = self.COEFF[index]*K.dot(temp).sum()*self.kappa_factor*1e-18
   #quit()
   #kappa = self.COEFF[index]*self.MFP[index][0]*self.MFP[index][0]

   #print(np.sum(self.FULL,axis=1))
   #quit()
   #TL_new = 0.75*np.outer(self.FULL[:,index],temp)+0.25*temperature
   TL_new = np.outer(self.FULL[:,index],temp)

   global_temperature = temp*self.TEMP[index]
   thermal_flux = np.outer(temp,self.COEFF[index])*self.MFP[index]*1e-18

   output = {'kappa':np.array([kappa]),'temperature':TL_new,'boundary_temperature':boundary_temperature_new,'global_temperature':global_temperature,'thermal_flux':thermal_flux}

   return output

  
  def  compute_diff(self,x,TL,mm):

   m = mm[0]
   t = mm[1]
   p = mm[2]
 
   sup = 0.0   
   for side in self.mesh.side_list['Cold']:
    (coeff,elem) = self.mesh.get_side_coeff(side)
    tmp = np.dot(coeff,self.dom['S'][t][p])*self.mesh.get_elem_volume(elem)
    
    sup += tmp*(x[elem]-0.5)/2.0
    #print(-0.5-x[elem])
    #if tmp < 0:
    # sup += tmp*0.5

    #if tmp > 0:
    # sup += tmp*(x[elem])
    
   return sup/self.mfp[m]



  def compute_function(self,**argv):
 
   max_iter = argv.setdefault('max_bte_iter',10)
   max_error = argv.setdefault('max_bte_error',1e-3)
   error = 2.0 * max_error
  
   previous_kappa = 0.0
   self.current_iter = 0

   temperature_fourier = np.zeros(self.n_el)
   compute_sum(self.solve_fourier,1,output = {'temperature':temperature_fourier}, options = {})
   #---------------------------------------------------------------------
   #---------------------------------------------------------------------
   #lattice_temperature = np.array(self.n_mfp *[temperature_fourier])
   lattice_temperature = np.array(self.n_mfp *[temperature_fourier])
   boundary_temperature = temperature_fourier.copy()
   global_temperature = np.zeros(self.n_elems)
   thermal_flux = np.zeros((self.n_elems,3))

   #-----------------------------------
   if MPI.COMM_WORLD.Get_rank() == 0:
    if max_iter > 0:
     print('')
     print colored('Solving BTE... started', 'green')
     print('')
     print('    Iter    Thermal Conductivity [W/m/K]      Error       Zeroth - Fourier - BTE')
     print('   ------------------------------------------------------------------------------------')

   kappa_old = 0.0
   kappa = np.zeros(1)
   while error > max_error and self.current_iter < max_iter:

    compute_sum(self.solve_bte,self.n_mfp, \
                          output = {'temperature':lattice_temperature,'kappa':kappa,'boundary_temperature':boundary_temperature,'global_temperature':global_temperature,'thermal_flux':thermal_flux},\
                          options = {'temperature':lattice_temperature.copy(),'boundary_temperature':boundary_temperature.copy()})
   
    
    
    lattice_temperature = lattice_temperature.copy() 
    boundary_temperature = boundary_temperature.copy() 

    error = abs(kappa_old-kappa[0])/abs(kappa[0])
    kappa_old = kappa[0]
    self.current_iter +=1
    if MPI.COMM_WORLD.Get_rank() == 0:
    # print(boundary_temperature)
     print(kappa)
   #  print(min(boundary_temperature),max(boundary_temperature))
   self.temperature = global_temperature
   self.thermal_flux = thermal_flux

   if MPI.COMM_WORLD.Get_rank() == 0:
    if max_iter > 0 :
     print('   ------------------------------------------------------------------------------------')

   if max_iter > 0:
    if MPI.COMM_WORLD.Get_rank() == 0:
     print('')
     print colored('Solving BTE... done', 'green')
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
     data_tmp_b.append(self.mesh.get_side_periodic_value(ll,kc2)*v_orth)
     data_tmp_b2.append(self.mesh.get_side_periodic_value(ll,kc2))
     #---------------------
     B[kc1] += self.mesh.get_side_periodic_value(ll,kc2)*v_orth/vol1
     
    #Boundary_elements
    FF = np.zeros(self.n_elems)
    boundary_mask = np.zeros(self.n_elems)
    for side in self.mesh.side_list['Boundary']:
     elem = self.mesh.side_elem_map[side][0]
     vol = self.mesh.get_elem_volume(elem)
     area = self.mesh.get_side_area(side)
     FF[elem] = area/vol
     boundary_mask[elem] = 1.0

    #Thermo = np.zeros(self.n_elems)
    #T_fixed = np.zeros(self.n_elems)
    #FF_Thermo = np.zeros(self.n_elems)
    #Hot side-------
    #for side in self.mesh.side_list['Hot']:
    # elem = self.mesh.side_elem_map[side][0]
    # vol = self.mesh.get_elem_volume(elem)
    # area = self.mesh.get_side_area(side)
    # v_orth = self.mesh.get_side_orthognal_direction(side)
     #Thermo[elem] = v_orth/vol*0.5
    # FF_Thermo[elem] = v_orth/area
     #FF[elem] = area/vol
     #T_fixed[elem] = 0.5
     #row_tmp.append(elem)
     #col_tmp.append(elem)
     #data_tmp.append(v_orth/vol)

    #for side in self.mesh.side_list['Cold']:
    # elem = self.mesh.side_elem_map[side][0]
    # vol = self.mesh.get_elem_volume(elem)
    # area = self.mesh.get_side_area(side)
    # v_orth = self.mesh.get_side_orthognal_direction(side)
     #Thermo[elem] -= v_orth/vol*0.5
    # FF_Thermo[elem] = v_orth/area
     #FF[elem] = area/vol
     #T_fixed[elem] = -0.5
     #row_tmp.append(elem)
     #col_tmp.append(elem)
     #data_tmp.append(v_orth/vol)

 
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
   #self.Thermo = data['Thermo']
   #self.T_fixed = data['T_fixed']
   #self.FF_Thermo = data['FF_Thermo']
   self.boundary_mask = data['boundary_mask']


  def solve_fourier(self,n,options):

   
    A = self.F 
    B = self.B 
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

    #if options['verbose'] > 0:    
     
    
    temperature= np.zeros(self.n_elems) 
    while error > min_err and n_iter < max_iter :

     RHS = B + C
     #if not 'temperature' in options.keys() and len(self.mesh.side_list['Hot']) == 0:
     RHS[10] = 0.0
     temp = SU.solve(RHS)
     #print(min(temp),max(temp))
     temp = temp - (max(temp)+min(temp))/2.0
   
     (C,flux) = self.compute_non_orth_contribution(temp)

     kappa = self.compute_diffusive_thermal_conductivity(temp)
     error = abs((kappa - kappa_old)/kappa)
     kappa_old = kappa
     #if options['verbose'] > 0:    
     # if max_iter > 0:
     #  print('{0:7d} {1:20.4E} {2:25.4E}'.format(n_iter,kappa*self.kappa_bulk, error))
     n_iter +=1 
    #if options['verbose'] > 0:    
    # print('   ---------------------------------------------------')
    # print('  ')
    #suppression = np.zeros(len(options['kappa_bulk']))
    #suppression[n] = kappa

    #gradient_temperature_mfp[n] = self.mesh.compute_grad(temp)
    #gradient_temperature = self.mesh.compute_grad(temp)
    #suppression = np.zeros((len(options['kappa_bulk']),self.n_theta,self.n_phi))
    #for t in range(self.n_theta):
    # for p in range(self.n_phi):
    #  suppression[n,t,p] = kappa*self.dom['ss'][t][p][self.mesh.direction][self.mesh.direction]*3.0/4.0/np.pi

    output = {'temperature':temp}

    return output

  def compute_non_orth_contribution(self,temp) :

    #self.gradT = self.mesh.compute_grad(temp,self.side_periodic_value)
    self.gradT = self.mesh.compute_grad(temp)
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


  #def compute_diffusive_thermal_conductivity(self,temp):

  # kappa = 0
  # for i,j in zip(*self.RHS.nonzero()):
  #  kappa += 0.5*self.RHS[i,j]*(temp[j]+self.RHS[i,j]/abs(self.RHS[i,j])-temp[i])*self.mesh.size[0]/self.area_flux

  # return kappa


  #def compute_diffusive_thermal_conductivity(self,temp,mat=np.eye(3),TL = 0.0,G_R = 1e20):
  def compute_diffusive_thermal_conductivity(self,temp,mat=np.eye(3)):

   kappa = 0
   for i,j in zip(*self.PER.nonzero()):
    (v_orth,dummy) = self.mesh.get_decomposed_directions(i,j,rot=mat)
    kappa += 0.5*v_orth *self.PER[i,j]*(temp[j]+self.PER[i,j]-temp[i])

   #grad = self.mesh.compute_grad(temp)
   #kappa_2 = 0
   #for side in self.mesh.side_list['Cold']:
   # elem = self.mesh.side_elem_map[side][0]
   # area = self.mesh.compute_side_area(side)
    #if not self.mfe:
   # G_P = self.mesh.get_side_orthognal_direction(side)/area
    # G = pow(pow(G_P,-1) + pow(G_R,-1),-1)
    #else:
   # if G_R < 1e15:
   #   G = G_R/(1.0+G_R/G_P)
   # else:
   #   G = G_P


   # kappa_2 -= area*G*(-0.5 -temp[elem])
    #else :
    # normal = self.mesh.get_side_normal(0,side)    
    # kappa_2 -= np.dot(grad[elem],normal)
    # print(grad[elem])
     #kappa_2 = -factor*(-0.5-temp[elem])
 
   #kappa_3 = 0
   #for side in self.mesh.side_list['Hot']:
   # elem = self.mesh.side_elem_map[side][0]
   # v_orth = self.mesh.get_side_orthognal_direction(side)
   # kappa_3 -= v_orth*(0.5-temp[elem])


 
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
    print(' ')
    print('Elements:          ' + str(self.n_el))
    print('Azimuthal angles:  ' + str(self.n_theta))
    print('Polar angles:      ' + str(self.n_phi))
    print('Mean-free-paths:   ' + str(self.n_mfp))
    print('Degree-of-freedom: ' + str(self.n_mfp*self.n_theta*self.n_phi*self.n_el))





