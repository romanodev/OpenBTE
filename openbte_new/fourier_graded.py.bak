from scipy.sparse.linalg import spsolve
import os,sys
import numpy as np 
from scipy.sparse import csc_matrix
import numpy as np
from mpi4py import MPI
from scipy.sparse.linalg import *
from scipy.sparse import diags
from mpi4py import MPI


class Fourier_graded(object):
 
  def __init__(self,argv):

   if  MPI.COMM_WORLD.Get_rank() == 0:
  
    self.mesh = argv['geometry']
    self.mat = argv['material'].state
    self.mat_b = argv['material_b'].state
    #self.mat2 = argv.setdefault('secondary_material',argv['material']).state
    self.compute_gradient = argv.setdefault('compute_gradient',False)
    #self.compute_gradient = False
    #self.kappa_bulk_2 = self.mat2['kappa_bulk_tot']
    self.kappa_bulk = self.mat['kappa_bulk_tot']
    self.kappa_bulk_b = self.mat_b['kappa_bulk_tot']
    #INITIALIZATION-------------------------
    self.n_el = len(self.mesh.elems)
    data = self.mesh.compute_boundary_condition_data('x')
    self.side_periodic_value = data['side_periodic_value']
    self.area_flux = data['area_flux']
    self.flux_sides = data['flux_sides']
    self.x = argv.setdefault('density',np.ones(len(self.mesh.elems)))
    #self.x = np.ones(len(self.mesh.elems))
    self.compute_connection_matrix()
    self.kappa_factor = 0.5*self.mesh.size[0]/self.area_flux

    
    #self.n = np.random.randint(0,len(self.mesh.elems))
    self.n = 10
    print(' ')
    print('Solving Fourier... started.')
    data = self.compute_function()
    print('Solving Fourier... done.')
    print(' ')
    print('Effective Thermal Conductivity (Fourier):   ' + str(round(data['kappa'],4)) + r' W/m/K')
    print(' ')
   else: 
    data = None
   MPI.COMM_WORLD.Barrier()
   data = MPI.COMM_WORLD.bcast(data,root=0)
   self.kappa = data['kappa']
   self.gradient = data['gradient']
   self.T_der = data['T_der']
   self.temperature = data['temperature']
   self.flux = data['flux']
   


  #def compute_connection_matrix(self) :
   
  # nc = len(self.mesh.elems)
  # row_tmp = []
  # col_tmp = []
  # data_tmp = [] 
  # data_tmp_b = [] 
   #B = np.zeros(nc) 

  # for ll in self.mesh.side_list['active'] :
  #  if not ll in self.mesh.side_list['Boundary'] :

   #  (kc1,kc2) = self.mesh.get_elems_from_side(ll)

   #  row_tmp.append(kc1)
   #  col_tmp.append(kc2)
   #  data_tmp.append(1.0)
   #  data_tmp_b.append(self.mesh.get_side_periodic_value(ll,kc2,self.side_periodic_value))

   #  row_tmp.append(kc2)
   #  col_tmp.append(kc1)
   #  data_tmp.append(1.0)
   #  data_tmp_b.append(self.mesh.get_side_periodic_value(ll,kc1,self.side_periodic_value))
  
   #self.A = csc_matrix( (np.array(data_tmp),(np.array(row_tmp),np.array(col_tmp))), shape=(nc,nc) )



  def compute_connection_matrix(self) :
   
   nc = len(self.mesh.elems)
   row_tmp = []
   col_tmp = []
   data_tmp = [] 
   data_tmp_b = [] 
   data_kappa = []
   data_kappa_der = []
   #B = np.zeros(nc) 

   for ll in self.mesh.side_list['active'] :
    if not ll in self.mesh.side_list['Boundary'] :

     (kc1,kc2) = self.mesh.get_elems_from_side(ll)
  
     row_tmp.append(kc1)
     col_tmp.append(kc2)

     (v_orth,dummy) = self.mesh.get_decomposed_directions(kc1,kc2)

     kappa = self.get_kappa(kc1,kc2)
 
     kappa_der = self.get_kappa_derivative(kc1,kc2)

     data_tmp.append(v_orth)

     data_kappa.append(kappa)
     data_kappa_der.append(kappa_der)

     data_tmp_b.append(self.mesh.get_side_periodic_value(ll,kc2,self.side_periodic_value)*v_orth)

     row_tmp.append(kc2)
     col_tmp.append(kc1)
     kappa_der = self.get_kappa_derivative(kc2,kc1)
     data_kappa_der.append(kappa_der)
     data_tmp.append(v_orth)
     data_tmp_b.append(self.mesh.get_side_periodic_value(ll,kc1,self.side_periodic_value)*v_orth)
     data_kappa.append(kappa)
  
   self.A = csc_matrix( (np.array(data_tmp),(np.array(row_tmp),np.array(col_tmp))), shape=(nc,nc) )
   self.Gamma = csc_matrix( (np.array(data_kappa),(np.array(row_tmp),np.array(col_tmp))), shape=(nc,nc) )
   self.Gamma_der = csc_matrix( (np.array(data_kappa_der),(np.array(row_tmp),np.array(col_tmp))), shape=(nc,nc) )
   self.RHS = csc_matrix( (np.array(data_tmp_b),(np.array(row_tmp),np.array(col_tmp))), shape=(nc,nc) )


  def compute_function(self):
 
    nc = len(self.x)
    #NEW-----------------------------------------------------------------
    A = self.A.multiply(self.Gamma)
    b = [tmp[0,0] for tmp in A.sum(axis=1)]
    F = diags([b],[0]).tocsc()-A
    F[self.n,self.n] = 1.0
    SU = splu(F)
    B = np.array(self.RHS.multiply(self.Gamma).sum(axis=1).T)[0]
    #B = np.array(self.RHS.sum(axis=1).T)[0]
    #-------------------------------------------------------------------

    #Start the cycle------------------------
    C = np.zeros(self.n_el)
    #--------------------------------------
    min_err = 1e-3
    error = 2*min_err
    max_iter = 5
    n_iter = 0
    kappa_old = 0
    gradient = np.zeros(self.n_el)
    T_der = np.zeros((self.n_el,self.n_el))
    
    print('  ')
    print('    Iter    Thermal Conductivity [W/m/K]      Error')
    print('   ---------------------------------------------------')
    while error > min_err and n_iter < max_iter :

     RHS = B + C
     RHS[self.n] = 0.0
     temp = SU.solve(RHS)
     temp = temp - (max(temp)+min(temp))/2.0
     self.calculate_temp_matrix(temp)   
  
     if self.compute_gradient:
       gradient,T_der = self.calculate_gradient()
    
     (C,flux) = self.compute_non_orth_contribution(temp)
     self.temp = temp
     kappa = self.compute_thermal_conductivity()
     error = abs((kappa - kappa_old)/kappa)
     kappa_old = kappa
     print('{0:7d} {1:20.4f} {2:25.4E}'.format(n_iter, kappa*self.kappa_bulk, error))
     n_iter +=1 
    print('   ---------------------------------------------------')
    print('  ')

    return {'kappa':kappa*self.kappa_bulk,'temperature':temp,'flux':flux,'gradient':gradient,'T_der':T_der}

  def get_kappa_derivative(self,c1,c2):

   x1 = self.x[c1]
   x2 = self.x[c2]

   if x1 == 0 and x2 == 0:
    return 0.0

   return 2.0*x2*x2/(x1 + x2)/(x1 + x2)

  def get_kappa(self,c1,c2):


   x1 = self.x[c1]
   x2 = self.x[c2]

   k1 = (1.0-x1)*self.kappa_bulk + x1*self.kappa_bulk_b
   k2 = (1.0-x2)*self.kappa_bulk + x2*self.kappa_bulk_b
   

   return 2.0*k1*k2/(k1 + k2)


  def compute_thermal_conductivity(self):
   

   kappa = 0
   for i,j in zip(*self.RHS.nonzero()):
    side = self.mesh.get_side_between_two_elements(i,j)  
    area = self.mesh.get_side_area(side) 

    if side in self.flux_sides:
     (v_orth,dummy) = self.mesh.get_decomposed_directions(i,j)
     kappa += 0.5*self.RHS[i,j]*self.Gamma[i,j]*(self.temp[j]+self.RHS[i,j]/abs(self.RHS[i,j])-self.temp[i])*self.mesh.size[0]/self.area_flux


   return kappa
   #kappa = self.RHS.multiply(self.Gamma.multiply(self.RHS-self.temp_mat))
   #return kappa
   #return np.sum(self.RHS.multiply(self.Gamma.multiply(self.RHS-self.temp_mat))) * self.kappa_factor
  
   #return np.sum(self.RHS.multiply(self.Gamma.multiply(self.RHS-self.temp_mat))) * self.kappa_factor



  def compute_non_orth_contribution(self,temp) :

    x = self.x
    self.gradT = self.mesh.compute_grad(temp,self.side_periodic_value)
    C = np.zeros(self.n_el)
    for i,j in zip(*self.A.nonzero()):
     #Get agerage gradient----
     side = self.mesh.get_side_between_two_elements(i,j)
     w = self.mesh.get_interpolation_weigths(side)
     grad_ave = w*self.gradT[i] + (1.0-w)*self.gradT[j]
     #------------------------
     kappa = self.get_kappa(i,j)
     (dumm,v_non_orth) = self.mesh.get_decomposed_directions(i,j)
     
     C[i] += np.dot(grad_ave,v_non_orth*kappa)/2.0
     C[j] -= np.dot(grad_ave,v_non_orth*kappa)/2.0

    return C,-self.gradT*self.kappa_bulk

  def get_kappa_derivative(self,i,j):

   x1 = self.x[i]
   x2 = self.x[j]
   return 2.0*x2*x2/(x1+x2)/(x1+x2)



  def calculate_temp_matrix(self,temp) :

    n_el = len(temp)
    tmp = np.zeros((n_el,n_el))
    for i in range(n_el): 
     for j in range(n_el): 
      tmp[i,j] = temp[i]-temp[j]
    self.temp_mat = csc_matrix(tmp)
   
    #H--------------------------------
  def calculate_gradient(self):  

    nc = len(self.x)
    #Recompute F----------------------------------
    A = self.A.multiply(self.Gamma)
    b = [tmp[0,0] for tmp in A.sum(axis=1)]
    F = diags([b],[0]).tocsc()-A
    F[self.n,self.n] = 1.0
    #------------------------------------

    S = self.A.multiply(self.Gamma_der.multiply(self.temp_mat))
    b = [tmp[0,0] for tmp in S.sum(axis=1)]
    H = diags([b],[0]).tocsc()-S.T
    #---------------------------------


    S = self.RHS.multiply(self.Gamma_der)
    b = [tmp[0,0] for tmp in S.sum(axis=1)]
    B =  diags([b],[0]).tocsc()-S.T
    #-----------------------------------

  
    #GP
    S = self.RHS.multiply(self.Gamma_der.multiply(self.RHS-self.temp_mat))
    GP = 2.0*np.array((S.sum(axis=1)).T)[0]*self.kappa_factor
    
    #GX
    S = self.RHS.multiply(self.Gamma)
    GX = 2.0*np.array((S.sum(axis=1)).T)[0]*self.kappa_factor


    fp = B-H
    L = spsolve(F.T,GX,use_umfpack = False)

    T_der = inv(F.tocsc())*fp

    return (GP - L.T*fp),T_der.todense()







































































































































































