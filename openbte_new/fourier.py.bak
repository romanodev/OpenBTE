from scipy.sparse.linalg import spsolve
import os,sys
import numpy as np 
from scipy.sparse import csc_matrix
import numpy as np
from mpi4py import MPI
from scipy.sparse.linalg import *
from scipy.sparse import diags
from mpi4py import MPI
from utils import *



class Fourier(object):
 
  def __init__(self,argv):

   #if  MPI.COMM_WORLD.Get_rank() == 0:
  
   self.mesh = argv['geometry']
   self.mat = argv['material'].state
   self.kappa_bulk = self.mat['kappa_bulk_tot']
   #INITIALIZATION-------------------------
   self.n_el = len(self.mesh.elems)
   data = self.mesh.compute_boundary_condition_data('x')
   self.side_periodic_value = data['side_periodic_value']
   self.area_flux = data['area_flux']
   self.flux_sides = data['flux_sides']
   self.n_elems = len(self.mesh.elems)
   self.n = 10
   self.compute_connection_matrix()
   self.kappa_factor = 0.5*self.mesh.size[0]/self.area_flux

   if  MPI.COMM_WORLD.Get_rank() == 0:
    print(' ')
    print('Solving Fourier... started.')


   n_sim = 1
   output = {'suppression':np.array([0.0]),'temperature':np.zeros(self.n_el),'flux':np.zeros((self.n_el,3))}
   output = compute_sum(self.solve,n_sim,output,{})
   
   self.suppression = output['suppression']
   self.kappa = self.kappa_bulk*self.suppression[0] 
   self.temperature = output['temperature'] 
   self.flux = output['flux'] 


   #data = self.compute_function()
   if  MPI.COMM_WORLD.Get_rank() == 0:
    print('Solving Fourier... done.')
    print(' ')
    print('Effective Thermal Conductivity (Fourier):   ' + str(round(self.kappa,4)) + r' W/m/K')
    print(' ')
   #else: 
   #data = None
   #MPI.COMM_WORLD.Barrier()
   #data = MPI.COMM_WORLD.bcast(data,root=0)
   #self.temperature = data['temperature']
   #self.flux = data['flux']
   

  def compute_connection_matrix(self) :
   
   if  MPI.COMM_WORLD.Get_rank() == 0:
    row_tmp = []
    col_tmp = []
    data_tmp = [] 
    data_tmp_b = [] 
    for kc1,kc2 in zip(*self.mesh.A.nonzero()):
     ll = self.mesh.get_side_between_two_elements(kc1,kc2)  
     (v_orth,dummy) = self.mesh.get_decomposed_directions(kc1,kc2)
     row_tmp.append(kc1)
     col_tmp.append(kc2)
     data_tmp.append(v_orth)
     data_tmp_b.append(self.mesh.get_side_periodic_value(ll,kc2,self.side_periodic_value)*v_orth)
    A = csc_matrix( (np.array(data_tmp),(np.array(row_tmp),np.array(col_tmp))), shape=(self.n_elems,self.n_elems) )
    RHS = csc_matrix( (np.array(data_tmp_b),(np.array(row_tmp),np.array(col_tmp))), shape=(self.n_elems,self.n_elems) )
    b = [tmp[0,0] for tmp in A.sum(axis=1)]
    F = diags([b],[0]).tocsc()-A
    F[self.n,self.n] = 1.0
    B = np.array(RHS.sum(axis=1).T)[0]
    data = {'F':F,'RHS':RHS,'B':B}
   else: data = None
   data =  MPI.COMM_WORLD.bcast(data,root=0)
   
   self.F = data['F']
   self.RHS = data['RHS']
   self.B = data['B']




  def solve(self,n,options):
 
    SU = splu(self.F)
    C = np.zeros(self.n_el)
    #--------------------------------------
    min_err = 1e-3
    error = 2*min_err
    max_iter = 10
    n_iter = 0
    kappa_old = 0
    
    print('  ')
    print('    Iter    Thermal Conductivity [W/m/K]      Error')
    print('   ---------------------------------------------------')
    while error > min_err and n_iter < max_iter :

     RHS = self.B + C
     RHS[self.n] = 0.0
     temp = SU.solve(RHS)
     temp = temp - (max(temp)+min(temp))/2.0
     (C,flux) = self.compute_non_orth_contribution(temp)

     self.temp = temp
     kappa = self.compute_thermal_conductivity()
     error = abs((kappa - kappa_old)/kappa)
     kappa_old = kappa
     print('{0:7d} {1:20.4E} {2:25.4E}'.format(n_iter, kappa*self.kappa_bulk, error))
     n_iter +=1 
    print('   ---------------------------------------------------')
    print('  ')
    output =  {'suppression':kappa,'temperature':temp,'flux':flux}
    return output


  def compute_thermal_conductivity(self):
   

   kappa = 0
   for i,j in zip(*self.RHS.nonzero()):
    side = self.mesh.get_side_between_two_elements(i,j)  
    area = self.mesh.get_side_area(side) 
    (v_orth,dummy) = self.mesh.get_decomposed_directions(i,j)
    kappa += 0.5*self.RHS[i,j]*(self.temp[j]+self.RHS[i,j]/abs(self.RHS[i,j])-self.temp[i])*self.mesh.size[0]/self.area_flux


   #return np.sum(self.RHS.multiply(self.RHS-self.temp_ma)) * self.kappa_factor
   return kappa



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
     
     C[i] += np.dot(grad_ave,v_non_orth)/2.0
     C[j] -= np.dot(grad_ave,v_non_orth)/2.0

    return C,-self.gradT*self.kappa_bulk

