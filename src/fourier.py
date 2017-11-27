from scipy.sparse.linalg import spsolve
import os,sys
import numpy as np 
from scipy.sparse import csc_matrix
import numpy as np
from mpi4py import MPI
from scipy.sparse.linalg import *
from scipy.sparse import diags
from mpi4py import MPI



class Fourier(object):
 
  def __init__(self,argv):

   if  MPI.COMM_WORLD.Get_rank() == 0:
  
    self.mesh = argv['geometry']
    self.mat = argv['material'].state

    self.kappa_bulk = self.mat['kappa_bulk_tot']
    #INITIALIZATION-------------------------
    self.n_el = len(self.mesh.elems)
    data = self.mesh.compute_boundary_condition_data('x')
    self.side_periodic_value = data['side_periodic_value']
    self.area_flux = data['area_flux']
    self.compute_connection_matrix()
    self.x = np.ones(len(self.mesh.elems))
    self.n = np.random.randint(0,len(self.mesh.elems))
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
   self.temperature = data['temperature']
   self.flux = data['flux']
   





  def compute_connection_matrix(self) :
   
   nc = len(self.mesh.elems)
   row_tmp = []
   col_tmp = []
   data_tmp = [] 
   data_tmp_b = [] 
   #B = np.zeros(nc) 

   for ll in self.mesh.side_list['active'] :
    if not ll in self.mesh.side_list['Boundary'] :

     (kc1,kc2) = self.mesh.get_elems_from_side(ll)

     row_tmp.append(kc1)
     col_tmp.append(kc2)
     data_tmp.append(1.0)
     data_tmp_b.append(self.mesh.get_side_periodic_value(ll,kc2,self.side_periodic_value))

     row_tmp.append(kc2)
     col_tmp.append(kc1)
     data_tmp.append(1.0)
     data_tmp_b.append(self.mesh.get_side_periodic_value(ll,kc1,self.side_periodic_value))
  
   self.A = csc_matrix( (np.array(data_tmp),(np.array(row_tmp),np.array(col_tmp))), shape=(nc,nc) )
   self.RHS = csc_matrix( (np.array(data_tmp_b),(np.array(row_tmp),np.array(col_tmp))), shape=(nc,nc) )


  def compute_function(self):
 
    nc = len(self.x)
    #Add kappa------------------
    r = []
    c = []
    d = [] 
    for i,j in zip(*self.A.nonzero()):
     r.append(i)
     c.append(j)
     (v_orth,dummy) = self.mesh.get_decomposed_directions(i,j)
     kappa = self.get_kappa(i,j)*v_orth
     d.append(self.A[i,j]*kappa)
    A = csc_matrix( (np.array(d),(np.array(r),np.array(c))), shape=(nc,nc) )
    b = [tmp[0,0] for tmp in A.sum(axis=1)]
    F = diags([b],[0]).tocsc()-A
    F[self.n,self.n] = 1.0
    SU = splu(F)
    
    #-----------------------------------
    #BUILD RHS--------------------------------- 
    B = np.zeros(nc)
    for i,j in zip(*self.RHS.nonzero()):
     kappa = self.get_kappa(i,j)
     (v_orth,dummy) = self.mesh.get_decomposed_directions(i,j)
     B[i] += self.RHS[i,j]*kappa*v_orth
    #------------------------------------------


    #Start the cycle------------------------
    C = np.zeros(self.n_el)
    #--------------------------------------
    min_err = 1e-3
    error = 2*min_err
    max_iter = 5
    n_iter = 0
    kappa_old = 0
    
    print('  ')
    print('    Iter    Thermal Conductivity [W/m/K]      Error')
    print('   ---------------------------------------------------')
    while error > min_err and n_iter < max_iter :

     RHS = B + C
     RHS[self.n] = 0.0
     temp = SU.solve(RHS)
     temp = temp - (max(temp)+min(temp))/2.0
     (C,flux) = self.compute_non_orth_contribution(temp)
     kappa = self.compute_thermal_conductivity(temp)
     error = abs((kappa - kappa_old)/kappa)
     kappa_old = kappa
     print('{0:7d} {1:20.4f} {2:25.4E}'.format(n_iter, kappa*self.kappa_bulk, error))
     n_iter +=1 
    print('   ---------------------------------------------------')
    print('  ')

    return {'kappa':kappa*self.kappa_bulk,'temperature':temp,'flux':flux}

 

  def get_kappa(self,c1,c2):

   x1 = self.x[c1]
   x2 = self.x[c2]

   if x1 == 0 and x2 == 0:
    return 0.0

   return 2.0*x1*x2/(x1 + x2)


  def compute_thermal_conductivity(self,temp):

   kappa = 0
   for i,j in zip(*self.RHS.nonzero()):
    (v_orth,dummy) = self.mesh.get_decomposed_directions(i,j)
    density = self.get_kappa(i,j) 
    kappa += 0.5*self.RHS[i,j]*density*(temp[j]+self.RHS[i,j]-temp[i])*self.mesh.size[0]/self.area_flux*v_orth

   return kappa



  def compute_non_orth_contribution(self,temp) :

    x = self.x
    gradT = self.mesh.compute_grad(temp,self.side_periodic_value)
    C = np.zeros(self.n_el)
    for i,j in zip(*self.A.nonzero()):
     #Get agerage gradient----
     side = self.mesh.get_side_between_two_elements(i,j)
     w = self.mesh.get_interpolation_weigths(side)
     grad_ave = w*gradT[i] + (1.0-w)*gradT[j]
     #------------------------
     kappa = self.get_kappa(i,j)
     (dumm,v_non_orth) = self.mesh.get_decomposed_directions(i,j)
     
     C[i] += np.dot(grad_ave,v_non_orth*kappa)/2.0
     C[j] -= np.dot(grad_ave,v_non_orth*kappa)/2.0

    return C,-gradT*self.kappa_bulk








































































































































































