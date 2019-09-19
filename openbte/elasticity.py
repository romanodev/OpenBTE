import numpy as np
import itertools
from openbte import Geometry
from scipy.sparse import *
from scipy.sparse.linalg import *
from scipy import *
from matplotlib.pyplot import *



#https://github.com/libAtoms/matscipy/blob/master/matscipy/elasticity.py
def cubic_to_Voigt_6x6(C11, C12, C44):
    return np.array([[C11,C12,C12,  0,  0,  0],
                     [C12,C11,C12,  0,  0,  0],
                     [C12,C12,C11,  0,  0,  0],
                     [  0,  0,  0,C44,  0,  0],
                     [  0,  0,  0,  0,C44,  0],
                     [  0,  0,  0,  0,  0,C44]])


def full_3x3_to_Voigt_6_index(i, j):
    if i == j:
        return i
    return 6-i-j

def isotropic_to_6x6(E,v):

  C = np.zeros((6,6))

  a = E/(1+v)/(1-2*v)

  C[0,0] = 1-v
  C[1,1] = 1-v
  C[2,2] = 1-v
  C[3,3] = (1-2*v)/2
  C[4,4] = (1-2*v)/2
  C[5,5] = (1-2*v)/2
  C[0,1] = v
  C[0,2] = v
  C[1,0] = v
  C[2,0] = v
  C[1,2] = v
  C[2,1] = v

  return C*a

def Voigt_6x6_to_full_3x3x3x3(C):

    C = np.asarray(C)
    C_out = np.zeros((3,3,3,3), dtype=float)
    for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3)):
        Voigt_i = full_3x3_to_Voigt_6_index(i, j)
        Voigt_j = full_3x3_to_Voigt_6_index(k, l)
        C_out[i, j, k, l] = C[Voigt_i, Voigt_j]
    return C_out

class Elasticity(object):

  def __init__(self,**argv):
   if 'geometry' in argv.keys():   
     self.mesh = argv['geometry']
     self.mesh._update_data()
   else: 
    self.mesh = Geometry(model='load',filename = argv.setdefault('geometry_filename','geometry.p'))
    
   C = argv['C']

   #C_voigt = cubic_to_Voigt_6x6(C['C11'],C['C12'],C['C44'])
   C_voigt = isotropic_to_6x6(C['E'],C['nu'])

   #print(C_voigt)

   self.C = Voigt_6x6_to_full_3x3x3x3(C_voigt)


   self.strain = argv['strain']
   self.pbcs = np.zeros((3,self.mesh.n_elems,self.mesh.n_elems))
   for s1 in self.mesh.periodic_sides.keys():
      (e1,e2) = self.mesh.side_elem_map[s1] 
      if s1 in self.mesh.side_list['Periodic']:
        s2 = self.mesh.periodic_sides[s1]
        L12 = self.mesh.compute_side_centroid(s2) - self.mesh.compute_side_centroid(s1)     
        u = np.dot(self.strain,L12)
        self.pbcs[:,e1,e2] = u

   self.solve()


  def compute_non_orth_contribution(self,u) :

    No = np.zeros(3*self.mesh.n_elems,dtype=float32)
    for i,k in zip(*self.iter):

      grad = self.mesh.compute_grad(u[k],add_jump=False,pbcs = self.pbcs[k])

      for e1,e2 in zip(*self.mesh.A.nonzero()):
       if not e1 == e2:

        #Get agerage gradient----
        side = self.mesh.get_side_between_two_elements(e1,e2)
        w = self.mesh.get_interpolation_weigths(side,e1)
        grad_ave = w*grad[e1] + (1.0-w)*grad[e2]
        #------------------------
        (dummy,v_non_orth) = self.mesh.get_decomposed_directions(e1,e2,rot=self.C[i,:,k,:])

        No[e1*3+i] += np.dot(grad_ave,v_non_orth)/2.0
        No[e2*3+i] -= np.dot(grad_ave,v_non_orth)/2.0


    return No


  def apply_strain(self):

    B = np.zeros(3*self.mesh.n_elems,dtype=float32)
    for s1 in self.mesh.periodic_sides.keys():
     if s1 in self.mesh.side_list['Periodic']:
      s2 = self.mesh.periodic_sides[s1]
      L12 = self.mesh.compute_side_centroid(s2) - self.mesh.compute_side_centroid(s1)     
      u = np.dot(self.strain,L12)
      (e1,e2) = self.mesh.side_elem_map[s1]
      for i,k in zip(*self.iter):
         (v_orth,dummy) = self.mesh.get_decomposed_directions(e1,e2,rot = self.C[i,:,k,:])
         B[e1*3 + i] += v_orth*u[k]#/v1  
         B[e2*3 + i] -= v_orth*u[k]#/v2

    #apply zero displacement along z
    for e in range(self.mesh.n_elems):
     B[e*3+2] = 0.0


    return B
 

  def assemble(self) :
    
    A = dok_matrix((3*self.mesh.n_elems,3*self.mesh.n_elems), dtype=float32)
    for e1,e2 in zip(*self.mesh.A.nonzero()):
     if not e1 == e2:
      for i,k in zip(*self.iter):
       (v_orth,dummy) = self.mesh.get_decomposed_directions(e1,e2,rot = self.C[i,:,k,:])
       A[e1*3+i,e1*3+k] += v_orth
       A[e1*3+i,e2*3+k] -= v_orth

    #Apply zero displacement along z
    for e in range(self.mesh.n_elems):
     A[e*3+2,e*3+2] = 1.0


    return A.tocsc()

  def compute_stress(self,u):


    grad = np.zeros((3,3,self.mesh.n_elems))
    for n in range(3):
     grad[n]= self.mesh.compute_grad(u[n],add_jump=False,pbcs = self.pbcs[n]).T

    stress = np.zeros((self.mesh.n_elems,3,3))
    for e1,e2 in zip(*self.mesh.A.nonzero()):

     if not e1 == e2:
      self.mesh.compute_grad(u[0])
 
      for i,j in zip(*self.iter):
       for k,l in zip(*self.iter):
        stress[:,i,j] += self.C[i,j,k,l]*grad[k,l]   


    return stress



  def compute_forces(self,u):

   tot_area=0.0
   S00 = 0
   for s1 in self.mesh.periodic_sides.keys():
    if s1 in self.mesh.side_list['Periodic']:
     s2 = self.mesh.periodic_sides[s1]
     area = self.mesh.compute_side_area(s1)
     
     (e1,e2) = self.mesh.side_elem_map[s1] 
     L12 = self.mesh.compute_side_centroid(s2) - self.mesh.compute_side_centroid(s1)     
     deltau = np.dot(self.strain,L12)
     L12 /= np.linalg.norm(L12)
     if abs(abs(L12[0]) -1)< 1e-3: 
      tot_area +=area
      for l in range(3):
      #for k,l in zip(*self.iter):
       deltax = u[l,e2] - u[l,e1] + deltau[l]
       (v_orth,dummy) = self.mesh.get_decomposed_directions(e1,e2,rot = self.C[0,:,:,l]) #over k
       S00 += deltax*v_orth
 
   return S00/tot_area

  def solve(self):


   self.iter = [[0,0,0,1,1,1,2,2,2],[0,1,2,0,1,2,0,1,2]]
   
   A = self.assemble()
   B = self.apply_strain()

   #Cycle----
   max_iter = 20
   max_error= 1e-3
   n_iter = 0
   C_0000_old = 0
   No = np.zeros(3*self.mesh.n_elems)
   error = 2*max_error
   while error > max_error and n_iter < max_iter :

    RHS = B + No

    x = spsolve(A,RHS)

    u = np.zeros((3,self.mesh.n_elems))
    for n in range(self.mesh.n_elems):
     for l in range(3):
      u[l,n] = x[3*n+l]

    self.u = u
    #scalting
    for i in range(3):
     mp = (max(u[i]) + min(u[i]))/2
     u[i] -= mp 
    stress = self.compute_forces(u)

    E = abs(stress/self.strain[0,0])
 
    print(E)
    #Eerror = abs(C_0000-C_0000_old)/C_0000
    #print(error)
    #EC_0000_old = C_0000

    No = self.compute_non_orth_contribution(u)#/10#/10

    n_iter +=1

    #self.C_0000 = C_0000
 
   self.stress = self.compute_stress(u)
  

#Plot(variable='map/dd',data=u[0])
#--------
