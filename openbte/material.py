import numpy as np
import math
from compute_dom import *
from mpi4py import MPI
import os
import os.path
import h5py
import deepdish as dd


class Material(object):

 def __init__(self,**argv):

  #defaults--
  argv.setdefault('grid',[50,12,24])

  #MFP discretization-----------------------------
  self.state = {}
  if argv.setdefault('model','isotropic') == 'isotropic':
    self.import_data(argv)
  if argv.setdefault('model','isotropic') == 'load':
   #MPI.COMM_WORLD.Barrier()
   if MPI.COMM_WORLD.Get_rank() == 0:
    data = dd.io.load('material.hdf5')
   else: data=None
   self.state.update(MPI.COMM_WORLD.bcast(data,root=0))
  #-----------------------------------------------

  #Angular discretization-------------------------
  if MPI.COMM_WORLD.Get_rank() == 0:
   dom = compute_dom_3d(argv) 
   data = {'dom':dom}
  else: data=None
  self.state.update(MPI.COMM_WORLD.bcast(data,root=0))
  #-----------------------------------------------

  #SAVE FILE--------------------
  if argv.setdefault('save',True):
   if MPI.COMM_WORLD.Get_rank() == 0:
    dd.io.save('material.hdf5', self.state)



 def import_data(self,argv):

  if MPI.COMM_WORLD.Get_rank() == 0:
    filename = os.path.dirname(__file__) + '/materials/'+ argv['filename'] + '.dat'
    data = {}
    [n_mfp,n_theta,n_phi] = argv['grid']
    #READ DATA-------------------------------------------
    tmp = np.loadtxt(filename)


    mfp_0 = 1e-12
    n_bands = -1
    for n in range(len(tmp)):
     n_bands = int(max(n_bands,tmp[n][0]))
    n_mfp_bulk = int(len(tmp)/n_bands)
    data.update({'n_bands':n_bands})
    data.update({'n_mfp_bulk':n_mfp_bulk})
    mfp = np.zeros((n_bands,n_mfp_bulk))
    for b in range(n_bands):
     for m in range(n_mfp_bulk):
      n = n_mfp_bulk * b + m
      mfp[b][m] = tmp[n][2]
    #-----------------------------------------------------

    #---DISCRETIZE MFPS in LOG SPACE---------------------
    m1 = argv.setdefault('min_mfp',np.min(mfp)*0.99)
    m2 = argv.setdefault('max_mfp',np.max(mfp)*1.09)
    mfp_sampled_log = np.linspace(np.log10(m1),np.log10(m2),n_mfp) 
    mfp_sampled = []
    for m in mfp_sampled_log:
     mfp_sampled.append(pow(10,m))
    #--------------------------------------------------------
  
    #MFP interpolation---------------------------
    B0 = np.zeros(n_mfp) 
    B1 = np.zeros(n_mfp) 
    B2 = np.zeros(n_mfp) 
    kappa_tot = 0
    for b in range(n_bands):
     for q in range(n_mfp_bulk):
      n = n_mfp_bulk * b + q
      kappa = np.max([tmp[n][3],0.0])
      mfp[b][q] = tmp[n][2]
      [m1,m2,f1,f2] = self.get_mfp_interpolation(mfp_sampled_log,np.log10(mfp[b][q]))
      b0 = kappa
      b1 = kappa/mfp[b][q]
      b2 = kappa/mfp[b][q]/mfp[b][q]
      B0[m1] += f1*b0; B0[m2] += f2*b0
      B1[m1] += f1*b1; B1[m2] += f2*b1
      B2[m1] += f1*b2; B2[m2] += f2*b2
      kappa_tot += np.max([tmp[n][3],0.0])
    B0 /=np.sum(B0)
    B1 /=np.sum(B1)
    B2 /=np.sum(B2)
    #---------------------------------------------

    #Compute DOM
    #----------------
    data.update({'kappa_bulk':kappa_tot})
    data.update({'B0':B0})
    data.update({'B1':B1})
    data.update({'B2':B2})
    data.update({'mfp_sampled':mfp_sampled})
  else: data=None
  self.state = MPI.COMM_WORLD.bcast(data,root=0)

 #def get_material_properties(self):
 # return self.state



 def get_mfp_interpolation(self,mfp_sampled,mfp_bulk) :

   n_mfp = len(mfp_sampled) 
   findm = 0
   for m in range(n_mfp-1) :
    if (mfp_bulk >= mfp_sampled[m]) and (mfp_bulk <= mfp_sampled[m+1]) : 
         
     #get data for mfp interpolation
     m1 = m
     m2 = m+1
     mfp1 = mfp_sampled[m1]
     mfp2 = mfp_sampled[m2] 
     fm = (mfp_bulk-mfp1)/(mfp2 - mfp1) #for interpolation
     findm = 1
     m_par = [m1,m2,1-fm,fm]
     assert mfp_bulk <= mfp2
     assert mfp_bulk >= mfp1
     assert abs(fm) <= 1
     return m_par

   assert findm == 1
     

