from __future__ import absolute_import
import numpy as np
import math
from .compute_dom import *
from mpi4py import MPI
import os
import os.path
import deepdish as dd
#from matplotlib.pylab import *

class Material(object):

 def __init__(self,**argv):

  #defaults--
  #argv.setdefault('grid',[50,12,24])

  #MFP discretization-----------------------------
  self.state = {}
 

  argv.setdefault('model','gray')
  if argv['model'] == 'load':
   #MPI.COMM_WORLD.Barrier()
   if MPI.COMM_WORLD.Get_rank() == 0:
    data = dd.io.load(argv.setdefault('filename','material.hdf5'))
   else: data=None
   self.state.update(MPI.COMM_WORLD.bcast(data,root=0))
   #print(self.state['region'])
  else:

   if MPI.COMM_WORLD.Get_rank() == 0:
    dom = compute_dom_2d(argv)

   if argv['model'] == 'nongray':
     argv.update({'dom':dom})   
     self.import_data(argv)

   
   if argv['model'] == 'gray':
    self.create_gray_model(argv)
   #-----------------------------------------------

   #Angular discretization-------------------------
   if MPI.COMM_WORLD.Get_rank() == 0:
    region = argv.setdefault('region','Matrix')
    data = {'dom':dom,'region':region}

   else: data=None
   self.state.update(MPI.COMM_WORLD.bcast(data,root=0))

   #-----------------------------------------------
   #SAVE FILE--------------------
   if argv.setdefault('save',True):
    if MPI.COMM_WORLD.Get_rank() == 0:
     dd.io.save(argv.setdefault('filename','material') + '.hdf5', self.state)


 #def get_region(self):
     #return self.region

 def create_gray_model(self,argv):

  if MPI.COMM_WORLD.Get_rank() == 0:
    data = {}
    mfp = argv.setdefault('mfps',[100e-9])
    n_mfp = len(mfp)
    kappa = argv.setdefault('kappa',1.0)

    data.update({'kappa_bulk_tot':kappa})
    data.update({'mfp_bulk':mfp})
    data.update({'kappa_bulk':[kappa]})
    data.update({'B0':np.array(n_mfp*[1.0/n_mfp])})

    if 'special_point' in argv.keys():
     sp = argv['special_point']
     tmp = np.zeros(n_mfp)
     tmp[sp] = 1.0
     tmp = np.tile(tmp,[n_mfp,1])
     data.update({'B1':tmp})
     data.update({'B2':tmp})
    else:
     data.update({'B1':np.eye(n_mfp)})
     data.update({'B2':np.eye(n_mfp)})


    data.update({'mfp_sampled':mfp})
  else: data=None
  self.state = MPI.COMM_WORLD.bcast(data,root=0)




 def import_data(self,argv):

  if MPI.COMM_WORLD.Get_rank() == 0:

    if '/' in argv['matfile']:
     filename = argv['matfile']
    else:
     filename = os.path.dirname(__file__) + '/materials/'+ argv['matfile']
    #-------
    data = {}

    n_rmfp = int(argv['n_rmfp'])
    
    #READ DATA-------------------------------------------
    tmp = np.loadtxt(filename)
    mfp_bulk = tmp[:,0]
    kappa_bulk = tmp[:,1]
    n_mfp_bulk = len(mfp_bulk)
    
    #experimental----------
    ftheta = argv['dom']['ftheta']
    dtheta_ave = argv['dom']['dtheta_ave']
   
    dtheta = argv['dom']['dtheta']
    n_theta = len(dtheta_ave)
    #---------------------------------
    
    min_rmfp = 0.99*min(mfp_bulk)*min(ftheta)
    max_rmfp = 1.01*max(mfp_bulk)*max(ftheta)
     
    
    #----------------------
    rmfp_log = np.linspace(np.log10(min_rmfp*0.99),np.log10(max_rmfp*1.01),n_rmfp)

    #---DISCRETIZE MFPS in LOG SPACE---------------------
    rmfp = []
    for m in rmfp_log:
     rmfp.append(pow(10,m))
    
    #--------------------------------------------------------
    #MFP interpolation---------------------------
   # B0 = np.zeros(n_rmfp)
    B1 = np.zeros(n_rmfp)
    B2 = np.zeros(n_rmfp)
    factor = 0
    #kappa_tot = 0
    for m in range(n_mfp_bulk):
     kappa = kappa_bulk[m]
     mfp = mfp_bulk[m]
     a = kappa/mfp
     factor += a
     for t in range(n_theta) :   
      [n1,n2,f1,f2] = self.get_mfp_interpolation(rmfp_log,np.log10(mfp*ftheta[t]))
     #n1,n2,f1,f2] = self.get_mfp_interpolation(rmfp_log,np.log10(mfp))
      #b0 = kappa * dtheta_ave[t]
      b1 = a*  ftheta[t]*dtheta[t]
      b2 = kappa/mfp/mfp*dtheta_ave[t]
      #B0[n1] += f1*b0; B0[n2] += f2*b0
      B1[n1] += f1*b1; B1[n2] += f2*b1
      B2[n1] += f1*b2; B2[n2] += f2*b2
    #B0 /=np.sum(B0)
    
    
    B1 /=factor
    B2 /=np.sum(B2)
    
    #---------------------------------------------
    #Compute DOM
    #----------------
    data.update({'kappa_bulk':sum(kappa_bulk)})
    data.update({'mfp_bulk':mfp_bulk})
    data.update({'rmfp':rmfp})
    data.update({'factor':factor})
    
    data.update({'B1':np.tile(B1,(n_rmfp,1))})
    data.update({'B2':np.tile(B2,(n_rmfp,1))})
    data.update({'rmfp':np.array(rmfp)})

  else: data=None
  self.state = MPI.COMM_WORLD.bcast(data,root=0)


 def get_mfp_interpolation(self,mfp_sampled,mfp_bulk) :

   n_mfp = len(mfp_sampled)
   findm = 0
   for m in range(n_mfp-1) :
    if (mfp_bulk >= mfp_sampled[m]) and (mfp_bulk <= mfp_sampled[m+1]) :
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
   assert findm == 1
   return m_par

