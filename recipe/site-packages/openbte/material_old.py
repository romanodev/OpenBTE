from __future__ import absolute_import
import numpy as np
import math
from .compute_dom import *
from mpi4py import MPI
import os
import os.path
import deepdish as dd


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

   if argv['model'] == 'nongray':
    self.import_data(argv)

   if argv['model'] == 'gray':
    self.create_gray_model(argv)
   #-----------------------------------------------

   #Angular discretization-------------------------
   if MPI.COMM_WORLD.Get_rank() == 0:
    region = argv.setdefault('region','Matrix')

    dom = compute_dom_3d(argv)
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

    data = {}
    #[n_mfp,n_theta,n_phi] = argv['grid']

    n_mfp = int(argv['n_mfp'])
    #READ DATA-------------------------------------------
    tmp = np.loadtxt(filename)
    kappa_bulk_tot = np.sum(tmp[:,1])
    mfp_bulk = tmp[:,0]
    kappa_bulk = tmp[:,1]
    n_mfp_bulk = len(mfp_bulk)
    #-----------------------------------------------------
    delta = argv.setdefault('delta',0.0)
    acc = 0
    mfp_bulk_new = []
    kappa_bulk_new = []
    for m in range(n_mfp_bulk):
     if acc > kappa_bulk_tot*delta and acc < kappa_bulk_tot*(1.0-delta):
      mfp_bulk_new.append(mfp_bulk[m])
      kappa_bulk_new.append(kappa_bulk[m])
     acc += kappa_bulk[m]


    min_mfp = min(mfp_bulk_new)
    min_mfp = min(min_mfp,argv.setdefault('min_mfp',min_mfp))
    max_mfp = max(mfp_bulk_new)
    max_mfp = max(max_mfp,argv.setdefault('max_mfp',max_mfp))

    mfp_sampled_log = np.linspace(np.log10(min_mfp*0.99),np.log10(max_mfp*1.01),n_mfp)

    #---DISCRETIZE MFPS in LOG SPACE---------------------
    mfp_sampled = []
    for m in mfp_sampled_log:
     mfp_sampled.append(pow(10,m))
    #--------------------------------------------------------



    #MFP interpolation---------------------------
    B0 = np.zeros(n_mfp)
    B1 = np.zeros(n_mfp)
    B2 = np.zeros(n_mfp)
    kappa_tot = 0
    for m in range(len(mfp_bulk_new)):
     kappa = kappa_bulk_new[m]
     mfp = mfp_bulk_new[m]
     [m1,m2,f1,f2] = self.get_mfp_interpolation(mfp_sampled_log,np.log10(mfp))
     b0 = kappa
     b1 = kappa/mfp
     b2 = kappa/mfp/mfp
     B0[m1] += f1*b0; B0[m2] += f2*b0
     B1[m1] += f1*b1; B1[m2] += f2*b1
     B2[m1] += f1*b2; B2[m2] += f2*b2
    B0 /=np.sum(B0)
    B1 /=np.sum(B1)
    B2 /=np.sum(B2)
    #---------------------------------------------
    #Compute DOM
    #----------------
    data.update({'kappa_bulk_tot':sum(kappa_bulk_new)})
    data.update({'mfp_bulk':mfp_bulk_new})
    data.update({'kappa_bulk':kappa_bulk_new})
    
    B1 = np.tile(B1,(n_mfp,1))
  
    data.update({'B0':B0})
    data.update({'B1':B1})
    data.update({'B2':np.tile(B2,(n_mfp,1))})
    #data.update({'B1':B1})
    #data.update({'B2':B2})
    data.update({'mfp_sampled':np.array(mfp_sampled)})

  else: data=None
  self.state = MPI.COMM_WORLD.bcast(data,root=0)


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
