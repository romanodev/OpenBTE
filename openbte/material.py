import numpy as np
import os
import deepdish as dd
from mpi4py import MPI
from numpy.testing import assert_array_equal


class Material(object):

 def __init__(self,**argv):

  if MPI.COMM_WORLD.Get_rank() == 0:
 
   output = {}   
   data = self.compute_mat_2D(argv)   
   output.update({'data_2D':data})
   
   data = self.compute_mat_3D(argv)   
   output.update({'data_3D':data})
  
   #region = argv.setdefault('region','bulk')
   #dd.io.save('material_' + region + '.hdf5',output)   
   dd.io.save('material.hdf5',output)   
  MPI.COMM_WORLD.Barrier()

 def compute_mat_2D(self,argv):  
   
  if MPI.COMM_WORLD.Get_rank() == 0:
   #Polar Angle-----
   n_phi = int(argv.setdefault('n_phi',48)); Dphi = 2*np.pi/n_phi
   phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)
   #--------------------
   
   #Azimuthal Angle------------------------------
   n_theta = int(argv.setdefault('n_theta',48)); Dtheta = np.pi/n_theta/2.0
   theta = np.linspace(Dtheta/2.0,np.pi/2.0 - Dtheta/2.0,n_theta)
   dtheta = 2.0*np.sin(Dtheta/2.0)*np.sin(theta)   
   
   #Compute directions---
   polar = np.array([np.sin(phi),np.cos(phi),np.ones(n_phi)]).T
   azimuthal = np.array([np.sin(theta),np.sin(theta),np.cos(theta)]).T
   direction = np.einsum('ij,kj->ikj',azimuthal,polar)
   
   #Compute average---
   ftheta = (1-np.cos(2*theta)*np.sinc(Dtheta/np.pi))/(np.sinc(Dtheta/2/np.pi)*(1-np.cos(2*theta)))
   fphi= np.sinc(Dphi/2.0/np.pi)
   polar_ave = np.array([fphi*np.ones(n_phi),fphi*np.ones(n_phi),np.ones(n_phi)]).T
   azimuthal_ave = np.array([ftheta,ftheta,np.cos(Dtheta/2)*np.ones(n_theta)]).T   
   direction_ave = np.multiply(np.einsum('ij,kj->ikj',azimuthal_ave,polar_ave),direction)
   #---------------------------------------------------------------
   
   #Import material
   if argv.setdefault('model','nongray')=='nongray':
    tmp = np.loadtxt(os.path.dirname(__file__) + '/materials/'+ argv['matfile'])
    mfp_bulk = tmp[:,0]*1e9; 
    kappa_bulk = tmp[:,1]
   else:
    mfp_bulk = np.array(argv.setdefault('mfp',[1]),dtype='double'); 
    kappa_bulk = argv.setdefault('kappa_bulk',np.ones(len(mfp_bulk)))

   n_mfp_bulk = len(mfp_bulk) 
   mfp = np.logspace(-3,np.log10(max(mfp_bulk)),argv.setdefault('n_mfp',100)) 
   trials = np.outer(mfp_bulk,ftheta*np.sin(theta))
   J0 = np.outer(np.ones(n_mfp_bulk),dtheta)
   J1 = np.outer(kappa_bulk/mfp_bulk,ftheta*dtheta*np.sin(theta))/sum(kappa_bulk/mfp_bulk)   
   J2 = np.outer(kappa_bulk/pow(mfp_bulk,2),dtheta)/sum(kappa_bulk/pow(mfp_bulk,2))   #x2 (symmetry) and /2 (average)
   J3 = np.outer(1/mfp_bulk,ftheta*dtheta*np.sin(theta))

   G = mfp/2
   kappa_mfe = np.square(mfp)/2

   domega = np.ones(n_phi)*Dphi/2.0/np.pi
 
   data = {'kappa_bulk_tot':sum(kappa_bulk),'kappa_bulk':kappa_bulk,\
                'mfp_bulk':mfp_bulk,\
                'fphi':fphi,\
                'G':G,\
                'kappa_mfe':kappa_mfe,\
                'dphi':Dphi,\
                'n_mfp':len(mfp),\
                'trials':trials,\
                'domega':domega,\
                'mfp':mfp,\
                'J0':J0,'J1':J1,'J2':J2,'J3':J3,\
                'n_phi':n_phi,'n_theta':n_theta,\
                'direction':direction,\
                'direction_ave':direction_ave,\
                'polar':polar,\
                'azimuthal':azimuthal,\
                'polar_ave':polar,\
                'control_angle':polar,\
                'azimuthal_ave':azimuthal}
   
   return data

 def compute_mat_3D(self,argv):  
   
  if MPI.COMM_WORLD.Get_rank() == 0:
    #Polar Angle-----
    n_phi = int(argv.setdefault('n_phi',48)); Dphi = 2.0*np.pi/n_phi
    phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)
   #--------------------
   
    #Azimuthal Angle------------------------------
    n_theta = int(argv.setdefault('n_theta',48)); Dtheta = np.pi/n_theta
    theta = np.linspace(Dtheta/2.0,np.pi - Dtheta/2.0,n_theta)
    dtheta = 2.0*np.sin(Dtheta/2.0)*np.sin(theta)   
   
    #Compute directions---
    polar = np.array([np.sin(phi),np.cos(phi),np.ones(n_phi)]).T
    azimuthal = np.array([np.sin(theta),np.sin(theta),np.cos(theta)]).T
    direction = np.einsum('ij,kj->ikj',azimuthal,polar)
   
    #Compute average---
    ftheta = (1-np.cos(2*theta)*np.sinc(Dtheta/np.pi))/(np.sinc(Dtheta/2/np.pi)*(1-np.cos(2*theta)))
    fphi= np.sinc(Dphi/2.0/np.pi)
    polar_ave = np.array([fphi*np.ones(n_phi),fphi*np.ones(n_phi),np.ones(n_phi)]).T
    azimuthal_ave = np.array([ftheta,ftheta,np.cos(Dtheta/2)*np.ones(n_theta)]).T   
    direction_ave = np.multiply(np.einsum('ij,kj->ikj',azimuthal_ave,polar_ave),direction)
    #---------------------------------------------------------------
    control_angle = direction_ave.reshape((n_phi*n_theta,3))

    
    #Import material
    if argv.setdefault('model','nongray')=='nongray':
     tmp = np.loadtxt(os.path.dirname(__file__) + '/materials/'+ argv['matfile'])
     mfp_bulk = tmp[:,0]*1e9; n_mfp_bulk = len(mfp_bulk)
     kappa_bulk = tmp[:,1]
    else: 
     mfp_bulk = np.array(argv.setdefault('mfp',[1]),dtype='double'); 
     kappa_bulk = argv.setdefault('kappa_bulk',np.ones(len(mfp_bulk)))

    n_mfp_bulk = len(mfp_bulk)
    J0 = np.ones(n_mfp_bulk)
    J1 = kappa_bulk/mfp_bulk/sum(kappa_bulk/mfp_bulk)   
    J2 = kappa_bulk/pow(mfp_bulk,2)/sum(kappa_bulk/pow(mfp_bulk,2))   
    J3 = 1/mfp_bulk

    
    mfp = np.logspace(-3,np.log10(max(mfp_bulk)),argv.setdefault('n_mfp',100))
    trials = mfp_bulk
    J0 = np.array([J0]).T
    J1 = np.array([J1]).T
    J2 = np.array([J2]).T
    J3 = np.array([J3]).T
    trials = np.array([trials]).T
    
    G = mfp/2
    kappa_mfe = np.square(mfp)/3

   
    domega = []
    for t in range(n_theta):
     for p in range(n_phi):
      domega.append(dtheta[t]*Dphi/4.0/np.pi)
    domega = np.array(domega)
   


    data = {'kappa_bulk_tot':sum(kappa_bulk),'kappa_bulk':kappa_bulk,\
                'mfp_bulk':mfp_bulk,\
                'fphi':fphi,\
                'dphi':Dphi,\
                'n_mfp':len(mfp),\
                'G':G,\
                'kappa_mfe':kappa_mfe,\
                'trials':trials,\
                'domega':domega,\
                'mfp':mfp,\
                'J0':J0,'J1':J1,'J2':J2,'J3':J3,\
                'n_phi':n_phi,'n_theta':n_theta,\
                'direction':direction,\
                'control_angle':control_angle,\
                'direction_ave':direction_ave,\
                'polar':polar,\
                'azimuthal':azimuthal,\
                'polar_ave':polar,\
                'azimuthal_ave':azimuthal}
   
    return data
   

   
