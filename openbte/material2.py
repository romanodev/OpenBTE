import numpy as np
import os
import pickle
from mpi4py import MPI
from numpy.testing import assert_array_equal
import math
import pickle
from matplotlib.pylab import *

class Material(object):


 def __init__(self,**argv):

  if MPI.COMM_WORLD.Get_rank() == 0:
 
   model = argv.setdefault('model','isotropic_2DSym')

   if model == 'full_2D':
     data = self.compute_full_2D(**argv)

   if model == 'full_2D_new':
     data = self.compute_full_2D_new(**argv)

   if model == 'isotropic_2DSym' or model == 'nongray':
     data = self.compute_isotropic_2DSym(**argv)
 
   if model == 'anisotropic_2DSym':
     data = self.compute_anisotropic_2DSym(**argv)

   if model == 'anisotropic_3D':
     data = self.compute_anisotropic_3D(**argv)

   if model == 'isotropic_3D':
     data = self.compute_isotropic_3D(**argv)

   data.update({'kappa_inclusion':argv.setdefault('kappa_inclusion',1e-3)})
   if argv.setdefault('save',True):
    pickle.dump(data,open(argv.setdefault('save_filename','material.p'),'wb+'))
  
  else : data = None

  self.state = MPI.COMM_WORLD.bcast(data,root=0)
    

 def compute_full_2D_new(self,**argv):

  if MPI.COMM_WORLD.Get_rank() == 0:

   data = pickle.load(open(argv['matfile'],'rb'))
   mfp = data['MFP']

   mfp_b,theta_b,phi_b = self.spherical(mfp)
   nm = len(mfp_b)
   versors = np.zeros((nm,3))
   gversors = np.zeros((nm,3))
   for i in range(nm):
    if mfp_b[i] == 0:
     versors[i] = [0,0,0]
    else: 
     versors[i] = mfp[i]/mfp_b[i]

   versors = np.array(versors)
   mfp_b *= 1e9   #from m to nm
   kbulk = data['KBULK']
   Tcoeff = data['TCOEFF']

   k_coeff = np.array(data['KCOEFF'])*1e-9
   k_coeff[:,2] = 0 #Enforce zeroflux on z (for visualization purposed)

   return {'B': data['B'],\
           'TCOEFF':Tcoeff,\
           'angle_map':np.arange(nm),\
           'temp_vec':range(nm),\
           'kappa_directional':k_coeff,\
           'n_serial':1,
           'n_parallel':nm,
           'mfp':mfp_b,
           'control_angle':versors,\
           'kappa_bulk_tot':kbulk}




 def compute_full_2D(self,**argv):

  if MPI.COMM_WORLD.Get_rank() == 0:

   data = dd.io.load(argv['matfile'])
   mfp = data['MFP']
   gmfp = data['GMFP']
   gmfp_b,gtheta_b,gphi_b = self.spherical(gmfp)

   mfp_b,theta_b,phi_b = self.spherical(mfp)
   nm = len(mfp_b)
   versors = np.zeros((nm,3))
   gversors = np.zeros((nm,3))
   for i in range(nm):
    if mfp_b[i] == 0:
     versors[i] = [0,0,0]
    else: 
     versors[i] = mfp[i]/mfp_b[i]
    if gmfp_b[i] == 0:
     gversors[i] = [0,0,0]
    else: 
     gversors[i] = gmfp[i]/gmfp_b[i]

   gversors = np.array(gversors)
   versors = np.array(versors)
   mfp_b *= 1e9   #from m to nm
   gmfp_b *= 1e9   #from m to nm

   if argv['rta'] :
    A = data['ARTA']
    kbulk = data['KBULKRTA']
    Tcoeff = data['TCOEFFRTA']
   else: 
    A = data['AFULL']
    kbulk = data['KBULKFULL']
    Tcoeff = data['TCOEFF']

   k_coeff = np.array(data['KCOEFF'])*1e-9
   k_coeff[:,2] = 0 #Enforce zeroflux on z (for visualization purposed)


   return {'CollisionMatrix':A,\
           'TCOEFF':Tcoeff,\
           'angle_map':np.arange(nm),\
           'temp_vec':range(nm),\
           'kappa_directional':k_coeff,\
           'n_serial':1,
           'n_parallel':nm,
           'mfp':mfp_b,
           'gmfp':gmfp_b,
           'control_angle':versors,\
           'gcontrol_angle':gversors,\
           'kappa_bulk_tot':kbulk}



 def get_linear_indexes(self,mfp,value,scale,extent):

   if value == 0:
     return -1,-1,-1,-1

   n = len(mfp)

   found = False
   beyond = False
   if extent:
    if value < mfp[0]:
      beyond = True
      found = True
      i=0;j=1;
    elif value > mfp[-1]:
      i=n-2;j=n-1;
      beyond = True
      found = True
   if beyond == False:
    for m in range(n-1):

      if (value <= mfp[m+1]) and (value >= mfp[m]) :
        i = m; j= m+1 
        found = True
        break
  
   if found == False:
      print('no interpolation found')
   else:  
    aj = (value-mfp[i])/(mfp[j]-mfp[i]) #OK.
    ai = 1-aj

    if scale=='inverse':
     ai *=mfp[i]/value
     aj *=mfp[j]/value

   return i,ai,j,aj  
  
 def spherical(self,xyz): 

    #ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    r = np.sqrt(xy + xyz[:,2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    phi = np.arctan2(xyz[:,0], xyz[:,1])

    return r,theta,phi


 def compute_anisotropic_2DSym(self,**argv):  
   
  if MPI.COMM_WORLD.Get_rank() == 0:

   #Polar Angle-----
   n_phi = int(argv.setdefault('n_phi',48)); Dphi = 2*np.pi/n_phi
   phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)

   #Azimuthal Angle------------------------------
   n_theta = int(argv.setdefault('n_theta',48)); Dtheta = np.pi/n_theta/2.0
   theta = np.linspace(Dtheta/2.0,np.pi/2.0 - Dtheta/2.0,n_theta)
   dtheta = 2.0*np.sin(Dtheta/2.0)*np.sin(theta)   
   domega = np.outer(dtheta,Dphi * np.ones(n_phi))

   #Compute directions---
   polar = np.array([np.sin(phi),np.cos(phi),np.ones(n_phi)]).T
   azimuthal = np.array([np.sin(theta),np.sin(theta),np.cos(theta)]).T
   direction = np.einsum('ij,kj->ikj',azimuthal,polar)

   #Compute average---
   ftheta = (1-np.cos(2*theta)*np.sinc(Dtheta/np.pi))/(np.sinc(Dtheta/2/np.pi)*(1-np.cos(2*theta)))
   fphi= np.sinc(Dphi/2.0/np.pi)
   polar_ave = polar*fphi
   azimuthal_ave = np.array([ftheta*np.sin(theta),ftheta*np.sin(theta),np.cos(Dtheta/2)*np.ones(n_theta)]).T   
   direction_ave = np.einsum('ij,kj->ikj',azimuthal_ave,polar_ave)

   #import material--------
   n_mfp = argv.setdefault('n_mfp',100)
   data = np.loadtxt(argv['matfile'],skiprows=1)
   C = data[:,0] #J/m^3/K
   tau = data[:,1] #s
   v = data[:,2:5] #m/s
   nm = len(v)

   gmfp = np.einsum('i,ij->ij',tau,v)
   mfp_b,theta_b,phi_b = self.spherical(gmfp)
   kappa_tot = np.sum(tau*C*v[:,0]*v[:,0])
   mfp_b *=1e9 #nm

   #mfp_bulk = np.array([np.linalg.norm(m) for m in gmfp]) #nm
   mfp = np.logspace(-2,np.log10(max(mfp_b)*1.01),n_mfp) 
   #------------------------
   kappa_directional = np.zeros((n_phi*n_mfp,3))
   temp_coeff = np.zeros(n_mfp*n_phi)
   n = 0
   for m,t,p,mp in zip(mfp_b,theta_b,phi_b,gmfp):
     (m1,am1,m2,am2) = self.get_linear_indexes(mfp,m*np.sin(t),scale='linear',extent=True)
     if not m1 == -1:
      if p < 0: p = 2*np.pi + p
      (p1,ap1,p2,ap2) = self.get_linear_indexes(phi,p,scale='linear',extent=True)
      if not p1 == -1:
        index_1 = p1 * n_mfp + m1
        index_2 = p1 * n_mfp + m2
        index_3 = p2 * n_mfp + m1
        index_4 = p2 * n_mfp + m2

        factor = v[n] * C[n]*1e-9 #from m to nm
        kappa_directional[index_1] += ap1 * am1 * factor
        kappa_directional[index_2] += ap1 * am2 * factor
        kappa_directional[index_3] += ap2 * am1 * factor
        kappa_directional[index_4] += ap2 * am2 * factor

        factor = C[n]/tau[n]
        temp_coeff[index_1] += ap1 * am1 * factor
        temp_coeff[index_2] += ap1 * am2 * factor
        temp_coeff[index_3] += ap2 * am1 * factor
        temp_coeff[index_4] += ap2 * am2 * factor

     n+=1

   A = np.array([temp_coeff/np.sum(temp_coeff)])
   temp_coeff = temp_coeff/np.sum(temp_coeff)
   temp_vec = np.zeros(n_mfp*n_phi,dtype=int)  
   angle_map = np.arange(n_phi)
   angle_map = np.repeat(angle_map,n_mfp)
   mfp = np.tile(mfp,n_phi)  
   polar_ave = np.array([np.repeat(polar_ave[:,i],n_mfp) for i in range(3)]).T
   kappa_directional[:,2] = 0 #Enforce zeroflux on z (for visualization purposed)
   #-----------------------


   
   return {'CollisionMatrix':A,\
           'TCOEFF':temp_coeff,\
           'temp_vec':temp_vec,\
           'angle_map':angle_map,\
           'kappa_directional':kappa_directional,\
           'n_serial':n_mfp,\
           'n_parallel':n_phi,\
           'mfp':mfp,\
           'control_angle':polar_ave,\
           'kappa_bulk_tot':np.sum(kappa_tot)}


 def compute_anisotropic_3D(self,**argv): 

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
   domega = np.outer(dtheta, Dphi * np.ones(n_phi))

   
   #direction_int = []
   n_mfp = argv.setdefault('n_mfp',100)
   control_angle = []
   for t in range(n_theta): 
    for p in range(n_phi): 
     for m in range(n_mfp):
      control_angle.append(direction_ave[t,p])

   
   #import material--------
   n_mfp = argv.setdefault('n_mfp',100)
   data = np.loadtxt(argv['matfile'],skiprows=1)
   C = data[:,0] #J/m^3/K
   tau = data[:,1] #s
   v = data[:,2:5] #m/s
   nm = len(v)

   #----Put values here------
   gmfp = np.einsum('i,ij->ij',tau,v)

   mfp_bulk,theta_b,phi_b = self.spherical(gmfp)
   mfp_bulk *=1e9 #nm
   

   kappa_tot = np.sum(tau*C*v[:,0]*v[:,0])
   n_mfp_bulk = len(mfp_bulk) 
   mfp = np.logspace(np.log10(min(mfp_bulk[mfp_bulk > 0])*0.99),np.log10(max(mfp_bulk)*1.01),n_mfp) 

   #------------------------
   temp_coeff = np.zeros(n_mfp*n_phi*n_theta) 
   kappa_directional = np.zeros((n_mfp*n_phi*n_theta,3)) 

   n = 0
   for m,t,p,mp in zip(mfp_bulk,theta_b,phi_b,gmfp):
    if m > 0:   
     (m1,am1,m2,am2) = self.get_linear_indexes(mfp,m,scale='linear',extent=True)
     if not m1 == -1:
      if p < 0: p = 2*np.pi + p
      (p1,ap1,p2,ap2) = self.get_linear_indexes(phi,p,scale='linear',extent=True)

      (t1,at1,t2,at2) = self.get_linear_indexes(theta,t,scale='linear',extent=True) #check Theta

      if (not p1 == -1) and (not t1 == -1):
        index_1 = t1 * n_phi * n_mfp + p1 * n_mfp + m1
        index_2 = t1 * n_phi * n_mfp + p1 * n_mfp + m2
        index_3 = t1 * n_phi * n_mfp + p2 * n_mfp + m1
        index_4 = t1 * n_phi * n_mfp + p2 * n_mfp + m2
        index_5 = t2 * n_phi * n_mfp + p1 * n_mfp + m1
        index_6 = t2 * n_phi * n_mfp + p1 * n_mfp + m2
        index_7 = t2 * n_phi * n_mfp + p2 * n_mfp + m1
        index_8 = t2 * n_phi * n_mfp + p2 * n_mfp + m2


        factor = v[n] * C[n]*1e-9 #from m to nm
        kappa_directional[index_1] += at1 * ap1 * am1 * factor
        kappa_directional[index_2] += at1 * ap1 * am2 * factor
        kappa_directional[index_3] += at1 * ap2 * am1 * factor
        kappa_directional[index_4] += at1 * ap2 * am2 * factor
        kappa_directional[index_5] += at2 * ap1 * am1 * factor
        kappa_directional[index_6] += at2 * ap1 * am2 * factor
        kappa_directional[index_7] += at2 * ap2 * am1 * factor
        kappa_directional[index_8] += at2 * ap2 * am2 * factor

        factor = C[n]/tau[n]
        temp_coeff[index_1] += at1 * ap1 * am1 * factor
        temp_coeff[index_2] += at1 * ap1 * am2 * factor
        temp_coeff[index_3] += at1 * ap2 * am1 * factor
        temp_coeff[index_4] += at1 * ap2 * am2 * factor
        temp_coeff[index_5] += at2 * ap1 * am1 * factor
        temp_coeff[index_6] += at2 * ap1 * am2 * factor
        temp_coeff[index_7] += at2 * ap2 * am1 * factor
        temp_coeff[index_8] += at2 * ap2 * am2 * factor

    n+=1


   A = np.array([temp_coeff/np.sum(temp_coeff)])
   temp_coeff = temp_coeff/np.sum(temp_coeff)
   temp_vec = np.zeros(n_mfp*n_phi*n_theta,dtype=int)   
   angle_map = np.arange(n_phi*n_theta)
   angle_map = np.repeat(angle_map,n_mfp)
   mfp = np.tile(mfp,n_phi*n_theta)  



   return {'CollisionMatrix':A,\
           'TCOEFF':temp_coeff,\
           'temp_vec':temp_vec,\
           'angle_map':angle_map,\
           'kappa_directional':kappa_directional,\
           'n_serial':n_mfp,\
           'n_parallel':n_phi*n_theta,\
           'mfp':mfp,\
           'control_angle':control_angle,\
           'kappa_bulk_tot':np.sum(kappa_tot)}



 def compute_isotropic_3D(self,**argv): 

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
   domega = np.outer(dtheta, Dphi * np.ones(n_phi))

   
   #direction_int = []
   n_mfp = argv.setdefault('n_mfp',100)
   control_angle = []
   for t in range(n_theta): 
    for p in range(n_phi): 
     for m in range(n_mfp):
      control_angle.append(direction_ave[t,p,:])

   #Import material-------------------------------------------------------------------
   if 'filename' in argv.keys():
    source = argv.setdefault('source',os.path.dirname(__file__) + '/materials/')
    tmp = np.loadtxt(source + argv['filename'],skiprows=1)
    mfp_bulk = tmp[:,0]*1e9; 
    kappa_bulk = tmp[:,1]
   else:
    mfp_bulk = argv['mfp_bulk']
    kappa_bulk = argv['kappa_bulk']


   n_mfp_bulk = len(mfp_bulk) 
   #mfp = np.logspace(np.log10(min(mfp_bulk)*0.99),np.log10(max(mfp_bulk)*1.01),n_mfp) 
   mfp = np.logspace(np.log10(min(mfp_bulk)*0.99),np.log10(max(mfp_bulk)*1.01),n_mfp) 
   #mfp = np.logspace(-3,np.log10(max(mfp_bulk)*1.01),n_mfp) 

   n_mfp = len(mfp)
   temp_coeff = np.zeros(n_mfp*n_phi*n_theta) 
   kappa_directional = np.zeros((n_mfp*n_phi*n_theta,3)) 
   norm = 0

   norm_tb = 0

   for m in range(n_mfp_bulk):
    (m1,a1,m2,a2) = self.get_linear_indexes(mfp,mfp_bulk[m],scale='linear',extent=True)
    norm_tb += kappa_bulk[m]/mfp_bulk[m]*3/4
    for t in range(n_theta): 
     for p in range(n_phi): 
      index_1 = t * n_mfp * n_phi + p * n_mfp + m1
      index_2 = t * n_mfp * n_phi + p * n_mfp + m2
      factor = kappa_bulk[m]/4.0/np.pi*domega[t,p]
          
      kappa_directional[index_1] += 3 * a1 * factor/mfp_bulk[m]*direction_ave[t,p]
      kappa_directional[index_2] += 3 * a2 * factor/mfp_bulk[m]*direction_ave[t,p]
      temp_coeff[index_1] += a1 * kappa_bulk[m]/mfp_bulk[m]/mfp_bulk[m]*domega[t,p]
      temp_coeff[index_2] += a2 * kappa_bulk[m]/mfp_bulk[m]/mfp_bulk[m]*domega[t,p]

      norm += kappa_bulk[m]/mfp_bulk[m]/mfp_bulk[m]*domega[t,p]

   #replicate bulk values---
   mfp = np.tile(mfp,n_phi*n_theta)  
   #-----------------------
   A = np.array([temp_coeff/norm])

   TCOEFF = temp_coeff/norm
   temp_vec = np.zeros(n_mfp*n_phi*n_theta,dtype=int)   

   angle_map = np.arange(n_phi*n_theta)
   angle_map = np.repeat(angle_map,n_mfp)


   return {'CollisionMatrix':A,\
           'TCOEFF':TCOEFF,\
           'temp_vec':temp_vec,\
           'angle_map':angle_map,\
           'kappa_directional':kappa_directional,\
           'norm_tb':norm_tb,\
           'n_serial':n_mfp,\
           'n_parallel':n_phi*n_theta,\
           'mfp':mfp,\
           'control_angle':control_angle,\
           'kappa_bulk_tot':np.sum(kappa_bulk)}



 def compute_isotropic_2DSym(self,**argv):  
   
   #Polar Angle-----
   n_phi = int(argv.setdefault('n_phi',48)); Dphi = 2*np.pi/n_phi
   phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)
   #--------------------
   
   #Azimuthal Angle------------------------------
   n_theta = int(argv.setdefault('n_theta',48)); Dtheta = np.pi/n_theta/2.0
   theta = np.linspace(Dtheta/2.0,np.pi/2.0 - Dtheta/2.0,n_theta)
   dtheta = 2.0*np.sin(Dtheta/2.0)*np.sin(theta)   
   domega = np.outer(dtheta,Dphi * np.ones(n_phi))

   #Compute directions---
   polar = np.array([np.sin(phi),np.cos(phi),np.zeros(n_phi)]).T
   azimuthal = np.array([np.sin(theta),np.sin(theta),np.zeros(n_theta)]).T
   direction = np.einsum('ij,kj->ikj',azimuthal,polar)
 
   #Compute average---
   ftheta = (1-np.cos(2*theta)*np.sinc(Dtheta/np.pi))/(np.sinc(Dtheta/2/np.pi)*(1-np.cos(2*theta)))
   fphi= np.sinc(Dphi/2.0/np.pi)
   polar_ave = polar*fphi
   azimuthal_ave = np.array([ftheta*np.sin(theta),ftheta*np.sin(theta),np.zeros(n_theta)]).T   
   direction_ave = np.einsum('ij,kj->ikj',azimuthal_ave,polar_ave)

   direction_int = np.einsum('ijl,ij->ijl',direction_ave,domega)

   #Import material-------------------------------------------------------------------
   if 'filename' in argv.keys():
    source = argv.setdefault('source',os.path.dirname(__file__) + '/materials/')
    tmp = np.loadtxt(source + argv['filename'],skiprows=1)
    mfp_bulk = tmp[:,0]*1e9; 
    kappa_bulk = tmp[:,1]
   else:
    mfp_bulk = argv['mfp_bulk']
    kappa_bulk = argv['kappa_bulk']
   #------------------------------------------------------

   n_mfp_bulk = len(mfp_bulk) 
   n_mfp = argv.setdefault('n_mfp',100)
   mfp = np.logspace(min([-2,np.log10(min(mfp_bulk)*0.99)]),np.log10(max(mfp_bulk)*1.01),n_mfp)#min MFP = 1e-2 
   mfp = np.logspace(-1,3,n_mfp)#min MFP = 1e-2 
   #mfp = np.logspace(np.log10(min(mfp_bulk)*0.99),np.log10(max(mfp_bulk)*1.01),n_mfp) 


   n_mfp = len(mfp)
   temp_coeff = np.zeros(n_mfp*n_phi) 
   kappa_directional = np.zeros((n_mfp*n_phi,3)) 
   kappa_directional_not_int = np.zeros((n_mfp*n_phi,3)) 
   norm = 0
   norm_tb = 0
   
   test = 0
   for p in range(n_phi): 
    for t in range(n_theta): 
     for m in range(n_mfp_bulk):
      reduced_mfp = mfp_bulk[m]*ftheta[t]*np.sin(theta[t])
      (m1,a1,m2,a2) = self.get_linear_indexes(mfp,reduced_mfp,scale='linear',extent=True)
      index_1 = p*n_mfp + m1
      index_2 = p*n_mfp + m2
      factor = kappa_bulk[m]/4.0/np.pi*2.0*domega[t,p]
      if mfp_bulk[m] > 0:
       kappa_directional[index_1]         += 3 * a1 * 2 * kappa_bulk[m]/mfp_bulk[m]/4.0/np.pi * direction_ave[t,p]*domega[t,p]#*mfp[m1]/mfp_bulk[m]
       kappa_directional[index_2]         += 3 * a2 * 2 * kappa_bulk[m]/mfp_bulk[m]/4.0/np.pi * direction_ave[t,p]*domega[t,p]#*mfp[m2]/mfp_bulk[m]
       temp_coeff[index_1] += a1 * kappa_bulk[m]/mfp_bulk[m]/mfp_bulk[m]*domega[t,p]#*mfp[m1]/mfp_bulk[m]
       temp_coeff[index_2] += a2 * kappa_bulk[m]/mfp_bulk[m]/mfp_bulk[m]*domega[t,p]#*mfp[m2]/mfp_bulk[m]
       norm += a1* kappa_bulk[m]/mfp_bulk[m]/mfp_bulk[m]*domega[t,p]#*mfp[m1]/mfp_bulk[m]
       norm += a2* kappa_bulk[m]/mfp_bulk[m]/mfp_bulk[m]*domega[t,p]#*mfp[m2]/mfp_bulk[m]



   #replicate bulk values---
   mfp = np.tile(mfp,n_phi)  
   #-----------------------
   #A = np.array([temp_coeff/norm])
   A = np.zeros_like([temp_coeff])


   TCOEFF = temp_coeff/norm
   temp_vec = np.zeros(n_mfp*n_phi,dtype=int)   

   angle_map = np.arange(n_phi)
   angle_map = np.repeat(angle_map,n_mfp)

   #repeat angle---

   kappa_directional[:,2] = 0 #Enforce zeroflux on z (for visualization purposed)


   polar_ave = np.array([np.repeat(polar_ave[:,i],n_mfp) for i in range(3)]).T
   polar = np.array([np.repeat(polar[:,i],n_mfp) for i in range(3)]).T
   #---------------------------------

   return {'B':A,\
           'TCOEFF':TCOEFF,\
           'temp_vec':temp_vec,\
           'angle_map':angle_map,\
           'dphi':Dphi,\
           'ftheta':ftheta*np.sin(theta),\
           'kappa_directional':kappa_directional,\
           'n_serial':n_mfp,\
           'n_parallel':n_phi,\
           'mfp':mfp,\
           'control_angle':polar_ave,\
           'angle':polar,\
           'direction_int':direction_int,\
           'factor':[np.sin(Dphi/2),np.cos(Dphi/2)/Dphi],\
           'mfp_bulk':mfp_bulk,\
           'kappa_bulk':kappa_bulk,\
           'kappa_bulk_tot':np.sum(kappa_bulk)}



