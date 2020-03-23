import deepdish as dd
import sys
import numpy as np
import scipy
import deepdish as dd

def get_linear_indexes(mfp,value,scale,extent):

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



def generate_mfp(**argv):

 n_phi = int(argv.setdefault('n_phi',48))
 n_mfp = int(argv.setdefault('n_mfp',50))
 n_theta = int(argv.setdefault('n_theta',24))
 submodel = argv.setdefault('submodel','2DSym')
 if submodel == '2DSym' or submodel == '2D':
   nm = n_phi*n_mfp
 else :  
   nm = n_phi*n_mfp*n_theta

 data = dd.io.load('mfp.h5')
 K = data['K']
 mfp_bulk = data['mfp']
 n_mfp_bulk = len(mfp_bulk)
 mfp_sampled = np.logspace(min([-2,np.log10(min(mfp_bulk)*0.99)]),np.log10(max(mfp_bulk)*1.01),n_mfp)#min MFP = 1e-2 

 #getting coeff matrix---
 coeff = np.zeros((n_mfp,n_mfp_bulk))
 for m,rr in enumerate(mfp_bulk):
    (m1,a1,m2,a2) = get_linear_indexes(mfp_sampled,rr,scale='linear',extent=True)
    coeff[m1,m] = a1
    coeff[m2,m] = a2
 #-----------------------

 #---------------------
 #Polar Angle---------
 Dphi = 2*np.pi/n_phi
 phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)
 #--------------------
   
 #Azimuthal Angle------------------------------
 Dtheta = np.pi/n_theta/2.0
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
 #------------------------------------------------------

 n_mfp_bulk = len(mfp_bulk) 
 n_mfp = argv.setdefault('n_mfp',100)
 mfp = np.logspace(min([-2,np.log10(min(mfp_bulk)*0.99)]),np.log10(max(mfp_bulk)*1.01),n_mfp)#min MFP = 1e-2 
 mfp = np.logspace(-1,3,n_mfp)#min MFP = 1e-2 

 n_mfp = len(mfp)
 temp_coeff = np.zeros(n_mfp*n_phi) 
 kappa_directional = np.zeros((n_mfp*n_phi,3)) 
 kappa_directional_not_int = np.zeros((n_mfp*n_phi,3)) 
 norm = 0
 norm_tb = 0
   
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

 #This is everything we need---
 data = {'tc':tc,'B':Wod,'sigma':sigma,'jc':sigma,'kappa':kappa,'a':a,'ac':np.zeros(nm)}



 return data

