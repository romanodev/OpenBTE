import numpy as np
from openbte.objects import Material,MaterialRTA
from icosphere import icosphere


def compute_polar(mfp_bulk):
     """Covert from real to angular space"""
     phi_bulk = np.array([np.arctan2(m[0],m[1]) for m in mfp_bulk])
     phi_bulk[np.where(phi_bulk < 0) ] = 2*np.pi + phi_bulk[np.where(phi_bulk <0)]
     r = np.linalg.norm(mfp_bulk[:,:2],axis=1) #absolute values of the projection
     return r,phi_bulk 

def compute_spherical(mfp_bulk):
 """Covert from real to spherical space"""
 r = np.linalg.norm(mfp_bulk,axis=1) #absolute values of the projection
 phi_bulk = np.array([np.arctan2(m[0],m[1]) for m in mfp_bulk])
 phi_bulk[np.where(phi_bulk < 0) ] = 2*np.pi + phi_bulk[np.where(phi_bulk <0)]
 theta_bulk = np.array([np.arccos((m/r[k])[2]) for k,m in enumerate(mfp_bulk)])

 return r,phi_bulk,theta_bulk




def fast_interpolation(fine,coarse,bound=False,scale='linear') :

 if scale == 'log':
   fine    = np.log10(fine)
   coarse  = np.log10(coarse)
 #--------------

 m2 = np.argmax(coarse >= fine[:,np.newaxis],axis=1)
 m1 = m2-1
 a2 = (fine-coarse[m1])/(coarse[m2]-coarse[m1])
 a1 = 1-a2

 if bound == 'periodic':
  Delta = coarse[-1]-coarse[-2]  
  a = np.where(m2==0)[0] #this can be either regions
  m1[a] = len(coarse) -1
  m2[a] = 0
  fine[fine < Delta/2] += 2*np.pi 
  a2[a] = (fine[a] - coarse[-1])/ Delta
  a1 = 1-a2


 if bound == 'extent':

   #Small values
   al = np.where(fine<coarse[0])[0] 
   m2[al] = 1; 
   m1[al] = 0;
   a2[al] = (fine[al]-coarse[0])/ (coarse[1]-coarse[0])
   a1[al] = 1-a2[al]


   #Large values
   ar = np.where(fine>coarse[-1])[0]
   m2[ar] = len(coarse)-1; 
   m1[ar] = len(coarse)-2;
   a2[ar] = (fine[ar]-coarse[-2])/ (coarse[-1]-coarse[-2])
   a1[ar] = 1-a2[ar]

 return a1,a2,m1,m2

def RTA3D(data : MaterialRTA,**kwargs)->Material:

    #Parse options
    n_phi   = kwargs.setdefault('n_phi',24)
    n_mfp   = kwargs.setdefault('n_mfp',30)
    n_theta = kwargs.setdefault('n_theta',24)
    #----------------

    n_angles = n_phi * n_theta
    #Angular discretization
    Dphi = 2.0*np.pi/n_phi
    #phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)
    phi = np.linspace(0,2.0*np.pi,n_phi,endpoint=False)
    Dtheta = np.pi/n_theta
    theta = np.linspace(Dtheta/2,np.pi-Dtheta/2,n_theta,endpoint=True)
    #theta = np.linspace(0,np.pi,n_theta,endpoint=True)


    polar = np.array([np.sin(phi),np.cos(phi),np.ones(n_phi)]).T
    azimuthal = np.array([np.sin(theta),np.sin(theta),np.cos(theta)]).T
    direction_original = np.einsum('lj,kj->lkj',azimuthal,polar)
    direction = direction_original.reshape((n_theta * n_phi,3))
    #-------------------

    #Build mode-resolved quantities
    tau      = data.scattering_time
    sigma    = np.einsum('k,ki->ki',data.heat_capacity,data.group_velocity)
    mfp_bulk = np.einsum('ki,k->ki',data.group_velocity,tau) #here we have the 2D one
    f        = np.divide(np.ones_like(tau),tau, out=np.zeros_like(tau), where=tau!=0)
    Wdiag    = data.heat_capacity*f


    #Filtering--
    r_bulk = np.linalg.norm(mfp_bulk,axis=1)
    I = np.where(r_bulk>1e-10)
    mfp_bulk = mfp_bulk[I]
    Wdiag = Wdiag[I]
    sigma = sigma[I]
    #------------------------

    r_bulk,phi_bulk,theta_bulk = compute_spherical(mfp_bulk)

    #Sampling
    mfp_max = np.max(r_bulk)*1.1
    mfp_min = np.min(r_bulk)*0.9
    mfp_sampled = np.logspace(np.log10(mfp_min)*1.01,np.log10(mfp_max),n_mfp,endpoint=True)

    Wdiag_sampled = np.zeros((n_mfp,n_angles))
    sigma_sampled = np.zeros((n_mfp,n_angles,3)) 

    a1,a2,m1,m2 = fast_interpolation(r_bulk,mfp_sampled,bound='extent')
    b1,b2,p1,p2 = fast_interpolation(phi_bulk,phi,bound='periodic')
    c1,c2,t1,t2 = fast_interpolation(theta_bulk,theta,bound='extent')

    index_1 =  t1 * n_phi + p1; u1 = b1*c1
    index_2 =  t1 * n_phi + p2; u2 = b2*c1
    index_3 =  t2 * n_phi + p1; u3 = b1*c2
    index_4 =  t2 * n_phi + p2; u4 = b2*c2

    #Temperature coefficients
    np.add.at(Wdiag_sampled,(m1,index_1),a1*u1*Wdiag)
    np.add.at(Wdiag_sampled,(m1,index_2),a1*u2*Wdiag)
    np.add.at(Wdiag_sampled,(m2,index_1),a2*u1*Wdiag)
    np.add.at(Wdiag_sampled,(m2,index_2),a2*u2*Wdiag)
    np.add.at(Wdiag_sampled,(m1,index_3),a1*u3*Wdiag)
    np.add.at(Wdiag_sampled,(m1,index_4),a1*u4*Wdiag)
    np.add.at(Wdiag_sampled,(m2,index_3),a2*u3*Wdiag)
    np.add.at(Wdiag_sampled,(m2,index_4),a2*u4*Wdiag)

    #Interpolate flux----
    np.add.at(sigma_sampled,(m1, index_1),sigma*(a1*u1)[:,np.newaxis])
    np.add.at(sigma_sampled,(m1, index_2),sigma*(a1*u2)[:,np.newaxis])
    np.add.at(sigma_sampled,(m2, index_1),sigma*(a2*u1)[:,np.newaxis])
    np.add.at(sigma_sampled,(m2, index_2),sigma*(a2*u2)[:,np.newaxis])
    np.add.at(sigma_sampled,(m1, index_3),sigma*(a1*u3)[:,np.newaxis])
    np.add.at(sigma_sampled,(m1, index_4),sigma*(a1*u4)[:,np.newaxis])
    np.add.at(sigma_sampled,(m2, index_3),sigma*(a2*u3)[:,np.newaxis])
    np.add.at(sigma_sampled,(m2, index_4),sigma*(a2*u4)[:,np.newaxis])
    #-----------------------

    #Compute kappa sampled
    Wdiag_inv = np.divide(1, Wdiag_sampled, out=np.zeros_like(Wdiag_sampled), where=Wdiag_sampled!=0)
    kappa_sampled = np.einsum('mqi,mq,mqj->ij',sigma_sampled,Wdiag_inv,sigma_sampled)
    t_coeff = Wdiag_sampled/np.sum(Wdiag_sampled)

    return Material(kappa_sampled,sigma_sampled*1e-9,direction,t_coeff,mfp_sampled*1e9,n_angles,n_mfp,1e20)



def RTA2DSym(data : MaterialRTA,**kwargs)->Material:
    """Interpolation from q-space to vectorial MFP-space"""

    n_phi = kwargs.setdefault('n_phi',48)
    n_mfp = kwargs.setdefault('n_mfp',50)
    nq = n_phi * n_mfp

    #Polar angle---
    Dphi = 2*np.pi/n_phi
    #phi = np.linspace(Dphi/2,2.0*np.pi-Dphi/2,n_phi,endpoint=True)
    phi = np.linspace(0,2.0*np.pi,n_phi,endpoint=False)
    polar_ave = np.array([np.sin(phi),np.cos(phi)]).T
    #----------------

    #Build mode-resolved quantities
    tau      = data.scattering_time
    sigma    = np.einsum('k,ki->ki',data.heat_capacity,data.group_velocity[:,:2])
    mfp_bulk = np.einsum('ki,k->ki',data.group_velocity[:,:2],tau) #here we have the 2D one
    f        = np.divide(np.ones_like(tau),tau, out=np.zeros_like(tau), where=tau!=0)
    Wdiag    = data.heat_capacity*f

    #Convert into polar space
    r_bulk,phi_bulk = compute_polar(mfp_bulk)

    #Filtering small MFPs out (1e-10m)
    I = np.where(r_bulk>1e-9)
    r_bulk = r_bulk[I]
    mfp_bulk = mfp_bulk[I]
    phi_bulk = phi_bulk[I]
    Wdiag = Wdiag[I]
    sigma = sigma[I]

    #Sampling
    mfp_max     = kwargs.setdefault('mfp_max',np.max(r_bulk)*1.1)
    mfp_min     = kwargs.setdefault('mfp_min',np.min(r_bulk)*0.9)
    mfp_sampled = np.logspace(np.log10(mfp_min)*1.01,np.log10(mfp_max),n_mfp,endpoint=True)
    #-----------------

    Wdiag_sampled = np.zeros((n_mfp,n_phi))
    sigma_sampled = np.zeros((n_mfp,n_phi,2)) 
    #Interpolation in the MFPs
    a1,a2,m1,m2 = fast_interpolation(r_bulk,mfp_sampled,bound='extent')
    #Interpolation in phi---
    b1,b2,p1,p2 = fast_interpolation(phi_bulk,phi,bound='periodic')
    
    np.add.at(Wdiag_sampled,(m1, p1),a1*b1*Wdiag)
    np.add.at(Wdiag_sampled,(m1, p2),a1*b2*Wdiag)
    np.add.at(Wdiag_sampled,(m2, p1),a2*b1*Wdiag)
    np.add.at(Wdiag_sampled,(m2, p2),a2*b2*Wdiag)

    np.add.at(sigma_sampled,(m1, p1),sigma*(a1*b1)[:,np.newaxis])
    np.add.at(sigma_sampled,(m1, p2),sigma*(a1*b2)[:,np.newaxis])
    np.add.at(sigma_sampled,(m2, p1),sigma*(a2*b1)[:,np.newaxis])
    np.add.at(sigma_sampled,(m2, p2),sigma*(a2*b2)[:,np.newaxis])

    #Compute kappa sample
    Wdiag_inv = np.divide(1, Wdiag_sampled, out=np.zeros_like(Wdiag_sampled), where=Wdiag_sampled!=0)
    kappa_sampled     = np.einsum('mqi,mq,mqj->ij',sigma_sampled,Wdiag_inv,sigma_sampled)
    kappa_sampled_tot = np.einsum('mqi,mq,mqj->mqij',sigma_sampled,Wdiag_inv,sigma_sampled)
    t_coeff = Wdiag_sampled/np.sum(Wdiag_sampled)


    #test--
    #kappa = np.einsum('u,u,u->',Wdiag,mfp_bulk[:,0].clip(min=0),mfp_bulk[:,0].clip(min=0))
    #kappa = np.einsum('u,u,u->',Wdiag,np.absolute(mfp_bulk[:,0]),np.absolute(mfp_bulk[:,0]))


    #Boundary resistance--
    h = 2/3*np.sum(Wdiag*mfp_bulk[:,0].clip(min=0))*1e-9 #from W/m^2/K to W/m/nm/K

    #Heat source ratio
    coeff = 1/np.sum(Wdiag)

    return Material(kappa_sampled,sigma_sampled*1e-9,polar_ave,t_coeff,mfp_sampled*1e9,n_phi,n_mfp,h,coeff*1e18,kappa_sampled_tot)



def Gray3DEqui(**kwargs):

     nu      = kwargs['nu']  
     kappa   = kwargs.setdefault('kappa',1)*np.eye(3)
     MFPs    = np.array([kwargs.setdefault('MFP',10)])
     
     vertices, faces = icosphere(nu)
     vv = vertices[faces]
     npts= vv.shape[0]
     s = np.zeros((npts,3))
     for n,v in enumerate(vv):
       tmp    = np.mean(v,axis=0)
       s[n] = tmp/np.linalg.norm(tmp)

     n_angles = len(s)

     t_coeff = np.expand_dims(1/n_angles*np.ones(n_angles),axis=0)

     sigma = 3*kappa[0,0]*s/n_angles

     sigma   = np.expand_dims(sigma,axis=0)

     return Material(kappa,sigma,s,t_coeff,MFPs,n_angles,1,1e20)


def Gray2D(**kwargs):

    n_phi = kwargs.setdefault('n_phi',48)
    MFP   = kwargs['MFP'] #in nm
    Dphi = 2*np.pi/n_phi
    phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)
    #phi = np.linspace(0,2.0*np.pi,n_phi,endpoint=False)


    polar = np.array([np.sin(phi),np.cos(phi)]).T
    fphi= np.sinc(Dphi/2.0/np.pi)
    polar_ave = polar#*fphi
    kappa = kwargs.setdefault('kappa',1)*np.eye(2)

    sigma = np.zeros((1,n_phi,2))
    sigma[0] =2*kappa[0,0]/MFP * polar_ave/n_phi

    n_angles = n_phi
    mfps = np.array([MFP])

    t_coeff = np.ones((1,n_phi))/n_phi
    
    #Boundary resistance---
    h = 4/np.pi*kappa[0,0]/mfps[0]

    #Heat source coeff
    heat_source_coeff = MFP*MFP*0.5/kappa[0,0] 

    return Material(kappa,sigma,polar_ave,t_coeff,mfps,n_angles,1,h,heat_source_coeff)

def Gray3D(**kwargs):

    n_phi   = kwargs.setdefault('n_phi',24)
    MFPs    = np.array([kwargs.setdefault('MFP',10)])
    n_theta = kwargs.setdefault('n_theta',24)
    kappa   = kwargs.setdefault('kappa',1)*np.eye(3)

    n_angles = n_phi*n_theta

    #Polar Angle-----
    Dphi = 2.0*np.pi/n_phi
    phi = np.linspace(0,2.0*np.pi,n_phi,endpoint=False)

    #--------------------

    #Azimuthal Angle-----
    Dtheta = np.pi/n_theta
    theta = np.linspace(Dtheta/2.0,np.pi - Dtheta/2.0,n_theta)
    dtheta = 2.0*np.sin(Dtheta/2.0)*np.sin(theta)

    #Directions---
    polar = np.array([np.sin(phi),np.cos(phi),np.ones(n_phi)]).T
    azimuthal = np.array([np.sin(theta),np.sin(theta),np.cos(theta)]).T
    direction = np.einsum('ij,kj->ikj',azimuthal,polar)

    #Integrated directions---
    ftheta = (1-np.cos(2*theta)*np.sinc(Dtheta/np.pi))/(np.sinc(Dtheta/2/np.pi)*(1-np.cos(2*theta)))
    fphi= np.sinc(Dphi/2.0/np.pi)
    polar_ave = np.array([fphi*np.ones(n_phi),fphi*np.ones(n_phi),np.ones(n_phi)]).T
    azimuthal_ave = np.array([ftheta,ftheta,np.cos(Dtheta/2)*np.ones(n_theta)]).T
    direction_ave = np.multiply(np.einsum('ij,kj->ikj',azimuthal_ave,polar_ave),direction)
    domega = np.outer(dtheta, Dphi * np.ones(n_phi))#.reshape(n_angles)

    direction_ave_flatten = np.zeros((n_angles,3))
    direction_flatten     = np.zeros((n_angles,3))
    domega_flatten        = np.zeros(n_angles)
    sigma_flatten         = np.zeros((n_angles,3))

    total = np.zeros((3,3))
    for t in range(n_theta):
     for p in range(n_phi):
      direction_ave_flatten[n_phi*t+p] = direction_ave[t,p]
      direction_flatten[n_phi*t+p] = direction[t,p]
      total += 3*np.outer(direction[t,p],direction[t,p])*domega[t,p]/4/np.pi
      sigma_flatten[n_phi*t+p] = 3*kappa[0,0]*direction_ave[t,p]*domega[t,p]/4/np.pi/MFPs[0]
      domega_flatten[n_phi*t+p] = domega[t,p]

    t_coeff = np.expand_dims(domega_flatten/np.pi/4,axis=0)
    sigma   = np.expand_dims(sigma_flatten,axis=0)

    #print(direction_ave_flatten[12*24])
    return Material(kappa,sigma,\
                         direction_ave_flatten,\
                         t_coeff,MFPs,\
                         n_angles,1,1e20)


def RTA2DMode(data : MaterialRTA,**kwargs)->Material:
    """Provide q-resolved material data"""

    #Build mode-resolved quantities
    tau      = data.scattering_time
    sigma    = np.einsum('k,ki->ki',data.heat_capacity,data.group_velocity[:,:2])
    mfp_bulk = np.einsum('ki,k->ki',data.group_velocity[:,:2],tau) #here we have the 2D one
    f        = np.divide(np.ones_like(tau),tau, out=np.zeros_like(tau), where=tau!=0)
    Wdiag    = data.heat_capacity*f

    #Filtering very low MFPs--
    r_bulk   =  np.linalg.norm(mfp_bulk,axis=1)
    I = np.where(r_bulk>1e-10)
    mfp_bulk = mfp_bulk[I]
    Wdiag = Wdiag[I]
    sigma = sigma[I]
    nq = len(mfp_bulk)
    #-----------------------

    #Compute kappa sample
    Wdiag_inv = np.divide(1, Wdiag, out=np.zeros_like(Wdiag), where=Wdiag!=0)
    kappa     = np.einsum('qi,q,qj->ij',sigma,Wdiag_inv,sigma)
    t_coeff   = Wdiag/np.sum(Wdiag)

    #Boundary resistance--
    h = 2/3*np.sum(Wdiag*mfp_bulk[:,0].clip(min=0))*1e-9 #from W/m^2/K to W/m/nm/K

    #Heat source ratio
    coeff = 1/np.sum(Wdiag)


    return Material(kappa,np.array([sigma*1e-9]),mfp_bulk*1e9,np.array([t_coeff]),np.array([1]),nq,1,h,coeff*1e18)



