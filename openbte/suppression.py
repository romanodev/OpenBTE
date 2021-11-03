import numpy as np
import openbte.utils as utils
import matplotlib.pylab as plt
from mpi4py import MPI
import mpl_toolkits.mplot3d.axes3d as axes3d

comm = MPI.COMM_WORLD


def suppression_3D(solver,material,options_suppression)->'suppression':

 data = None
 if comm.rank == 0:

  #m1 = options_suppression.setdefault('m',0)
  #l1 = options_suppression.setdefault('l',3)

  phi = material['phi']
  mfp_sampled = material['mfp_sampled']
  theta = material['theta']
  n_mfp = len(mfp_sampled)
  n_phi = len(phi)
  n_theta = len(theta)
  
  ##--------------- 
  #l2 = int(n_phi/2)-l1
  #l3 = int(n_phi/2)+l1
  #l4 = int(n_phi)  -l1

  #Mesh--

  #-----------
  mfp_0 = 1e-10
  mfp   = np.log10(mfp_sampled)
  mfp  -= np.min(mfp)
  mfp  +=1
  #------------

  #R, P = np.meshgrid(mfp, phi)
  #X, Y = R*np.cos(P+np.pi/2), R*np.sin(P+np.pi/2)
  #----------
  mfp_nano = solver['mfp_nano_sampled'].reshape((n_mfp,n_theta,n_phi))


  S = np.divide(mfp_nano,material['F'][:,:,:,0],\
          out=np.zeros_like(mfp_nano), where=material['F'][:,:,:,0]!=0)*1e9


  #S = np.divide(mfp_nano,np.linalg.norm(material['F'][:,:,:],axis=3),\
  #        out=np.zeros_like(mfp_nano), where=material['F'][:,:,:,0]!=0)*1e9


  #S = np.log10(S.clip(min=1e-5))
  #print(S[30,24,:])
  #fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
  #ax.plot(phi, S[10,24,:])
  #plt.show()
  #quit()

  #S -= np.min(S)
  #S += 1

  THETA, PHI = np.meshgrid(theta, phi)

  #R = np.absolute(mfp_nano[20].T)*1e18
  R = np.absolute(S[20].T)
  #R = np.ones_like(THETA)

  X = R * np.sin(PHI) * np.sin(THETA)
  Y = R * np.cos(PHI) * np.sin(THETA)
  Z = R * np.cos(THETA)

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1, projection='3d')
  plot = ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
    linewidth=0, antialiased=False, alpha=0.5)

  v = [0.2,0.2,0.01]
  #ax.set_xlim(-150, 150) 
  #ax.set_ylim(-150, 150) 
  #ax.set_zlim(-150, 150) 
  ax.set_xlim(-v[0], v[0]) 
  ax.set_ylim(-v[1], v[1]) 
  ax.set_zlim(-v[2], v[2]) 

  plt.show()



  print(S.shape)
  quit()

  Sm = np.mean(S,axis=0)

  k = 0.5484670176525239

  plt.plot(Sm)
  plt.plot([0,len(Sm)],[k,k])

  plt.show()
  #print(Sm.shape)
  quit()

    
  #Sm = 0.5*(np.mean(S[l1:l2,m1:] + S[l3:l4,m1:],axis=0))
  #Sm = 0.5*(np.mean(S[l1:l2,m1:] + S[l3:l4,m1:],axis=0))

  #Cut unnecessary data
  S[0 :l1] = 0
  S[l2:l3] = 0
  S[l4:  ] = 0
  S[:,:m1] = 0
  
  data = {'S':S,'X':X,'Y':Y,'mfp':mfp_sampled,'Sm':Sm}


def suppression_2DSym(solver,material,options_suppression)->'suppression':

 data = None
 if comm.rank == 0:

  m1 = options_suppression.setdefault('m',0)
  l1 = options_suppression.setdefault('l',3)

  phi = material['phi']
  mfp_sampled = material['mfp_sampled']
  n_mfp = len(mfp_sampled)
  n_phi = len(phi)

  #--------------- 
  l2 = int(n_phi/2)-l1
  l3 = int(n_phi/2)+l1
  l4 = int(n_phi)  -l1

  #Mesh--

  #-----------
  mfp_0 = 1e-10
  mfp   = np.log10(mfp_sampled)
  mfp  -= np.min(mfp)
  mfp  +=1
  #------------

  R, P = np.meshgrid(mfp, phi)
  X, Y = R*np.cos(P+np.pi/2), R*np.sin(P+np.pi/2)
  #----------
  mfp_nano = solver['mfp_nano_sampled']

  S = np.divide(mfp_nano,material['F'][:,:,0],\
          out=np.zeros_like(mfp_nano), where=material['F'][:,:,0]!=0)*1e9
  S = S.reshape((n_mfp,n_phi)).T

  Sm = 0.5*(np.mean(S[l1:l2,m1:] + S[l3:l4,m1:],axis=0))

  #Cut unnecessary data
  S[0 :l1] = 0
  S[l2:l3] = 0
  S[l4:  ] = 0
  S[:,:m1] = 0


  data = {'S':S,'X':X,'Y':Y,'mfp':mfp_sampled,'Sm':Sm}

 return utils.create_shared_memory_dict(data)


def plot_suppression(suppression):

  from pubeasy import MakeFigure

  Sm = suppression['Sm']
  mfp = suppression['mfp']
  fig = MakeFigure()
  fig.add_plot(mfp,Sm,color='k')
  fig.add_labels('MFP [nm]','Suppression')
  fig.finalize(grid=True,xscale='log',write = True,show=True)
 

def plot_angular_suppression(suppression):

  S  = suppression['S']
  X  = suppression['X']
  Y  = suppression['Y']
  axs = plt.subplot(111,projection='3d')
  axs.set_xlabel('X')
  axs.set_ylabel('Y')
  axs.plot_surface(X, Y, S,antialiased=True,cmap='viridis', edgecolor='none',vmin=0,vmax=0.5)
  axs.set_zlim([0,1])
  plt.show() 


