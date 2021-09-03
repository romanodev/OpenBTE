import numpy as np
import openbte.utils as utils
import matplotlib.pylab as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD

def suppression(solver,material,options_suppression)->'suppression':

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


