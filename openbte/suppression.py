import numpy as np
from openbte.utils import *
from pubeasy import MakeFigure



def plot_suppression(**argv):


 sol = load_data('solver')
 T_sampled = sol['kappa_mode'].T
 #----------------------------------
 #Compute MFPs
 mat = load_data('material')
 mfp_sampled  = mat['mfp_sampled']*1e9
 sigma  = mat['sigma']*1e9

 vmfp  = mat['VMFP']
 [n_phi,n_theta,n_mfp]  = mat['sampling']

 Dphi = 2*np.pi/n_phi
 phi = np.linspace(Dphi/2,2.0*np.pi-Dphi/2,n_phi,endpoint=True)
 phi = list(phi)
 #mfp_vec = np.outer(vmfp[:,0],mfp_sampled)
 bmfp = np.linalg.norm(vmfp,axis=1)
 #print(np.shape(bmfp))
 #quit()
 mfp_vec = np.outer(vmfp[:,0],mfp_sampled)
 M = mfp_vec
 #-----------------------------

 S = np.zeros_like(T_sampled)
 for n in range(np.shape(T_sampled)[0]):
  for m in range(np.shape(T_sampled)[1]):
       S[n,m]             =  T_sampled[n,m]/M[n,m]*1e18
       #S[n,m]             =  T_sampled[n,m]/mfp_sampled[m]#*1e18


 m1 = 5
 l1 = 20
 l2 = int(n_phi/2)-l1
 l3 = int(n_phi/2)+l1
 l4 = int(n_phi)  -l1
 
 Sm = np.zeros(n_mfp)
 
 for n in range(np.shape(T_sampled)[0])[l1:l2]:
  for m in range(np.shape(T_sampled)[1]):
    Sm[m] += S[n,m]
 for n in range(np.shape(T_sampled)[0])[l3:l4]:
  for m in range(np.shape(T_sampled)[1]):
    Sm[m] += S[n,m]

 Sm = Sm/(n_phi-4*l1)


 if argv.setdefault('show',True):
  fig = MakeFigure()
  fig.add_plot(mfp_sampled*1e-3,Sm,color='k')
  fig.add_labels('Mean Free Path [$\mu$ m]','Suppression')
  fig.finalize(grid=True,xscale='log',write = False,show=True)

 save_data(argv.setdefault('suppression_file','suppression'),{'mfp':mfp_sampled,'suppression':Sm})

 return mfp_sampled,Sm

if __name__ == '__main__':

    main(show=True)
