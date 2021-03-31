import numpy as np
from openbte.utils import *
from pubeasy import MakeFigure



def plot_suppression(**argv):


 sol = load_data('solver')
 T_sampled = sol['kappa_mode'].T

 if argv.setdefault('fourier',False):
  T_sampled_f = sol['kappa_mode_f'].T

 if argv.setdefault('zero',False):
  T_sampled_0 = sol['kappa_0'].T


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
 if argv.setdefault('fourier',False):
  Sf = np.zeros_like(T_sampled_f)
 if argv.setdefault('zero',False):
  S0 = np.zeros_like(T_sampled_0)
 for n in range(np.shape(T_sampled)[0]):
  for m in range(np.shape(T_sampled)[1]):
       S[n,m]             =  T_sampled[n,m]/M[n,m]*1e18
       if argv.setdefault('fourier',False):
        Sf[n,m]           =  T_sampled_f[n,m]/M[n,m]*1e18
       if argv.setdefault('zero',False):
        S0[n,m]           =  T_sampled_0[n,m]/M[n,m]*1e18



 m1 = 0
 l1 = 0
 l2 = int(n_phi/2)-l1
 l3 = int(n_phi/2)+l1
 l4 = int(n_phi)  -l1
 
 Sm = np.zeros(n_mfp)



 for n in range(np.shape(T_sampled_f)[0])[l1:l2]:
  for m in range(np.shape(T_sampled_f)[1]):
    Sm[m] += S[n,m]
 for n in range(np.shape(T_sampled_f)[0])[l3:l4]:
  for m in range(np.shape(T_sampled_f)[1]):
    Sm[m] += S[n,m]
 Sm = Sm/(n_phi-4*l1)

 if argv.setdefault('fourier',False):
  Smf = np.zeros(n_mfp)
  for n in range(np.shape(T_sampled_f)[0])[l1:l2]:
   for m in range(np.shape(T_sampled_f)[1]):
    Smf[m] += Sf[n,m]
  for n in range(np.shape(T_sampled_f)[0])[l3:l4]:
   for m in range(np.shape(T_sampled_f)[1]):
    Smf[m] += Sf[n,m]
  Smf = Smf/(n_phi-4*l1)

 if argv.setdefault('zero',False):
  Sm0 = np.zeros(n_mfp)
  for n in range(np.shape(T_sampled_0)[0])[l1:l2]:
   for m in range(np.shape(T_sampled_0)[1]):
    Sm0[m] += S0[n,m]
  for n in range(np.shape(T_sampled_0)[0])[l3:l4]:
   for m in range(np.shape(T_sampled_0)[1]):
    Sm0[m] += S0[n,m]
  Sm0 = Sm0/(n_phi-4*l1)


 if argv.setdefault('show',True):
  fig = MakeFigure()
  fig.add_plot(mfp_sampled*1e-3,Sm,color='k',name='bte')
  fig.add_plot(mfp_sampled*1e-3,Smf,color='g',name='fourier')
  fig.add_labels('Mean Free Path [$\mu$ m]','Suppression')
  fig.finalize(grid=True,xscale='log',write = False,show=True,ylim=[0,1])

 if argv.setdefault('mode_resolved',False):
 
     nm = 40
     fig = MakeFigure()
     #phis = range(n_phi)
     phis = [12]
     for p in phis:
      #fig.add_plot(mfp_sampled[:nm]*1e-3,T_sampled[p,:nm],color='k')
      #fig.add_plot(mfp_sampled[:nm]*1e-3,T_sampled_f[p,:nm],color='r')
      fig.add_plot(mfp_sampled*1e-3,S[p],color='k')
      fig.add_plot(mfp_sampled*1e-3,Sf[p],color='r')
      fig.add_plot(mfp_sampled*1e-3,S0[p],color='g')
    
     fig.finalize(grid=True,xscale='log',write = False,show=True)


 save_data(argv.setdefault('suppression_file','suppression'),{'mfp':mfp_sampled,'suppression':Sm})

 return mfp_sampled,Sm

if __name__ == '__main__':

    main(show=True)
