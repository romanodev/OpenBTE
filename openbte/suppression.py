import numpy as np
from openbte.utils import *



def write_suppression(**argv):


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
 bmfp = np.linalg.norm(vmfp,axis=1)
 mfp_vec = np.outer(vmfp[:,0],mfp_sampled)
 M = mfp_vec
 #-----------------------------

 S = T_sampled/M*1e18
 if argv.setdefault('fourier',False):
  Sf = T_sampled_f/M*1e18
  Smf = np.mean(Sf,axis=0)
 Sm = np.mean(S,axis=0)
 
 #if argv.setdefault('show',True):
 # fig = MakeFigure()
 # fig.add_plot(mfp_sampled*1e-3,Sm,color='k',name='bte')
 # if argv.setdefault('fourier',False):
 #  fig.add_plot(mfp_sampled*1e-3,Smf,color='g',name='fourier')
 # fig.add_labels('Mean Free Path [$\mu$ m]','Suppression')
 #fig.finalize(grid=True,xscale='log',write = False,show=True,ylim=[0,1])

 #if argv.setdefault('mode_resolved',False):
 
 #    nm = n_mfp
 #    phis = range(n_phi)

 #    for p in phis:
 #        fig = MakeFigure()
 #        fig.add_plot(mfp_sampled[:nm]*1e-3,S[p,:nm],color='k',marker='o')
 #        fig.add_plot(mfp_sampled[:nm]*1e-3,Sf[p,:nm],color='r',marker='o')
 #        fig.finalize(grid=True,xscale='log',write = False,show=True)
    

 save_data(argv.setdefault('suppression_file','suppression'),{'mfp':mfp_sampled,'suppression':Sm})

 return mfp_sampled,Sm

if __name__ == '__main__':

    main(show=True)
