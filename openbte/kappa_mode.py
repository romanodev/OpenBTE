import numpy as np
from openbte.utils import *

def write_mode_kappa(**argv):
 #-----------------------
 mat = load_data('material')
 data = load_data(argv.setdefault('rta_material_file','rta'))
 mfp_0 = 1e-9
 #I = np.where(np.linalg.norm(mfp_bulk,axis=1) > mfp_0)[0]
 I = np.arange(len(data['tau']))
 tau = data['tau'][I]
 v = data['v'][I]
 C = data['C'][I]
 f = data['f'][I]
 #f = np.divide(np.ones_like(tau),tau, out=np.zeros_like(tau), where=tau!=0)
 kappam = np.einsum('u,u,u,u->u',tau,C,v[:,0],v[:,0]) 
 Jc = np.einsum('k,ki->ki',C,v[:,:2])
 nm = len(C)
 mfp_bulk = np.einsum('ki,k->ki',v,tau)
 r = np.linalg.norm(mfp_bulk[:,:2],axis=1)
 phi_bulk = np.array([np.arctan2(m[0],m[1]) for m in mfp_bulk[:,:2]])
 phi_bulk[np.where(phi_bulk < 0) ] = 2*np.pi + phi_bulk[np.where(phi_bulk <0)]
 kappa = data['kappa']
 #---------------------------

 #Compute MFPs
 mat = load_data('material')
 mfp_sampled  = mat['mfp_sampled']*1e9
 [n_phi,n_theta,n_mfp]  = mat['sampling']
 vmfp  = mat['VMFP']
 sigma  = mat['sigma']
 sigma = sigma[:,:,0].T
 Dphi = 2*np.pi/n_phi
 phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)
 phi = list(phi)
 #--------------------------------

 tmp = load_data('geometry')
 dirr = int(tmp['meta'][-1])

 #--------


 #Compute kappa
 sol = load_data('solver')
 T_sampled = sol['kappa_mode']

 kappa_fourier = sol['kappa_fourier']

 #Interpolation to get kappa mode
 ratio = kappa_fourier/kappa[0,0]

 T_mode = np.zeros_like(kappam)
 for m in range(nm):

     (m1,a1,m2,a2) = interpolate(mfp_sampled,r[m]*1e9,bounds='extent')
     (p1,b1,p2,b2) = interpolate(phi,phi_bulk[m],bounds='periodic',period=2*np.pi)

     T_mode[m] += T_sampled[m1,p1]*a1*b1
     T_mode[m] += T_sampled[m1,p2]*a1*b2
     T_mode[m] += T_sampled[m2,p1]*a2*b1
     T_mode[m] += T_sampled[m2,p2]*a2*b2

 T_mode *=1e9

 kappa_nano = np.einsum('i,i,i->i',C,v[:,0],T_mode)
 kappa_bulk = np.einsum('i,i,i->i',C,v[:,0],mfp_bulk[:,0])
 kappa_nano_ori = np.einsum('ij,ji',sigma,T_sampled)*1e9

 print('Check:')
 print('kappa (bulk):  ' + str(round(np.sum(kappa_bulk),2)) + ' W/m/K')
 print('kappa (mode-sampled):  ' + str(round(np.sum(kappa_nano),2)) + ' W/m/K')
 print('kappa (mfp-sampled): ' + str(round(kappa_nano_ori,2)) + ' W/m/K')


 mfp_nano = mfp_bulk.copy()


 mfp_nano[:,dirr] = T_mode

 #if argv.setdefault('show',True):
 # fig = MakeFigure()
 # fig.add_plot(f*1e-12,mfp_nano[:,dirr]*1e6,model='scatter',color='r')
 # fig.add_plot(f*1e-12,mfp_bulk[:,dirr]*1e6,model='scatter',color='b')
 # fig.add_labels('Frequency [THz]','Mean Free Path [$\mu$m]')
 # fig.finalize(grid=True,yscale='log',write = True,show=True,ylim=[1e-4,1e2])

 data = {'kappa_nano':kappa_nano,'mfp_nano':mfp_nano[:,dirr],'kappa_fourier':kappa_fourier,'mfp_bulk':mfp_bulk[:,dirr],'f':f,'kappa_bulk':kappa_bulk}
 if argv.setdefault('save',True):
     save_data(argv.setdefault('kappa_mode_file','kappa_mode'),data)

 return data


if __name__ == "__main__":

 main()
