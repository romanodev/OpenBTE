



def kappa_mode_2DSym(material,solver,options)->'kappa_mode':

 import numpy as np
 import openbte.utils as utils
 from mpi4py import MPI
 import plotly.express as px
 import plotly.graph_objects as go
 import numpy as np
 comm = MPI.COMM_WORLD
 data = None  
  
 if comm.rank == 0:
   mfp_bulk    = material['mfp_bulk']
   r_bulk      = material['r_bulk']
   sigma_bulk  = material['sigma_bulk']
   phi_bulk    = material['phi_bulk']
   mfp_sampled = material['mfp_sampled']
   phi         = material['phi']

   mfp_nano_sampled = solver['mfp_nano_sampled'] * 1e9
   kappa_bulk = sigma_bulk[:,0]*mfp_bulk[:,0]

   n_bulk = len(mfp_bulk)

   mfp_nano = np.zeros(n_bulk)
   a1,a2,m1,m2 = utils.fast_interpolation(r_bulk,mfp_sampled,scale='linear')
   b1,b2,p1,p2 = utils.fast_interpolation(phi_bulk,phi,bound='periodic')
   np.add.at(mfp_nano,np.arange(n_bulk),a1*b1*mfp_nano_sampled[m1,p1])
   np.add.at(mfp_nano,np.arange(n_bulk),a1*b2*mfp_nano_sampled[m1,p2])
   np.add.at(mfp_nano,np.arange(n_bulk),a2*b1*mfp_nano_sampled[m2,p1])
   np.add.at(mfp_nano,np.arange(n_bulk),a2*b2*mfp_nano_sampled[m2,p2])

   kappa_nano = sigma_bulk[:,0]*mfp_nano

   data = {'kappa_nano':kappa_nano,'mfp_nano':mfp_nano,'kappa_bulk':kappa_bulk,'mfp_bulk':mfp_bulk,'f':material['f']}

   if options.setdefault('show',False):
    fig = go.Figure(data=go.Scatter(x=mfp_bulk*1e-6, y=kappa_bulk))
    fig = go.Figure(data=go.Scatter(x=mfp_nano*1e-6, y=kappa_nano))
    fig.update_xaxes(title_text="Mean Free Path [$\mu m$]")
    fig.update_yaxes(title_text="Thermal Conductivity [W/m/K]")
    fig.show()

 
 return utils.create_shared_memory_dict(data)


def kappa_mode_3D(material,solver)->'kappa_mode':

 import numpy as np
 import openbte.utils as utils
 from mpi4py import MPI
 comm = MPI.COMM_WORLD
 data = None   
 if comm.rank == 0:
   mfp_bulk    = material['mfp_bulk']
   r_bulk      = material['r_bulk']
   sigma_bulk  = material['sigma_bulk']
   phi_bulk    = material['phi_bulk']
   theta_bulk  = material['theta_bulk']
   mfp_sampled = material['mfp_sampled']
   phi         = material['phi']
   theta   = material['theta']

   mfp_nano_sampled = solver['mfp_nano_sampled'] * 1e9
   kappa_bulk = sigma_bulk[:,0]*mfp_bulk[:,0]

   n_bulk = len(mfp_bulk)
   n_phi  = len(phi)

   a1,a2,m1,m2 = utils.fast_interpolation(r_bulk,mfp_sampled,scale='linear')
   b1,b2,p1,p2 = utils.fast_interpolation(phi_bulk,phi,bound='periodic')
   c1,c2,t1,t2 = utils.fast_interpolation(theta_bulk,theta,bound='periodic')

   index_1 =  t1 * n_phi + p1; 
   index_2 =  t1 * n_phi + p2;
   index_3 =  t2 * n_phi + p1; 
   index_4 =  t2 * n_phi + p2; 

   mfp_nano = np.zeros(n_bulk)
   np.add.at(mfp_nano,np.arange(n_bulk),a1*c1*b1*mfp_nano_sampled[m1,index_1])
   np.add.at(mfp_nano,np.arange(n_bulk),a1*c1*b2*mfp_nano_sampled[m1,index_2])
   np.add.at(mfp_nano,np.arange(n_bulk),a1*c2*b1*mfp_nano_sampled[m1,index_3])
   np.add.at(mfp_nano,np.arange(n_bulk),a1*c2*b2*mfp_nano_sampled[m1,index_4])
 
   np.add.at(mfp_nano,np.arange(n_bulk),a2*c1*b1*mfp_nano_sampled[m2,index_1])
   np.add.at(mfp_nano,np.arange(n_bulk),a2*c1*b2*mfp_nano_sampled[m2,index_2])
   np.add.at(mfp_nano,np.arange(n_bulk),a2*c2*b1*mfp_nano_sampled[m2,index_3])
   np.add.at(mfp_nano,np.arange(n_bulk),a2*c2*b2*mfp_nano_sampled[m2,index_4])

   kappa_nano = sigma_bulk[:,0]*mfp_nano

   data =  {'kappa_nano':kappa_nano,'mfp_nano':mfp_nano,'kappa_bulk':kappa_bulk,'mfp_bulk':mfp_bulk,'f':material['f']}

 return utils.create_shared_memory_dict(data)
