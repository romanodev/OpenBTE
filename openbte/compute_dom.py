import numpy as np
import os


   
   
def compute_dom_2d(argv):
    
   #Polar Angle-----
   n_phi = int(argv.setdefault('n_phi',48)); Dphi = 2.0*np.pi/n_phi
   phi = np.linspace(0.0,2.0*np.pi,n_phi,endpoint=False)
   phi += argv.setdefault('polar_offset',0.0)*np.pi/180.0
   phi = phi %(2.0*np.pi)
   fphi= np.sinc(Dphi/2.0/np.pi)
   dphi_ave = 1/n_phi*np.ones(n_phi)
   #--------------------
   
   #Azimuthal Angle------------------------------
   n_theta = int(argv.setdefault('n_theta',24)); 
   Dtheta = np.pi/n_theta/2.0
   theta = np.linspace(Dtheta/2.0,np.pi/2.0 - Dtheta/2.0,n_theta)
   Dtheta_int = 2.0*np.sin(Dtheta/2.0)*np.sin(theta)
   
   
   ftheta = np.sin(theta)*(1-np.cos(2*theta)*np.sinc(Dtheta/np.pi))/(np.sinc(Dtheta/2/np.pi)*(1-np.cos(2*theta)))
   dtheta_ave = np.sin(Dtheta/2.0)*np.sin(theta) 
   dtheta = 2.0*np.sin(Dtheta/2.0)*np.sin(theta)
   #---------------------------------------------
   
   #phonon directions-------
   polar_dir = np.array([np.sin(phi),np.cos(phi),np.zeros(n_phi)]).T
   
   
   output = {'n_phi':n_phi,\
             'dtheta':dtheta,\
             'n_theta':n_theta,\
             'dphi_ave':dphi_ave,\
             'dphi':Dphi,\
             'dtheta_ave':dtheta_ave,\
             'polar_dir':polar_dir,\
             'fphi':fphi,\
             'ftheta':ftheta}
   
   return output



#def compute_dom_3d(argv) :

   #print(argv.setdefault('n_theta',16))


#   n_theta = int(argv.setdefault('n_theta',16))
#   n_phi = int(argv.setdefault('n_phi',48))
   #n_phi = int(argv['n_phi'])


   #apply_symmetry = False
#   norm_dom = 4.0*np.pi
#   output = {}
#   Dtheta = np.pi/n_theta
#   d_theta_plain = Dtheta*np.ones(n_theta)
#   theta_vec = np.linspace(Dtheta/2.0,np.pi - Dtheta/2.0,n_theta)
 #  output.update({'theta_vec':theta_vec})
 #  output.update({'n_theta':n_theta})
   #---------------------------------------

   #Phi ----------------------------------
 #  Dphi = 2.0*np.pi/n_phi
 #  d_phi_plain = Dphi*np.ones(n_phi)
 #  d_phi_int = Dphi*np.ones(n_phi)
 #  output.update({'dphi':Dphi})
   
 #  output.update({'fphi':2.0*np.sin(Dphi/2.0)/Dphi})   #
   

   #phi_vec = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True) #this is the correct one

 #  phi_vec = np.linspace(0.0,2.0*np.pi,n_phi,endpoint=False)
 #  phi_vec += argv.setdefault('polar_offset',0.0)*np.pi/180.0
 #  phi_vec = phi_vec %(2.0*np.pi)


 #  output.update({'phi_vec':phi_vec})
 #  output.update({'d_phi_vec':d_phi_plain})
 #  output.update({'n_phi':n_phi})
 #  output.update({'n_theta':n_theta})
 #  output.update({'d_phi':Dphi})
   #-----------------------------------------------------------------

 #  d_theta_int = np.zeros(n_theta)
 #  ss = np.zeros((n_theta,n_phi,3,3))
#   d_theta_vec = np.zeros(n_theta)
 #  phonon_dir = np.zeros((n_theta,n_phi,3))
 #  integrated_dir = np.zeros((n_theta,n_phi,3))
 #  phi_dir = np.zeros((n_phi,3))
 #  polar_dir = np.zeros((n_phi,3))

 #  at = np.zeros(n_theta)
 #  tt = np.zeros(n_theta)
 #  for t in range(n_theta):
 #   theta = theta_vec[t]

  #  int_sin_theta = 2.0*np.sin(d_theta_plain[t]/2.0)*np.sin(theta)
  #  int_sin_theta_2 = 1.0/2.0*(d_theta_plain[t]-np.cos(2.0*theta)*np.sin(d_theta_plain[t]))
  #  int_sin_theta_3 = 1.0/6.0*(9.0*np.sin(theta)*np.sin(d_theta_plain[t]/2.0)-np.sin(3.0*theta)*np.sin(3.0/2.0*d_theta_plain[t])   )
  #  int_cos_theta_2_sin_theta = 1.0/6.0*(3.0*np.sin(theta)*np.sin(d_theta_plain[t]/2.0)+\
   #                                       np.sin(3.0*theta)*np.sin(3.0/2.0*d_theta_plain[t]))
  #  int_sin_theta_2_cos_theta = 1.0/3.0*(pow(np.sin(1.0/2.0*(d_theta_plain[t]-2.0*theta)),3.0) +\
   #                                      (pow(np.sin(theta+d_theta_plain[t]/2.0),3)))
   # int_cos_theta_sin_theta = np.sin(d_theta_plain[t])*np.sin(theta)*np.cos(theta)
#
#    d_theta_int[t] = int_sin_theta
 #   at[t] = int_sin_theta_2
    
    
  #  tt[t]=np.sin(theta)*(1-np.cos(2*theta)*np.sinc(d_theta_plain[t]/np.pi))/(np.sinc(d_theta_plain[t]/2/np.pi)*(1-np.cos(2*theta)))



   # for p in range(n_phi):

    # phi = phi_vec[p]
     #------------------------------------------------------
    # int_cos_phi = 2.0*np.cos(phi)*np.sin(d_phi_plain[p]/2.0)
    # int_sin_phi = 2.0*np.sin(phi)*np.sin(d_phi_plain[p]/2.0)
    # int_sin_phi_2 = 1.0/2.0*(d_phi_plain[p]-np.cos(2.0*phi)*np.sin(d_phi_plain[p]))
    # int_cos_phi_2 = 1.0/2.0*(np.cos(2.0*phi)*np.sin(d_phi_plain[p])+d_phi_plain[p])
    # int_cos_phi_sin_phi = np.sin(phi)*np.cos(phi)*np.sin(d_phi_plain[p])
     #-------------------------------------------

     
    # x = sin(theta) * sin(phi)
    # y = sin(theta) * cos(phi)
    # z = cos(theta)
    # phonon_dir[t][p] = np.array([x,y,z])
    # polar_dir[p] = [round(sin(phi),8),round(cos(phi),8),0.0]

    # phi_dir[p] = [round(int_sin_phi,8),round(int_cos_phi,8),Dphi]
    # integrated_dir[t][p][0] = phi_dir[p][0]*at[t]
    # integrated_dir[t][p][1] = phi_dir[p][1]*at[t]
    # integrated_dir[t][p][2] = phi_dir[p][2]*int_cos_theta_sin_theta


    # ss[t][p][0][0] = int_sin_theta_3*int_sin_phi_2
    # ss[t][p][0][1] = int_sin_theta_3*int_cos_phi_sin_phi
    # ss[t][p][1][0] = ss[t][p][0][1]
    # ss[t][p][0][2] = int_sin_theta_2_cos_theta * int_sin_phi
    # ss[t][p][2][0] = ss[t][p][0][2]
    # ss[t][p][1][2] = int_sin_theta_2_cos_theta * int_cos_phi
    # ss[t][p][2][1] = ss[t][p][1][2]
    # ss[t][p][1][1] = int_sin_theta_3*int_cos_phi_2
    # ss[t][p][2][2] = int_cos_theta_2_sin_theta*d_phi_plain[p]



   #d_omega = np.outer(d_theta_int,d_phi_int)

   #ss2 = np.zeros((n_theta,n_phi,3,3))

   #tmp = np.zeros((3,3))
   #tmp2 = 0.0
   #for t in range(n_theta):
   # for p in range(n_phi):
   #  ss2[t,p] =  np.outer(integrated_dir[t,p],integrated_dir[t,p])/d_omega[t,p]
   #  tmp += ss2[t][p]
   #  if np.dot(phonon_dir[t][p],[1,0,0]) >= 0:
   #   tmp2 +=integrated_dir[t,p][0]


   #correction = 4.0*tmp2/np.trace(tmp)

   #print(correction)
   #quit()
   #for t in range(n_theta):
#    for p in range(n_phi):#
      #print(np.cross(integrated_dir[t][p][0:2],phonon_dir[t][p][0:2]))


   #output.update({'d_omega':d_omega})
   #output.update({'correction':correction})
   #output.update({'d_theta_vec':d_theta_int})
   #output.update({'ss':ss})
   #output.update({'ss2':ss2})
   #output.update({'n_theta':n_theta})
   #output.update({'n_phi':n_phi})
   #output.update({'phonon_dir':phonon_dir})
   #output.update({'integrated_dir':integrated_dir})
   #output.update({'S':integrated_dir})
   #output.update({'s':phonon_dir})
   #output.update({'norm_dom':norm_dom})
   #output.update({'phi_dir':phi_dir})
   #output.update({'polar_dir':polar_dir})
   #output.update({'at':at})


   #return output
