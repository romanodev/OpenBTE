from mpi4py import MPI
from fourier import Fourier
from bte import BTE
from pyvtk import *
import numpy as np
import deepdish as dd
import os
import matplotlib
import matplotlib.patches as patches
from matplotlib.path import Path
from fig_maker import *
from shapely import geometry,wkt


def get_suppression(mfps,sup,mfp):


  if mfp < mfps[0]:
   output = sup[0]
  else:
   included = False
   for n in range(len(mfps)-1):
    mfp_1 = mfps[n]
    mfp_2 = mfps[n+1]
    if mfp < mfp_2 and mfp > mfp_1:
     output = sup[n] + (sup[n+1]-sup[n])/(np.log10(mfps[n+1])-np.log10(mfps[n]))*(np.log10(mfp)-np.log10(mfps[n]))
     included = True
     break

   if included == False:
    output = sup[-1]*mfps[-1]/mfp

  return output



def create_path(obj):

   codes = [Path.MOVETO]
   for n in range(len(obj)-1): 
    codes.append(Path.LINETO)
   codes.append(Path.CLOSEPOLY)

   verts = []
   for tmp in obj:
     verts.append(tmp) 
   verts.append(verts[0]) 
   
   path = Path(verts,codes)
   return path

class Plot(object):

 def __init__(self,**argv):

  if len(argv['plot'].split('/')) == 2:
   if argv['plot'].split('/')[0] == 'map':
     self.plot_map(argv)
   if argv['plot'].split('/')[0] == 'material':
     self.plot_material(argv)

  if argv['plot'] == 'suppression_function' :
   self.plot_suppression_function()

  if argv['plot'] == 'line_temperature' :
   self.plot_line_temperature(argv)

  if argv['plot'] == 'directional_suppression_function' :
   self.plot_directional_suppression_function(argv)

  if argv['plot'] == 'distr' :
   self.plot_distribution()

 def plot_map(self,argv):

  if MPI.COMM_WORLD.Get_rank() == 0:

   
   var = argv['plot'].split('/')[1]
   geo = dd.io.load('geometry.hdf5')
   Nx = argv.setdefault('n_x',1)
   Ny = argv.setdefault('n_y',1)
   Ny = argv['n_y']
   increment = [0,0]
   #data = np.ones(len(geo['elems']))
   variable = argv['plot'].split('/')[1]
   model = 'geometry'

   if variable == 'temperature':
    model = 'data'
    solver = dd.io.load('solver.hdf5')
    data = solver['bte_temperature']
    increment = [1,0]
    maxt = max(data)
    mint = min(data)
    a = 1.0; b = (float(Nx)-1.0)/float(Nx)
    data -= mint
    data /= (maxt-mint)
    data *= a-b
    data += b

   if variable == 'flux_magnitude':
    model = 'data'
    solver = dd.io.load('solver.hdf5')
    tmp = solver['bte_flux']
    data = []
    for t in tmp:
     data.append(np.linalg.norm(t))
    maxt = max(data)
    mint = min(data)
    data = (data-mint)/(maxt-mint) 

   mm =1e4
   mm2 =-1e4
   Lx = geo['size'][0]
   Ly = geo['size'][1]

   #init_plotting(presentation=True)
   for nx in range(Nx):
    for ny in range(Ny):
     Px = nx* Lx
     Py = ny * Ly
     displ = [float(nx)/float(Nx),float(ny)/float(Ny)]
     cmap = matplotlib.cm.get_cmap('Wistia')
     for e,elem in enumerate(geo['elems']):
      poly = []
      for n in elem:
       poly.append(geo['nodes'][n][0:2])
      p = np.array(poly).copy()
      for i in p:
       i[0] += Px 
       i[1] += Py 
      path = create_path(p)
      #value = data[e]#-np.dot(displ,increment)
      #if value < mm :  
      # mm = value
      #if value > mm2 :  
      # mm2 = value
      if model == 'geometry': color = 'gray'
      if model == 'data': color = color=cmap(data[e]-np.dot(displ,increment))
  
      patch = patches.PathPatch(path,linestyle=None,linewidth=0,color=color,zorder=10,alpha=0.5)
      gca().add_patch(patch) 
      gca().add_patch(patch)
  
   if variable == 'geometry':
    for ll in geo['side_list']['Periodic'] + \
              geo['side_list']['Inactive']:
     p1 = geo['nodes'][geo['sides'][ll][0]]
     p2 = geo['nodes'][geo['sides'][ll][1]]
     plot([p1[0],p2[0]],[p1[1],p2[1]],c2,lw=3,zorder=0)

    for ll in geo['side_list']['Boundary'] :
     p1 = geo['nodes'][geo['sides'][ll][0]]
     p2 = geo['nodes'][geo['sides'][ll][1]]
     plot([p1[0],p2[0]],[p1[1],p2[1]],c3,lw=3,zorder=0)
 
   axis('off')
   axis('equal')
   show() 

 def plot_material(self,argv):
  if MPI.COMM_WORLD.Get_rank() == 0:
   init_plotting()
   data = dd.io.load('material.hdf5')
   #dis_nano = list(data['B0']*data['kappa_bulk_tot'])
   dis_bulk = list(data['kappa_bulk'])
   mfp = list(np.multiply(data['mfp_bulk'],1e6))
   #mfp_nano = list(np.multiply(data['mfp_sampled'],1e6))
   fill([mfp[0]] + mfp + [mfp[-1]],[0] + dis_bulk + [0])
   
   #fill([mfp_nano[0]] + mfp_nano + [mfp_nano[-1]],[0] + dis_nano + [0],color=c2,alpha=0.3)

   xscale('log')
   #grid(which='both')
   xlabel('$\Lambda_{\mathrm{b}}$ [$\mu$ m]')
   ylabel('$K_{\mathrm{b}}d\Lambda_{\mathrm{b}}$ [Wm$^{-1}$K$^{-1}$]')
   ylim([0,max(dis_bulk)*1.3])
   show()

 def plot_line_temperature(self,argv):

  if MPI.COMM_WORLD.Get_rank() == 0:
   geo = dd.io.load('geometry.hdf5')
   solver = dd.io.load('solver.hdf5')
   temp = solver['bte_temperature']

   #Compute the closest element on the left-----
   y0 = argv.setdefault('y0',0)
   m = argv.setdefault('m',0)
   size = geo['size'][0]   
   P = [-size/2.0,-m*size/2.0+y0,0]
   ce = -1
   dmin = 1e4
   for n,elem in enumerate(geo['elems']):   
    c = geo['elem_centroids'][n]
    if np.linalg.norm(c-P)<dmin:
     dmin = np.linalg.norm(c-P)
     ce = n
   #---------------------------------------

   line = geometry.LineString([(P[0],P[1]),(P[0]+size,P[1]+m*size)])     
   kc1 = ce
   nn = 1
   while nn > 0:
    for s in geo['elem_side_map'][ce]:
     kc2 = mesh.get_neighbor_elem(kc1,s)
     
     #Create polygons-------------------
     poly = []
     for n in geo['elems'][kc2]:
      poly.append(geo['nodes'][n][0:2])
     poly.append(poly[0])      
     polygon = geometry.Polygon(poly)
     #----------------------------------
     ips = line.intersection(poly.boundary)
     print(ips)
     quit()

 

 def plot_distribution(self):
  if MPI.COMM_WORLD.Get_rank() == 0:
   init_plotting(extra_x_padding = 0.05)
   data = dd.io.load('material.hdf5')
   dis_bulk = list(data['kappa_bulk'])
   mfp = list(data['mfp_bulk'])
  
   #plot nano------ 
   solver = dd.io.load('solver.hdf5')
   mfp_sampled = data['mfp_sampled']
   sup = solver['suppression_function']
   kappa_nano = []
   mfp_nano = []
   for n in range(len(mfp)):
    sup_interp = get_suppression(mfp_sampled,sup,mfp[n])
    kappa_nano.append(dis_bulk[n]*sup_interp)
    mfp_nano.append(mfp[n]*sup_interp*1e6)

   #reordering-----
   I = np.argsort(np.array(mfp_nano))
   kappa_sorted = []
   for i in I:
    kappa_sorted.append(kappa_nano[i])
   mfp_nano = sorted(mfp_nano)

   #----------------

   fill([mfp_nano[0]] + mfp_nano + [mfp_nano[-1]],[0] + kappa_sorted + [0])


   xscale('log')
   xlabel('$\Lambda_{\mathrm{n}}$ [$\mu$ m]')
   ylabel('$K_{\mathrm{n}}d\Lambda_{\mathrm{n}}$ [Wm$^{-1}$K$^{-1}$]')
   ylim([0,max(kappa_sorted)*1.3])
   show()

  
 def plot_directional_suppression_function(self,argv):

  if MPI.COMM_WORLD.Get_rank() == 0:
   from mayavi import mlab
   #init_plotting()
   data = dd.io.load('solver.hdf5')
   sup = data['directional_suppression']
   mat = dd.io.load('material.hdf5')
   n_theta = mat['dom']['n_theta'] 
   phonon_dir = mat['dom']['phonon_dir'] 
   n_phi = mat['dom']['n_phi'] 
   n_mfp = len(mat['mfp_sampled'])

   sup_sph = np.zeros((n_mfp,n_theta,n_phi))
   for m in range(n_mfp):
    for t in range(n_theta):
     for p in range(n_phi):
      sup_sph[m][t][p] +=abs(sup[m][t][p])


   delta= 1.2
   cg = mlab.figure(bgcolor=(1,1,1), fgcolor=None, engine=None, size=(400, 250)) 
    
   m = argv.setdefault('plot_m',0)
   z =  np.zeros((n_theta,n_phi))
   y =  np.zeros((n_theta,n_phi))
   x =  np.zeros((n_theta,n_phi))
   s =  np.zeros((n_theta,n_phi))
   for t in range(n_theta):
    for p in range(n_phi):
     tmp = phonon_dir[t][p] 
     x[t][p] = tmp[0]*sup_sph[m][t][p] 
     y[t][p] = tmp[1]*sup_sph[m][t][p]
     z[t][p] = tmp[2]*sup_sph[m][t][p]
     s[t][p] = sup_sph[m][t][p]
   mlab.mesh(x,y,z,scalars = s,colormap="blue-red")

   #mlab.orientation_axes('off')
   #mlab.view(azimuth=90, elevation=180)
   cam = cg.scene.camera
   cam.zoom(1.4)
   mlab.savefig('tmp.png',size=(800,500))
   mlab.show()




 def plot_suppression_function(self):

  if MPI.COMM_WORLD.Get_rank() == 0:
   init_plotting()
   data = dd.io.load('solver.hdf5')
   sup = data['suppression_function']
   mfp = data['mfp']*1e6
   plot(mfp,sup,color=c2)
   xscale('log')
   grid('on')
   xlabel('Mean Free Path [$\mu$ m]')
   ylabel('Suppression Function')
   show()



  


