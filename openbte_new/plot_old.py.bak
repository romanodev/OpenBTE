from mpi4py import MPI
from solver import Solver
from pyvtk import *
import numpy as np
import deepdish as dd
import os
import matplotlib
import matplotlib.patches as patches
from matplotlib.path import Path
from fig_maker import *
from shapely import geometry,wkt
from shapely.geometry import MultiPoint,Point,Polygon,LineString
from scipy.interpolate import BarycentricInterpolator

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

  if '/' in argv['variable']:
   model= argv['variable'].split('/')[0]
   if model == 'map':
    self.plot_map(argv)

  if argv['variable'] == 'suppression_function' :
   self.plot_suppression_function(argv)

  if argv['variable'] == 'distribution' :
   self.plot_distribution(argv)


  #if len(argv['plot'].split('/')) == 2:
  # if argv['plot'].split('/')[0] == 'map':
   #if argv['plot'].split('/')[0] == 'material':
   #  self.plot_material(argv)
   #if argv['plot'].split('/')[0] == 'line_plot':
   #  self.plot_line_temperature(argv)
     #self.plot_material(argv)
  #if argv['plot'] == 'line_temperature' :
  # self.plot_line_temperature(argv)
  #if argv['plot'] == 'directional_suppression_function' :
  # self.plot_directional_suppression_function(argv)


 def plot_map(self,argv):

  if MPI.COMM_WORLD.Get_rank() == 0:
   
   variable = argv['variable'].split('/')[1]
   geo = dd.io.load('geometry.hdf5')
   Nx = argv.setdefault('repeat_x',3)
   Ny = argv.setdefault('repeat_y',3)

   increment = [0,0]
   #data = np.ones(len(geo['elems']))
   solver = dd.io.load('solver.hdf5')
  
   if not variable == 'geometry': 
    data = solver[variable]

    if len(np.shape(data)) == 1: #It is temperature
      increment = [1,0]
      a = 1.0; b = (float(Nx)-1.0)/float(Nx)
      data *= a-b
      data += b
    else: #it is temperature
      increment = [0,0]


    #In case we have flux
    if variable == 'bte_flux' or variable == 'fourier_flux':
     component = argv['variable'].split('/')[2]
     if component == 'x':   
       data = data[0]
     if component == 'y':   
       data = data[1]
     if component == 'z':   
       data = data[2]
     if component == 'magnitude':   
      tmp = []
      for d in data :
       tmp.append(np.linalg.norm(d))
      data = tmp
     #-------------------------

    #normalization----
    maxt = max(data)
    mint = min(data)
    data -= mint
    data /= (maxt-mint)


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
      if variable == 'geometry': color = 'gray'
      if variable == 'data': color = color=cmap(data[e]-np.dot(displ,increment))
      #color = color=cmap(data[e]-np.dot(displ,increment))
  
      patch = patches.PathPatch(path,linestyle=None,linewidth=0,color=color,zorder=10,alpha=1.0)
      gca().add_patch(patch) 
      gca().add_patch(patch)
  
   #if variable == 'geometry':
   # for ll in geo['side_list']['Periodic'] + \
   #           geo['side_list']['Inactive']:
   #  p1 = geo['nodes'][geo['sides'][ll][0]]
   #  p2 = geo['nodes'][geo['sides'][ll][1]]
   #  plot([p1[0],p2[0]],[p1[1],p2[1]],c2,lw=3,zorder=0)

   # for ll in geo['side_list']['Boundary'] :
   #  p1 = geo['nodes'][geo['sides'][ll][0]]
   #  p2 = geo['nodes'][geo['sides'][ll][1]]
   #  plot([p1[0],p2[0]],[p1[1],p2[1]],c3,lw=3,zorder=0)
 
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
   data = solver[argv['plot'].split('/')[1]+'_nodal']['data']

  #['data']
   N = argv.setdefault('N',100)

   #Compute the closest element on the left-----
   y0 = argv.setdefault('y0',0)
   m = argv.setdefault('m',0)
   Lx = geo['size'][0]   
   Ly = geo['size'][1]   
   p1 = Point(-Lx,-m*Lx+y0)
   p2 = Point(Lx,m*Lx+y0)
   poly = Polygon([(-Lx/2.0,-Ly/2.0),(Lx/2.0,-Ly/2.0),(Lx/2.0,Ly/2.0),(-Lx/2.0,Ly/2.0)])
   l =  LineString([p1, p2])
   isp = l.intersection(poly.boundary)
   P1 = np.array([list(isp)[0].x,list(isp)[0].y])
   P2 = np.array([list(isp)[1].x,list(isp)[1].y])


   elems = range(len(geo['elems']))
   line_data = np.zeros(N)
   x = []
   x_old = 0.0
  
   P_old = P1 
   for k in range(N):
    tmp = P1 + (1e-4 + float(k)/float(N-1))*(P2-P1) 
    dx = np.linalg.norm(tmp-P_old)
    x.append(x_old + dx)
    P_old = tmp
    x_old = x[-1]

    p = Point(tmp[0],tmp[1])
    for e in elems:   
     elem = geo['elems'][e]
     nodes = geo['nodes'][elem][:,0:2]
     polygon = geometry.Polygon(nodes)
     if p.within(polygon):
      #Barycentric interpolation------
      area = geo['elem_volumes'][e]
      v = 0
      for n in range(3):
       n1 = (n+1)%3    
       n2 = (n+2)%3   
       new_nodes = [[p.x,p.y],list(nodes[n1,:]),list(nodes[n2,:])]
       partial_area = self.compute_area(new_nodes)
       coeff = partial_area/area
       v += coeff * data[elem[n]]
      break

    line_data[k] = v

   init_plotting(extra_x_padding = 0.05)
   plot(x,line_data,color=c1)  
   grid(which='both')
   xlabel('Distance [nm]')
   
   yl = solver[argv['plot'].split('/')[1]+'_nodal']['label']
   ylabel(yl)
   show()   
  



 def compute_area(self,points):

   x = []; y= []
   for p in points:
    x.append(p[0])
    y.append(p[1])
   return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
 

 def plot_distribution(self,argv):
  if MPI.COMM_WORLD.Get_rank() == 0:
   init_plotting(extra_x_padding = 0.05)
   data = dd.io.load('material.hdf5')
   dis_bulk = list(data['kappa_bulk'])
   mfp = list(data['mfp_bulk']) #m
 
   #plot nano------ 
   solver = dd.io.load(argv.setdefault('filename','solver.hdf5'))
   mfp_sampled = data['mfp_sampled'] #m
   suppression = solver['suppression']
   sup = [suppression[m,:,:].sum() for m in range(np.shape(suppression)[0])]
   kappa_nano = []
   mfp_nano = []
   for n in range(len(mfp)):
    sup_interp = get_suppression(mfp_sampled,sup,mfp[n])
    kappa_nano.append(dis_bulk[n]*sup_interp)
    mfp_nano.append(mfp[n]*sup_interp*1e9)

   #reordering-----
   I = np.argsort(np.array(mfp_nano))
   kappa_sorted = []
   for i in I:
    kappa_sorted.append(kappa_nano[i])
   mfp_nano = sorted(mfp_nano)

   #----------------
   mfp_bulk = np.array(mfp)*1e9
   

   #fill([mfp_bulk[0]] + list(mfp_bulk) + [mfp_bulk[-1]+1e-6],[0] + dis_bulk + [0],lw=2,alpha=0.6)
   fill([mfp_nano[0]] + mfp_nano + [mfp_nano[-1]],[0] + kappa_sorted + [0],color=c2)
   
   f = open('mfp_nano.dat','w+')
   for n in range(len(mfp)):
    f.write('{0:10.4E} {1:20.4E}'.format(mfp_nano[n]*1e-9,kappa_sorted[n]))
    f.write('\n')
   f.close()

   xscale('log')
   xlabel('$\Lambda$ [nm]')
   ylabel('$Kd\Lambda$ [Wm$^{-1}$K$^{-1}$]')
   #ylim([0,0.75])
   grid()
   #xlim([5e-1,4e1]) 
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


 def plot_suppression_function(self,argv):

  if MPI.COMM_WORLD.Get_rank() == 0:
   init_plotting(extra_bottom_padding= 0.01,extra_x_padding = 0.02)
   data = dd.io.load(argv.setdefault('filename','solver.hdf5'))
   suppression = data['suppression']
   sup = [suppression[m,:,:].sum() for m in range(np.shape(suppression)[0])]
   
   
   tmp = data['fourier_suppression']
   sup_fourier = [tmp[m,:,:].sum() for m in range(np.shape(tmp)[0])]
   mfp = data['mfp']*1e6

   
   #sup_zero = len(mfp) * list(data['zero_suppression'])
   #kappa_bulk = data['kappa_bulk']
   #kappa_fourier = data['kappa_fourier']
   #ratio = kappa_fourier/kappa_bulk   
   #print(max(sup))
   plot(mfp,sup,color=c1)
   #print(sup[0])
   #print(sup[-1])

   plot(mfp,sup_fourier,color=c3)
   #plot(mfp,sup_zero,color=c1)
   ##plot(mfp,sup_iso,color='black')
   #plot([mfp[0],mfp[-1]],[ratio,ratio],color='black',ls='--')
   xscale('log')
   legend(['S','S$_D$'])
   #legend(['S','$S_F$','$S_0$'])#,'S$_0$'])#,'S$_0$','S$_{ISO}$'])
   ylim([0,0.5])
   grid('on')
   xlabel('Mean Free Path [$\mu$ m]')
   ylabel('Suppression Function')
   #show()



  


