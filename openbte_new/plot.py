from __future__ import absolute_import
from mpi4py import MPI
from .solver import Solver
from pyvtk import *
import numpy as np
import deepdish as dd
import os
import matplotlib
import matplotlib.patches as patches
from matplotlib.path import Path
from .fig_maker import *
from shapely import geometry,wkt
from shapely.geometry import MultiPoint,Point,Polygon,LineString
from scipy.interpolate import BarycentricInterpolator
from .WriteVTK import *
from .geometry import *

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


  if argv['variable'] == 'vtk' :
   self.save_vtk(argv)

 def save_vtk(self,argv):

   #Write data-------  
   #geo = dd.io.load('geometry.hdf5')
   geo = Geometry(type='load')
   solver = dd.io.load('solver.hdf5')
   argv.update({'Geometry':geo})
   vw = WriteVtk(argv)
   vw.add_variable(solver['fourier_temperature'],label = 'Fourier Temperature [K]')
   vw.add_variable(solver['fourier_flux'],label = r'''Thermal Flux [W/m/m]''')

   vw.add_variable(solver['bte_temperature'],label = r'''BTE Temperature [K]''')
   vw.add_variable(solver['bte_flux'],label = r'''BTE Thermal Flux [W/m/m]''')
   vw.write_vtk()  

 def plot_map(self,argv):

  if MPI.COMM_WORLD.Get_rank() == 0:
  
   init_plotting() 
   variable = argv['variable'].split('/')[1]
   geo = dd.io.load('geometry.hdf5')
   Nx = argv.setdefault('repeat_x',1)
   Ny = argv.setdefault('repeat_y',1)

   increment = [0,0]
   #data = np.ones(len(geo['elems']))
   if not variable == 'geometry': 
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
      if variable == 'geometry': 
        color = 'gray'
      else:
        color = color=cmap(data[e]-np.dot(displ,increment))
  
      patch = patches.PathPatch(path,linestyle=None,linewidth=0.1,color=color,zorder=10,alpha=1.0,joinstyle='miter')
      gca().add_patch(patch) 
      gca().add_patch(patch)
  
   axis('off')
   axis('equal')
   xlim([-Lx/2,Lx/2])
   ylim([-Ly/2,Ly/2])
   if argv['save_fig']:
    savefig(argv.setdefault('namefile','geometry.png'))
    close()
   else:
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
   grid()
   show()

  

 def plot_suppression_function(self,argv):

  if MPI.COMM_WORLD.Get_rank() == 0:
   init_plotting(extra_bottom_padding= 0.01,extra_x_padding = 0.02)
   data = dd.io.load(argv.setdefault('filename','solver.hdf5'))
   suppression = data['suppression']
   sup = [suppression[m,:,:].sum() for m in range(np.shape(suppression)[0])]
   
   
   tmp = data['fourier_suppression']
   mfp = data['mfp']*1e6

   
   plot(mfp,sup,color=c1)
   xscale('log')
   grid('on')
   xlabel('Mean Free Path [$\mu$ m]')
   ylabel('Suppression Function')
   show()



  


