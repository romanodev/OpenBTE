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
 
  if len(argv['model'].split('/')) == 2:
   if argv['model'].split('/')[0] == 'map':
     self.plot_map(argv)

  if argv['model'] == 'suppression_function' :
   self.plot_suppression_function()


 def plot_map(self,argv):

  if MPI.COMM_WORLD.Get_rank() == 0:


   
   var = argv['model'].split('/')[1]
   geo = dd.io.load('geometry.hdf5')
   solver = dd.io.load('solver.hdf5')
   Nx = argv.setdefault('Nx',4)
   Ny = argv.setdefault('Ny',4)
   increment = [0,0]

   data = np.ones(len(geo['elems']))
   variable = argv['model'].split('/')[1]
   if variable == 'temperature':
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
     Px = nx * Lx
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
      value = data[e]#-np.dot(displ,increment)
      if value < mm :  
       mm = value
      if value > mm2 :  
       mm2 = value
      patch = patches.PathPatch(path,linestyle=None,linewidth=0,edgecolor=None,color=cmap(value),facecolor='gray',zorder=10,alpha=0.5)
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

 def plot_suppression_function(self):

  if MPI.COMM_WORLD.Get_rank() == 0:
   init_plotting()
   data = dd.io.load('solver.hdf5')
   sup = data['suppression_function']
   mfp = data['mfp']*1e6
   plot(mfp,sup)
   xscale('log')
   grid('on')
   xlabel('Mean Free Path [$\mu$ m]')
   ylabel('Suppression Function')
   show()



  
