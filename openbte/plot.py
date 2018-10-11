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
from scipy.interpolate import griddata

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
    #self.plot_map(argv)
    self.plot_node_map(argv)

  if argv['variable'] == 'suppression_function' :
   self.plot_suppression_function(argv)

  if argv['variable'] == 'distribution' :
   self.plot_distribution(argv)

  if argv['variable'] == 'vtk' :
   self.save_vtk(argv)


  if argv['variable'] == 'kappa_bte' :
   solver = dd.io.load('solver.hdf5')
   kappa_bte = solver['kappa_bte']
   if argv.setdefault('save',True) :
    f = open('kappa_bte.dat','w+')
    f.write(str(kappa_bte))
    f.close()

 def save_vtk(self,argv):
  #if MPI.COMM_WORLD.Get_rank() == 0:
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

   #init_plotting()
   variable = argv['variable']
   geo = dd.io.load('geometry.hdf5')
   Nx = argv.setdefault('repeat_x',1)
   Ny = argv.setdefault('repeat_y',1)

   increment = [0,0]
   #data = np.ones(len(geo['elems']))
   if not variable == 'geometry':
    solver = dd.io.load('solver.hdf5')

   if not variable == 'geometry':
    data = solver['fourier_flux']

    if len(np.shape(data)) == 1: #It is temperature
      increment = [1,0]
      a = 1.0; b = (float(Nx)-1.0)/float(Nx)
      data *= a-b
      data += b
    else: #it is temperature
      increment = [0,0]


    #In case we have flux
    if variable == 'bte_flux' or variable == 'fourier_flux':
     component = argv['direction']
     if component == 'x':
       data = data[0]
     if component == 'y':
       data = data[1]
     if component == 'z':
       data = data[2]
     #if component == 'magnitude':
     # print('ss')
    tmp = []
    for d in data :
     tmp.append(np.linalg.norm(d))
    data = tmp
     #-------------------------
    #print(np.shape(data))
    #quit()
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
   if argv.setdefault('save_fig',False):
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


 def plot_node_map(self,argv):


  if MPI.COMM_WORLD.Get_rank() == 0:

   geo = Geometry(type='load')
   variable = argv['variable'].split('/')[1]

   #init_plotting(extra_bottom_padding = -0.0,extra_x_padding = -0.0)
   #figure(num=None, figsize=(4, 4), dpi=80, facecolor='w', edgecolor='k')
   fig = gcf()
   #fig.figsize = (10,10)
   ax = fig.gca()
   ax.clear()



   #geo = dd.io.load('geometry.hdf5')

   solver = dd.io.load('solver.hdf5')
   argv.update({'Geometry':geo})

   vw = WriteVtk(argv)


   size = geo.state['size']
   Nx = argv.setdefault('repeat_x',1)
   Ny = argv.setdefault('repeat_y',1)
   Nz = argv.setdefault('repeat_z',1)


   (triangulation,tmp,nodes) = vw.get_node_data(solver[variable])

   #flip nodes-----This has to be checked

   #for node in nodes:
#    node[0] -= node[0]
    #node[1] -= node[1]
   #---------------

   if 'direction' in argv.keys():
    if argv['direction'] == 'x':
     data = np.array(tmp).T[0]
    if argv['direction'] == 'y':
     data = np.array(tmp).T[1]
    if argv['direction'] == 'z':
     data = np.array(tmp).T[2]
    if argv['direction'] == 'magnitude':
      data = []
      for d in tmp:
       data.append(np.linalg.norm(d))
      data = np.array(data)
   else:
    data = np.array(tmp).T[0]

   #data += 1.0
   vmin = argv.setdefault('vmin',min(data))
   vmax = argv.setdefault('vmax',max(data))
   #print(vmin)
   #print(vmax)


   ax.tripcolor(triangulation,np.array(data),shading='gouraud',norm=mpl.colors.Normalize(vmin=vmin,vmax=vmax),zorder=1)


   #colorbar(norm=mpl.colors.Normalize(vmin=min(data),vmax=max(data)))

    #Contour-----
   if argv.setdefault('iso_values',False):
    t = tricontour(triangulation,data,levels=np.linspace(vmin,vmax,10),colors='black',linewidths=1.5)

   Lx = geo.size[0]*Nx
   Ly = geo.size[1]*Ny
   if argv.setdefault('streamlines',False) and (variable == 'fourier_flux' or variable == 'bte_flux' ):


       n_lines = argv.setdefault('n_lines',10)*Ny #assuming transport across x
       xi = np.linspace(-Lx*0.5,Lx*0.5,300*Nx)
       yi = np.linspace(-Ly*0.5,Ly*0.5,300*Ny)
       x = np.array(nodes)[:,0]
       y = np.array(nodes)[:,1]
       z = np.array(tmp).T[0]
       Fx = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
       z = np.array(tmp).T[1]
       Fy = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
       #sx = np.concatenate((-Lx*0.5*np.ones(n_lines),Lx*0.5*np.ones(n_lines)))
       #sy = np.concatenate((np.linspace(-Ly*0.5*0.9,Ly*0.5*0.9,n_lines),np.linspace(-4.8,4.8,n_lines)))

       seed_points = np.array([-Lx*0.5*0.99*np.ones(n_lines),np.linspace(-Ly*0.5*0.98,Ly*0.5*0.98,n_lines)])
       ss = streamplot(xi, yi, Fx, Fy,maxlength = 1e8,start_points=seed_points.T,integration_direction='both',color='r',minlength=0.95,linewidth=1)
       #a = np.shape(np.array(ss.lines.get_segments()))
       #print(a)

       #quit()

   if argv.setdefault('plot_interfaces',False):

    for ll in geo.side_list['Interface']:
      p1 = geo.state['nodes'][geo.state['sides'][ll][0]][:2]
      p2 = geo.state['nodes'][geo.state['sides'][ll][1]][:2]
      for nx in range(Nx):
       for ny in range(Ny):
        for nz in range(Nz):
         P = np.array([size[0]*(nx-(Nx-1)*0.5),\
              size[1]*(ny-(Ny-1)*0.5)])

         pp1 = np.array(p1) + P
         pp2 = np.array(p2) + P

         plot([pp1[0],pp2[0]],[pp1[1],pp2[1]],color='w',ls='--',zorder=1)


       #plot(seed_points[0], seed_points[1], 'bo')



   axis('equal')
   axis('on')
   #tight_layout()
   gca().invert_yaxis()
   xlim([-Lx*0.5,Lx*0.5])
   ylim([-Ly*0.5,Ly*0.5])
   show()


 def plot_suppression_function(self,argv):

  #if MPI.COMM_WORLD.Get_rank() == 0:
   #init_plotting(extra_bottom_padding= 0.01,extra_x_padding = 0.02)
   data = dd.io.load(argv.setdefault('filename','solver.hdf5'))

   suppression = data['suppression']
   iso_sup = data['iso_suppression']
   #iso_sup = [iso_suppression[m].sum() for m in range(np.shape(iso_suppression)[0])]

   #print(suppression)
   #quit()
   tmp = data['fourier_suppression']
   sup = [suppression[m,:,:].sum() for m in range(np.shape(suppression)[0])]
   sup_fourier = [tmp[m,:,:].sum() for m in range(np.shape(tmp)[0])]


   mfp = data['mfp']*1e6
   #print(sup_fourier[0])
   if argv.setdefault('show',True):
    plot(mfp,sup,color=c1)
    plot(mfp,sup_fourier,color=c2)
    plot(mfp,iso_sup,color=c3)

    xscale('log')
    grid('on')
    xlabel('Mean Free Path [$\mu$ m]')
    ylabel('Suppression Function')
    show()
   if argv.setdefault('write',True):
    suppression.dump('suppression.dat')
