from __future__ import absolute_import
from mpi4py import MPI
from .solver import Solver
from pyvtk import *
import numpy as np
import deepdish as dd
import os
import matplotlib
if not matplotlib.get_backend() == 'Qt5Agg': matplotlib.use('Qt5Agg')
import matplotlib.pylab as plt
from matplotlib.colors import Colormap
import matplotlib.patches as patches
from matplotlib.path import Path
from .fig_maker import *
from shapely import geometry,wkt
from scipy.interpolate import BarycentricInterpolator
from .WriteVTK import *
from .geometry import *
from scipy.interpolate import griddata
from shapely.geometry import MultiPoint,Point,Polygon,LineString
import shapely
from matplotlib.widgets import Slider, Button, RadioButtons

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


  if MPI.COMM_WORLD.Get_rank() == 0:
   self.solver = dd.io.load(argv.setdefault('filename','solver.hdf5'))
   self.Nx = argv.setdefault('repeat_x',1)
   self.Ny = argv.setdefault('repeat_y',1)

   if argv['variable'] == 'map':
     self.plot_map_new(**argv)


   if '/' in argv['variable']:
    model= argv['variable'].split('/')[0]
    if model == 'map':
     self.plot_map(argv)
    if model == 'line':
     self.plot_over_line(argv)

   #if argv['variable'] == 'suppression_function' :
   # self.plot_suppression_function(argv)

   if argv['variable'] == 'SUP' :
    self.SUP()


   if argv['variable'] == 'distribution' :
    self.plot_distribution(argv)

   if argv['variable'] == 'vtk' :
    self.save_vtk(argv)


   if argv['variable'] == 'kappa_bte' :
    solver = dd.io.load('state.hdf5')
    kappa_bte = solver['kappa_bte']
    if argv.setdefault('save',True) :
     f = open('kappa_bte.dat','w+')
     f.write(str(kappa_bte))
     f.close()

 def save_vtk(self,argv):
  #if MPI.COMM_WORLD.Get_rank() == 0:
   #Write data-------
   #geo = dd.io.load('geometry.hdf5')
   geo = Geometry(model='load')
   #solver = dd.io.load('solver.hdf5')
   solver = self.solver
   argv.update({'Geometry':geo})
   vw = WriteVtk(argv)
   #vw.add_variable(solver['fourier_temperature'],label = 'Fourier Temperature [K]')
   #vw.add_variable(solver['fourier_flux'],label = r'''Thermal Flux [W/m/m]''')

   if 'temperature' in solver.keys():
    vw.add_variable(solver['temperature'],label = r'''BTE Temperature [K]''')

   if 'flux' in solver.keys():
    vw.add_variable(solver['flux'],label = r'''BTE Thermal Flux [W/m/m]''')

   vw.add_variable(solver['temperature_fourier'],label = r'''Fourier Temperature [K]''')
   vw.add_variable(solver['flux_fourier'],label = r'''Fourier Thermal Flux [W/m/m]''')


   vw.write_vtk()


 def SUP(self):

    SUP = self.solver['SUP'] 
    MFP = self.solver['MFP']

    plt.plot(MFP,SUP)

    plt.xscale('log')

    plt.show()


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


 def plot_over_line(self,argv):

  if MPI.COMM_WORLD.Get_rank() == 0:
   self.geo = Geometry(type='load')
   (data,nodes,triangulation) = self.get_data(argv)
   (Lx,Ly) = self.geo.get_repeated_size(argv)

   #--------
   p1 = np.array([-Lx/2.0+1e-7,0.0])
   p2 = np.array([Lx/2.0*(1-1e-7),0.0])
   (x,data,int_points) = self.compute_line_data(p1,p2,data)
   #--------
   axes([0.5,0.01,0.5,1.0])
   plot(x,data,linewidth=2)

   for p in int_points:
    plot([p,p],[min(data),max(data)],ls='--',color='k')
   plot([min(x),max(x)],[0,0],ls='--',color='k')
   gca().yaxis.tick_right()


 def point_in_elem(self,elem,p):
  poly = []
  for n in self.geo.elems[elem]:
   t1 = self.geo.nodes[n][0]
   t2 = self.geo.nodes[n][1]
   poly.append((t1,t2))
  polygon = Polygon(poly)
  if polygon.contains(Point(p[0],p[1])):
   return True
  else:
   return False

 def compute_line_data(self,p1,p2,original):

  N = 1000
  tt = []
  delta = np.linalg.norm(p2-p1)/N
  neighbors = range(len(self.geo.elems))
  h1 = 0
  h2 = 0
  gamma = 1e-4
  data = []
  x = []
  int_points = []
  for n in range(N+1):
   r = p1 + n*delta*(p2-p1)
   #Here we include extra points close to the interfaces-------
   if n > 0:
    tmp = self.geo.cross_interface(p1 + (n-1)*delta*(p2-p1),r)
    versor = r - p1 + (n-1)*delta*(p2-p1)
    versor /= np.linalg.norm(versor)
    if len(tmp) > 0:
     pp = [tmp-gamma*versor,tmp + gamma*versor,r]
     x.append(n*delta-np.linalg.norm(r-tmp)-gamma)
     x.append(n*delta-np.linalg.norm(r-tmp)+gamma)
     x.append(n*delta)
     int_points.append(n*delta-np.linalg.norm(r-tmp)-gamma)
    else:
     pp = [r]
     x.append(n*delta)
   else:
    pp = [r]
    x.append(n*delta)
   #----------------------------------------------------------
   for p in pp:
    is_in = False
    for elem in neighbors:
     if self.point_in_elem(elem,p) :
      is_in = True
      value = self.geo.compute_2D_interpolation(original,p,elem)
      data.append(value)
      neighbors = self.geo.get_elem_extended_neighbors(elem)
      break
    if not is_in:
     for elem in range(len(self.geo.elems)):
      if self.point_in_elem(elem,p) :
       neighbors = self.geo.get_elem_extended_neighbors(elem)
       value = self.geo.compute_2D_interpolation(original,p,elem)
       data.append(value)
       break

  return x,data,int_points




 def get_data(self,argv):

  size = self.geo.size
  Nx = argv.setdefault('repeat_x',1)
  Ny = argv.setdefault('repeat_y',1)
  Nz = argv.setdefault('repeat_z',1)
  Lx = size[0]*self.Nx
  Ly = size[1]*self.Ny
  variable = argv['variable'].split('/')[1]
  solver = dd.io.load('solver.hdf5')
  argv.update({'Geometry':self.geo})

  vw = WriteVtk(argv)

  if 'index' in argv.keys():
    ii = int(argv['index'])
    if variable == 'temperature':
     data = solver['temperature_vec'][ii]
     #vmin = np.min(solver['temperature_vec'])
     #vmax = np.max(solver['temperature_vec'])
    else: 
     data = solver['flux_vec'][ii]
     #vmin = np.min(solver['flux_vec'])
     #vmax = np.max(solver['flux_vec'])

  else:
    data = solver[variable]
    #vmin = min(data)
    #vmax = min(data)

  (triangulation,tmp,nodes) = vw.get_node_data(data)

  argv.setdefault('direction',None)  
  if argv['direction'] == 'x':
     data = np.array(tmp).T[0]
  elif argv['direction'] == 'y':
     data = np.array(tmp).T[1]
  elif argv['direction'] == 'z':
     data = np.array(tmp).T[2]
  elif argv['direction'] == 'magnitude':
      data = []
      for d in tmp:
       data.append(np.linalg.norm(d))
      data = np.array(data)
  else:
    data = np.array(tmp).T[0]

  return data,nodes,triangulation



 def plot_variable(self,label_in):
    
    (Lx,Ly) = self.geo.get_repeated(self.Nx,self.Ny)
    Sx = 8
    Sy = Sx*Ly/Lx
    delta = 0.2

    argv = {}
    
    if len(label_in.split('_')) > 1:
     n = label_in.split('_')[1] 
     argv['index'] = n 
     label_in = label_in.split('_')[0] 
     #if label_in.split('_')[0] == 'temperature':        
     #  label = temperature
     #self.current_label = 'temperature'

    if label_in == 'Temperature':
       label = 'temperature' 
       self.current_label = label
    elif label_in == 'Flux (X)':
       label = 'flux' 
       argv['direction']='x'
       self.current_label = 'fluxx'
    elif label_in == 'Flux (Y)':
       label = 'flux' 
       argv['direction']='y'
       self.current_label = 'fluxy'
    else:  
       label = 'flux' 
       argv['direction']='magnitude'
       self.current_label = 'fluxm'

    self.ax = axes([delta,delta*Sx/(Sy+Sx*delta),1.0,1.0])
    argv['variable'] = 'variable/' + label
    (data,nodes,triangulation) =  self.get_data(argv)
    vmin = argv.setdefault('vmin',min(data))
    vmax = argv.setdefault('vmax',max(data))
    self.ax.tripcolor(triangulation,np.array(data),shading='gouraud',norm=mpl.colors.Normalize(vmin=vmin,vmax=vmax),zorder=1)
    self.ax.axis('off')
    self.ax.axis('equal')
    self.ax.invert_yaxis()
    plt.xlim([-Lx*0.5,Lx*0.5*(1 + 0.5)])
    plt.ylim([-Ly*0.5,Ly*0.5*(1 + 0.5)])
    plt.draw()
    self.plotted = True



 def plot_map_new(self,**argv):
   self.mode = 'integrated'  
   self.ax = -1
   #set main figure----
   self.geo = Geometry(model='load')
   self.original_size = self.geo.size[0]
   self.Nx = argv.setdefault('repeat_x',1)
   self.Ny = argv.setdefault('repeat_y',1)
   self.plotted = False
   self.geo = Geometry(model='load')
   (Lx,Ly) = self.geo.get_repeated(self.Nx,self.Ny)
   Sx = 8
   Sy = Sx*Ly/Lx
   delta = 0.2
   fig = figure(num=' ', figsize=(Sx*(1+delta), Sy+Sx*delta), dpi=80, facecolor='w', edgecolor='k')


   self.plot_variable('Temperature')
   axcolor = 'lightgoldenrodyellow'
   rax = plt.axes([delta*0.1, 0.5,delta*0.8, delta*0.8], facecolor=axcolor)
   radio = RadioButtons(rax, ('Temperature', 'Flux (X)','Flux (Y)','Flux'), active=0)
   radio.on_clicked(self.plot_variable)


   mfp_min = self.solver['MFP_SAMPLED'][0]
   mfp_max = self.solver['MFP_SAMPLED'][-1]
   vmin = np.log10(mfp_min/self.original_size)
   vmax = np.log10(mfp_max/self.original_size)

   axamp = plt.axes([0.3,0.02,0.6,0.05], facecolor=axcolor)
   samp = Slider(axamp, 'Log Kn', vmin, vmax, valinit=(vmin+vmax)/2,dragging=True,closedmin=False,closedmax=False)
   samp.on_changed(self.update_slider)
   self.current_slider_value = (vmin+vmax)/2
   rax = plt.axes([delta*0.1, 0.3,delta*0.8, delta*0.8], facecolor=axcolor)
   radio2 = RadioButtons(rax, ('Integrated', 'MFP'), active=0)
   radio2.on_clicked(self.switch_mode)

   show()

 def switch_mode(self,label):

     if label == 'Integrated':
        self.mode == 'integrated' 
        if self.current_label == 'temperature':   
          self.plot_variable('Temperature') 

        if self.current_label == 'fluxx':   
          self.plot_variable('Flux (X)') 
       
        if self.current_label == 'fluxy':   
          self.plot_variable('Flux (Y)') 

        if self.current_label == 'fluxm':   
          self.plot_variable('Flux') 
     else:
      self.mode = 'mfp'
      self.update_slider(self.current_slider_value)


 def update_slider(self,value):

  self.current_slider_value = value
  if self.mode == 'mfp':
   mfp = pow(10,value)*self.original_size
   mfp_sampled = self.solver['MFP_SAMPLED'].copy()

   for n in range(len(mfp_sampled)-1):
    if (mfp_sampled[n] > mfp):
      if self.current_label == 'temperature':
         self.plot_variable('Temperature_' + str(n)) 
         return
      elif self.current_label == 'fluxx': 
         self.plot_variable('Flux (X)_' + str(n)) 
         return
      elif self.current_label == 'fluxy': 
         self.plot_variable('Flux (Y)_' + str(n)) 
         return
      elif self.current_label == 'fluxm': 
         self.plot_variable('Flux_' + str(n)) 
         return


 def plot_map(self,argv):

   self.geo = Geometry(model='load')
   (Lx,Ly) = self.geo.get_repeated_size(argv)
   Sx = 8
   Sy = Sx*Ly/Lx

   fig = figure(num=' ', figsize=(Sx,Sy), dpi=80, facecolor='w', edgecolor='k')
   axes([0,0,1.0,1.0])

   #fig = figure(num=' ', figsize=(8*Lx/Ly, 8), dpi=80, facecolor='w', edgecolor='k')
   #axes([0,0,0.5,1.0])
   #axes([0,0,1.0,1.0])


   #data += 1.0
   plt.set_cmap(Colormap('hot'))
   
   (data,nodes,triangulation) =  self.get_data(argv)
   vmin = argv.setdefault('vmin',min(data))
   vmax = argv.setdefault('vmax',max(data))
   tripcolor(triangulation,np.array(data),shading='gouraud',norm=mpl.colors.Normalize(vmin=vmin,vmax=vmax),zorder=1)

   #colorbar(norm=mpl.colors.Normalize(vmin=min(data),vmax=max(data)))

    #Contour-----
   if argv.setdefault('iso_values',False):
    t = tricontour(triangulation,data,levels=np.linspace(vmin,vmax,10),colors='black',linewidths=1.5)

   if argv.setdefault('streamlines',False):# and (variable == 'fourier_flux' or variable == 'flux' ):


       n_lines = argv.setdefault('n_lines',10)*self.Ny #assuming transport across x
       xi = np.linspace(-Lx*0.5,Lx*0.5,300*self.Nx)
       yi = np.linspace(-Ly*0.5,Ly*0.5,300*self.Ny)
       x = np.array(nodes)[:,0]
       y = np.array(nodes)[:,1]
       z = np.array(data).T[0]
       Fx = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
       z = np.array(data).T[1]
       Fy = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')

       seed_points = np.array([-Lx*0.5*0.99*np.ones(n_lines),np.linspace(-Ly*0.5*0.98,Ly*0.5*0.98,n_lines)])
       ss = streamplot(xi, yi, Fx, Fy,maxlength = 1e8,start_points=seed_points.T,integration_direction='both',color='r',minlength=0.95,linewidth=1)


   if argv.setdefault('plot_interfaces',False):
    pp = self.geo.get_interface_point_couples(argv)



   axis('off')
   gca().invert_yaxis()
   xlim([-Lx*0.5,Lx*0.5])
   ylim([-Ly*0.5,Ly*0.5])
   gca().margins(x=0,y=0)

   axis('tight')
   axis('equal')
   savefig('fig.png',dpi=200)
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
