from pyvtk import *
import numpy as np
import deepdish as dd
import os
import matplotlib
be = matplotlib.get_backend()
if not be=='nbAgg' and not be=='module://ipykernel.pylab.backend_inline':
 if not be == 'Qt5Agg': matplotlib.use('Qt5Agg')
#import matplotlib.pylab as plt
from matplotlib.pylab import *
from matplotlib.colors import Colormap
#import os.path
from  matplotlib import cm
from matplotlib.tri import Triangulation
from .utils import *
import deepdish as dd

class Plot(object):

 def __init__(self,**argv):

   #import data-------------------------------
   if 'geometry' in argv.keys():
    self.mesh = argv['geometry'].data
   else: 
    if os.path.isfile('geometry.h5') :   
     self.mesh = dd.io.load('geometry.h5')



   if 'solver' in argv.keys():
    self.solver = argv['solver']
   else: 
       if os.path.isfile('solver.h5') :
        #self.solver = load_dictionary('solver.h5')
        self.solver = dd.io.load('solver.h5')

   #if 'material' in argv.keys():
   # self.solver = argv['material']
   #else: 
   #    if os.path.isfile('material.h5') :
   #     self.material = load_dictionary('material.h5')
   #--------------------------------------------

   model = argv['model'].split('/')[0]

   if model == 'geometry':
    self.plot_geometry(**argv)

   elif model == 'maps':
    self.plot_maps(**argv)

   elif model == 'vtk':
    self.write_vtk(**argv)   


 def write_vtk(self,**argv):

   output = []
   strc = 'PointData('
   for n,(key, value) in enumerate(self.solver['variables'].items()):
     node_data = self.get_node_data(key)
     output.append(node_data)
     if node_data.ndim == 1:
       strc += r'''Scalars(output[''' + str(n) + r'''],name =' ''' + value +  r''' ')'''
    
     if node_data.ndim == 2:
       strc += r'''Vectors(output[''' + str(n) + r'''],name =' ''' + value +  r''' ')'''

     if n == len(self.solver['variables'])-1:
      strc += ')'
     else:
      strc += ','
   data = eval(strc)

   if self.mesh['dim'] == 3:
    vtk = VtkData(UnstructuredGrid(self.mesh['nodes'],tetra=self.mesh['elems']),data)
   else :
    vtk = VtkData(UnstructuredGrid(self.mesh['nodes'],triangle=self.mesh['elems']),data)

   vtk.tofile('output.vtk','ascii')



 def get_node_data(self,variable):


   data = self.solver[variable]

   if data.ndim > 1:
       node_data = np.zeros((len(self.mesh['nodes']),3))
   else:   
       node_data = np.zeros(len(self.mesh['nodes']))


   for k,e in enumerate(self.mesh['elems']):
     for n in e:
      node_data[n] += data[k]/self.mesh['conn'][n]

   return node_data


 def _plot_data(self,tri,variable):


   data = self.solver[variable]
   if data.ndim == 2:
      data = np.array([np.linalg.norm(d) for d in data]) 

   node_data = np.zeros(len(self.mesh['nodes']))
   conn = np.zeros(len(self.mesh['nodes']))
   for k,e in enumerate(self.mesh['elems']):
     for n in e:
      conn[n] +=1
      node_data[n] += data[k]


   node_data = [node_data[n]/conn[n] for n in range(len(node_data))]
   vmin = min(node_data)
   vmax = max(node_data)
   node_data -=vmin
   node_data /= (vmax-vmin)

   cc= 'viridis'
   tripcolor(tri,np.array(node_data),cmap=cc,shading='gouraud',norm=mpl.colors.Normalize(vmin=0,vmax=1),zorder=1)
   axis('off')
   axis('equal')
   Lx = self.mesh['size'][0]
   Ly = self.mesh['size'][1]
   xlim([-Lx/2,Lx/2])
   ylim([-Ly/2,Ly/2])

   show()



 def plot_maps(self,**argv):

   from google.colab import widgets

   tri = Triangulation(self.mesh['nodes'][:,0],self.mesh['nodes'][:,1], triangles=self.mesh['elems'], mask=None)

   figure(num=' ', figsize=(5,5), dpi=80, facecolor='w', edgecolor='k');

   titles = []
   variables = []
   directions = []


   if 'temperature' in self.solver.keys():
     titles.append('BTE Temperature')
     variables.append('temperature')
     directions.append(-1)

   if 'temperature_fourier' in self.solver.keys():
     titles.append('Fourier Temperature')
     variables.append('temperature_fourier')
     directions.append(-1)

   if 'flux' in self.solver.keys():
     titles.append('BTE Flux (magnitude)')
     variables.append('flux')
     directions.append(-1)

   if 'flux_fourier' in self.solver.keys():
     titles.append('Fourier flux (magnitude)')
     variables.append('flux_fourier')
     directions.append(-1)

   tb = widgets.TabBar(titles, location='top')

   for n,(variable,direction) in enumerate(zip(variables,directions)):
     with tb.output_to(n): self._plot_data(tri,variable)

 def plot_geometry_2D(self,**argv):

      translate = argv.setdefault('translate',[0,0]) 
    
      size = self.mesh['size']
      frame = self.mesh['frame']
      for ne in range(len(self.mesh['elems'])):
        cc =  self.mesh['elem_mat_map'][ne]
        if cc == 1:
           color = 'gray'
        else:   
           color = 'gray'
        self.plot_elem(ne,color=color,translate = translate)


      if argv.setdefault('plot_boundary',False):
       #plot Boundary Conditions-----
       for side in self.mesh['side_list']['Boundary'] :
        p1 = self.mesh['sides'][side][0]
        p2 = self.mesh['sides'][side][1]
        n1 = self.mesh['nodes'][p1] + translate[0]
        n2 = self.mesh['nodes'][p2] + translate[1]
        gca().plot([n1[0]+translate[0],n2[0]+translate[0]],[n1[1]+translate[1],n2[1]+translate[1]],color='#f77f0e',lw=2)
     
       for side in self.mesh['side_list']['Periodic'] + self.mesh['side_list']['Inactive']  :
        p1 = self.mesh['sides'][side][0]
        p2 = self.mesh['sides'][side][1]
        n1 = self.mesh['nodes'][p1] 
        n2 = self.mesh['nodes'][p2] 
        gca().plot([n1[0]+translate[0],n2[0]+translate[0]],[n1[1]+translate[1],n2[1]+translate[1]],color='g',lw=2,zorder=1)
        
      gca().axis('off')
     
   



 def plot_geometry(self,**argv):

     if self.mesh['dim'][0] == 2:

      lx = self.mesh['lx']
      ly = self.mesh['ly']
      fig = figure(num=" ", figsize=(4*lx/ly, 4), dpi=80, facecolor='w', edgecolor='k')
      ax = axes([0.,0.,1,1])
      nx = argv['nx']
      ny = argv['ny']
  
      ix = int(nx/2)
      iy = int(ny/2)
      for i in range(argv['nx']):
       for j in range(argv['ny']):
        if i == ix and j ==iy:
          plot_boundary = argv.setdefault('plot_boundary',False)
        else:  
          plot_boundary = False

        self.plot_geometry_2D(translate = [lx*(i-ix),ly*(j-iy)],plot_boundary = plot_boundary)

      ax.set_xlim([-lx/2*nx,lx/2*ny])
      ax.set_ylim([-ly/2*nx,ly/2*ny])
      savefig('test.png',dpi=600)
      show()

     else:
         from mpl_toolkits.mplot3d import Axes3D 
         from matplotlib.colors import LightSource
         x = []; y = [];z = []
         triangles = []
         #nodes = []
         for l in list(self.mesh['side_list']['Boundary']) + \
                  list(self.mesh['side_list']['Periodic']) + \
                  list(self.mesh['side_list']['Inactive']) :
          tmp = []   
          for i in self.mesh['sides'][l]: 
              x.append(self.mesh['nodes'][i][0])  
              y.append(self.mesh['nodes'][i][1])  
              z.append(self.mesh['nodes'][i][2]) 
              k = len(x)-1
              tmp.append(k)
          triangles.append(tmp) 
         
         fig = figure()
         ax = fig.gca(projection='3d')

         ax.plot_trisurf(x, y,z, linewidth=0.2,triangles=triangles,antialiased=True,edgecolor='gray',cmap='viridis',vmin=0.2,vmax=0.2)
         ax.axis('off')

         MAX = np.max(np.vstack((x,y,z)))
         for direction in (-1, 1):
          for point in np.diag(direction * MAX * np.array([1,1,1])):
           ax.plot([point[0]], [point[1]], [point[2]], 'w')

         show() 
            

 def plot_elem(self,ne,color='gray',translate = [0,0]) :

   
    elem = self.mesh['elems'][ne]
    pp = []
    for e in elem:
     pp.append(self.mesh['nodes'][e][:2] + np.array(translate))
    path = create_path(pp)
    patch = patches.PathPatch(path,linestyle=None,linewidth=0,color=color,zorder=1,joinstyle='miter',alpha=0.7)
    gca().add_patch(patch)


