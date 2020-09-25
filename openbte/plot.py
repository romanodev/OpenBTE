import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pyvtk import *
import numpy as np
import os
from .utils import *
#import deepdish as dd
from .viewer import *
import matplotlib
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.pylab import *

from mpi4py import MPI
comm = MPI.COMM_WORLD

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

  if comm.rank == 0:

   #import data-------------------------------
   if 'geometry' in argv.keys():
    self.mesh = argv['geometry'].data
   else: 
     a = np.load('geometry.npz',allow_pickle=True)
     self.mesh = {key:a[key].item() for key in a}['arr_0']

   self.dim = int(self.mesh['meta'][2])

   if 'solver' in argv.keys():
    self.solver = argv['solver'].state
   else: 
     self.solver = load_data('solver')

   #if 'material' in argv.keys():
   # self.solver = argv['material']
   #else: 
   #    if os.path.isfile('material.h5') :
   #     self.material = load_dictionary('material.h5')
   #--------------------------------------------

   model = argv['model'].split('/')[0]

   if model == 'structure':
    self.duplicate_cells(**argv)
    self.plot_structure(**argv)

   elif model == 'maps':
    self.duplicate_cells(**argv)
    self.plot_maps(**argv)

   elif model == 'matplotlib':
    self.plot_matplotlib(**argv)

   elif model == 'vtu':
    #self.duplicate_cells(**argv)
    #self.write_cell_vtu(**argv)   
    self.write_vtu(**argv)   


 def plot_matplotlib(self,**argv):

    cmap = matplotlib.cm.get_cmap('viridis')
    size = self.mesh['size']
    frame = self.mesh['frame']
    lx = size[0]
    ly = size[1]
    fig = figure(num=" ", figsize=(4*lx/ly, 4), dpi=80, facecolor='w', edgecolor='k')
    axes([0,0,1.0,1.0])
    xlim([-lx/2.0,lx/2.0])
    ylim([-ly/2.0,ly/2.0])
     
    #plot frame---
    #path = create_path(frame)
    #patch = patches.PathPatch(path,linestyle=None,linewidth=0.1,color='#D4D4D4',zorder=1,joinstyle='miter')
    #gca().add_patch(patch);

    idd = 3
    if idd in [1,3]:
     data = np.array([np.linalg.norm(tmp) for tmp in self.solver['variables'][idd]['data']])
     print(min(data),max(data))
    else: 
     data = self.solver['variables'][0]['data']

    data = (data - min(data))/(max(data)-min(data))

    for ne in range(len(self.mesh['elems'])):
      cc =  self.mesh['elem_mat_map'][ne]
      if cc == 1:
         color = 'gray'
      else:   
         color = 'green'
      self.plot_elem(ne,color = cmap(data[ne]))
      #self.plot_elem(ne,color = color)

    #plot interface sides 
    for side in self.mesh['interface_sides']:
      p1 = self.mesh['sides'][side][0]
      p2 = self.mesh['sides'][side][1]
      n1 = self.mesh['nodes'][p1]
      n2 = self.mesh['nodes'][p2]
      plot([n1[0],n2[0]],[n1[1],n2[1]],color='#1f77b4',lw=1,zorder=1)

    show()  
 
 def plot_elem(self,ne,color='gray') :

    elem = self.mesh['elems'][ne][:self.mesh['side_per_elem'][ne]]
    pp = self.mesh['nodes'][elem,:2]
    path = create_path(pp)
    patch = patches.PathPatch(path,linestyle=None,linewidth=0.0,color=color,zorder=1,joinstyle='bevel',alpha=0.7)
    gca().add_patch(patch)


 def duplicate_cells(self,**argv):

   #Compute periodic_nodes
   
   nodes = np.round(self.mesh['nodes'],4)
   pnodes = []
   for s in list(self.mesh['periodic_sides'])+ list(self.mesh['inactive_sides']):
     pnodes += list(self.mesh['sides'][s])
   pnodes = list(np.unique(np.array(pnodes)))

   #---------------------------
   size = self.mesh['size']
   n_nodes = len(nodes)
   repeat = argv.setdefault('repeat',[1,1,1])
   Nx = repeat[0] 
   Ny = repeat[1] 
   Nz = repeat[2] 
   new_nodes = list(nodes.copy())
   corr = np.zeros(n_nodes*Nx*Ny*Nz,dtype=int)
   corr[:n_nodes] = range(n_nodes)
   for nx in range(Nx):
    for ny in range(Ny):
     for nz in range(Nz):
      P = [size[0]*nx,size[1]*ny,size[2]*nz]
      index = nx * Ny * Nz * n_nodes +  ny *  Nz * n_nodes + nz * n_nodes
      if index > 0:
       for n in range(n_nodes):
        node_trial = [nodes[n,0] + P[0],\
                      nodes[n,1] + P[1],\
                      nodes[n,2] + P[2]]
 
        if n in pnodes:
         a = np.where(np.linalg.norm(np.array(new_nodes)[pnodes]-node_trial,axis=1)<1e-3)[0]
         if len(a) > 0:
           corr[index+n] = pnodes[a[0]]
         else:    
           new_nodes.append(node_trial)
           pnodes.append(len(new_nodes)-1)
           corr[index+n] = len(new_nodes)-1
        else:  
         new_nodes.append(node_trial)
         corr[index+n] = len(new_nodes)-1

   new_nodes = np.array(new_nodes)

   #Duplicate Elements
   cells = []
   mapp = {}
   for nx in range(Nx):
    for ny in range(Ny):
     for nz in range(Nz):
      index = nx * Ny * Nz * n_nodes +  ny *  Nz * n_nodes + nz * n_nodes
      for k,gg in enumerate(self.mesh['elems']):
        tmp = []
        for n in gg:
         tmp.append(corr[n+index])
        cells.append(tmp)
        mapp.setdefault(k,[]).append(len(cells)-1)

   #Duplicate sides---------------------------------
   self.surface_sides = []
   self.surface_nodes = []
   self.indices = []
   for nx in range(Nx):
    for ny in range(Ny):
     for nz in range(Nz):
      index = nx * Ny * Nz * n_nodes +  ny *  Nz * n_nodes + nz * n_nodes
      for s in list(self.mesh['boundary_sides']) + list(self.mesh['periodic_sides']) + list(self.mesh['inactive_sides']):
        nodes = self.mesh['sides'][s]  
        tmp = []
        for n in nodes:
         self.surface_nodes.append(new_nodes[corr[n+index]])   
         self.indices.append(corr[n+index])
         tmp.append(len(self.surface_nodes)-1)
        self.surface_sides.append(tmp)

   self.surface_sides = np.array(self.surface_sides)     
   self.surface_nodes = np.array(self.surface_nodes)     
   self.indices = np.array(self.indices)     
   #-------------------------------------------------

   #duplicate variables---
   for n,(key, value) in enumerate(self.solver['variables'].items()):
     if value['data'].ndim == 1:
      tmp = np.zeros(len(self.mesh['elems'])*np.prod(argv['repeat']))
     else: 
      tmp = np.zeros((len(self.mesh['elems'])*np.prod(argv['repeat']),self.dim))

     #NEW
     g = 0
     for nx in range(Nx):
      for ny in range(Ny):
       for nz in range(Nz):
        inc = value['increment'][0]*nx + value['increment'][1]*ny + value['increment'][2]*nz
        for k,gg in enumerate(self.mesh['elems']):
          tmp[g] = value['data'][k] - inc
          g +=1

     #OLD
     #for k,v in mapp.items():
     #  for vv in v: tmp[vv] = value['data'][k]  

     self.solver['variables'][key]['data'] = tmp

   self.mesh['nodes'] = new_nodes
   self.mesh['elems'] = cells



 def write_cell_vtu(self,**argv):

   output = []
   strc = 'CellData('
   for n,(key, value) in enumerate(self.solver['variables'].items()):

     #if  value['data'].shape[0] == 1:  value['data'] =  value['data'][0] #band-aid solution

     #if value['data'].ndim == 1:
     # tmp = np.zeros(len(self.mesh['elems'])*np.prod(argv['repeat']))
     #else: 
     # tmp = np.zeros((len(self.mesh['elems'])*np.prod(argv['repeat']),3))
     #for k,v in mapp.items():
     #  for vv in v:
     #    tmp[vv] = value['data'][k]  

     #output.append(tmp)

     output.append(value['data'])
     name = value['name'] + '[' + value['units'] + ']'
     if value['data'].ndim == 1:
       strc += r'''Scalars(output[''' + str(n) + r'''],name =' ''' + name +  r''' ')'''
    
     if value['data'].ndim == 2:
       strc += r'''Vectors(output[''' + str(n) + r'''],name =' ''' + name +  r''' ')'''

     if n == len(self.solver['variables'])-1:
      strc += ')'
     else:
      strc += ','

   print(strc)
   data = eval(strc)
   if self.mesh['dim'] == 3:
    vtk = VtkData(UnstructuredGrid(self.mesh['nodes'],tetra=self.mesh['elems']),data)
   else :
    vtk = VtkData(UnstructuredGrid(self.mesh['nodes'][:,:2],triangle=self.mesh['elems']),data)

   vtk.tofile('output','ascii')


 def write_vtu(self,**argv):

   self.duplicate_cells(**argv)

   for key in self.solver['variables'].keys():
       self.solver['variables'][key]['data'] = self._get_node_data(self.solver['variables'][key]['data'])


   output = []
   strc = 'PointData('
   for n,(key, value) in enumerate(self.solver['variables'].items()):


     if  value['data'].shape[0] == 1:  value['data'] =  value['data'][0] #band-aid solution



     name = value['name'] + '[' + value['units'] + ']'
     if value['data'].ndim == 1:
       strc += r'''Scalars(output[''' + str(n) + r'''],name =' ''' + name +  r''' ')'''
       output.append(value['data'])
    
     if value['data'].ndim == 2:

       if self.mesh['dim'] == 2:
        t = np.zeros_like(value['data'])
        value['data'] = np.concatenate((value['data'],t),axis=1)[:,:3]
       output.append(value['data'])

       strc += r'''Vectors(output[''' + str(n) + r'''],name =' ''' + name +  r''' ')'''

     if n == len(self.solver['variables'])-1:
      strc += ')'
     else:
      strc += ','

   data = eval(strc)

   if self.mesh['dim'] == 3:
    vtk = VtkData(UnstructuredGrid(self.mesh['nodes'],tetra=self.mesh['elems']),data)
   else :
    nodes = self.mesh['nodes']
    vtk = VtkData(UnstructuredGrid(nodes,triangle=self.mesh['elems']),data)

   vtk.tofile('output','ascii')


 def _get_node_data(self,data,indices=[]):


   if data.ndim > 1:
       node_data = np.zeros((len(self.mesh['nodes']),int(self.dim)))
   else:   
       node_data = np.zeros(len(self.mesh['nodes']))

   conn = np.zeros(len(self.mesh['nodes']))
   for k,e in enumerate(self.mesh['elems']):
     for n in e:   
      node_data[n] += data[k]
      conn[n] +=1
   for n in range(len(self.mesh['nodes'])):   
    node_data[n] /= conn[n]


   if len(indices) > 0: 
    node_data = node_data[indices]


   return node_data


 def get_surface_nodes(self):

     triangles = []
     nodes = []
     for l in  list(self.mesh['boundary_sides']) + \
               list(self.mesh['periodic_sides']) + \
               list(self.mesh['inactive_sides']) :
          tmp = []
          for i in self.mesh['sides'][l]:
              #if i in nodes:  
              # k = nodes.index(i)   
              #else: 
               nodes.append(i)
               k = len(nodes)-1
               tmp.append(k)
          triangles.append(tmp)


     return np.array(triangles),nodes

 
 def plot_structure(self,**argv):

    data = {0:{'name':'Structure','units':'','data':np.zeros(len(self.mesh['nodes'] if self.dim == 2 else self.indices))}}

    plot_results(data,np.array(self.mesh['nodes']) if self.dim == 2 else self.surface_nodes,\
                                        np.array(self.mesh['elems']) if self.dim == 2 else self.surface_sides,**argv)

 def plot_maps(self,**argv):

   for key in self.solver['variables'].keys():
        self.solver['variables'][key]['data'] = self._get_node_data(self.solver['variables'][key]['data'],\
                indices= self.indices if self.dim == 3 else range(len(self.mesh['nodes'])))
   

   self.solver['variables'][-1] = {'name':'Structure','units':'','data':np.zeros(len(self.mesh['nodes'] if self.dim == 2 else self.indices))}

   if argv.setdefault('save_plot',False):
    dd.io.save('plot.h5',{'variables':self.solver['variables'],'nodes':self.mesh['nodes'] if self.dim == 2 else self.surface_nodes,\
                                                               'elems':self.mesh['elems'] if self.dim == 2 else self.surface_sides})

   if argv.setdefault('show',True):
    plot_results(self.solver['variables'],np.array(self.mesh['nodes']) if self.dim == 2 else self.surface_nodes,\
                                          np.array(self.mesh['elems']) if self.dim == 2 else self.surface_sides,**argv)

