from __future__ import print_function
from __future__ import absolute_import
import os,sys
import numpy as np
import subprocess
import mpi4py
from pyvtk import *
from .GenerateSquareLatticePores import *
from .ComputeStructuredMesh import *
from .GenerateHexagonalLatticePores import *
from .GenerateStaggeredLatticePores import *
from .GenerateCustomPores import *
from .Generate2DInterface import *
from .GenerateRandomPoresOverlap import *
from .GenerateCorrelatedPores import *
from .GenerateInterface import *
from .GenerateRandomPoresGrid import *
from . import GenerateBulk2D
from . import GenerateBulk3D
from .porous import *
import networkx as nx


import matplotlib
be = matplotlib.get_backend()

if not be=='nbAgg' and not be=='module://ipykernel.pylab.backend_inline':
 if not be == 'Qt5Agg': matplotlib.use('Qt5Agg')

import matplotlib.patches as patches
from .fig_maker import *
from matplotlib.path import Path
from IPython.display import display, HTML
from shapely.geometry import LineString
import shapely
import pickle
import sparse
from .utils import *

#import GenerateInterface2D
#from nanowire import *
import deepdish as dd
from scipy.sparse import csc_matrix,dok_matrix
from matplotlib.pylab import *
from shapely.geometry import MultiPoint,Point,Polygon,LineString
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



def periodic_kernel(x1, x2, p,l,variance):
    return variance*np.exp(-2/l/l * np.sin(np.pi*abs(x1-x2)/p) ** 2)

def gram_matrix(xs,p,l,variance):
    return [[periodic_kernel(x1,x2,p,l,variance) for x2 in xs] for x1 in xs]

def generate_random_interface(p,l,variance,scale):

 xs = np.arange(-p/2, p/2,p/200)
 mean = [0 for x in xs]
 gram = gram_matrix(xs,p,l,variance)
 ys = np.random.multivariate_normal(mean, gram)*scale
 f = interpolate.interp1d(xs, ys,fill_value='extrapolate')

 return f



def create_path(obj):

   codes = [Path.MOVETO]
   for n in list(range(len(obj)-1)):
    codes.append(Path.LINETO)
   codes.append(Path.CLOSEPOLY)

   verts = []
   for tmp in obj:
     verts.append(tmp)
   verts.append(verts[0])

   path = Path(verts,codes)
   return path

class GeometryFull(object):

 def __init__(self,**argv):

  
  model = argv['model']

  if model == 'porous':

    if argv['submodel'] == 'aligned':   
     (self.frame,polygons) = GenerateSquareLatticePores(argv)

    if argv['submodel'] == 'staggered':
     (frame,polygons) = GenerateStaggeredLatticePores(argv)

    argv['polygons'] = polygons
    argv['frame'] = self.frame
    Porous(**argv)

  if model == 'bulk':
    GenerateBulk2D.mesh(argv)

  if model == 'interface':
       generate_interface(**argv)   
       Lx = float(argv['lx'])
       Ly = float(argv['ly'])
       self.frame = []
       self.frame.append([-Lx/2,Ly/2])
       self.frame.append([Lx/2,Ly/2])
       self.frame.append([Lx/2,-Ly/2])
       self.frame.append([-Lx/2,-Ly/2])

       self.data = self.compute_mesh_data(argv)
       self.build_interface(**argv)


  if argv.setdefault('save',True):
   save_dictionary(self.data,'geometry.h5')

  if argv.setdefault('show',False):
   self.plot_structure(show=True)
   

  
  '''
  self.structured = False
  direction = argv.setdefault('direction','x')
  if direction == 'x':
     self.direction = 0
     self.applied_grad = [1,0,0]
  if direction == 'y': 
     self.direction = 1
     self.applied_grad = [0,1,0]
  if direction == 'z':
     self.direction = 2
     self.applied_grad = [0,0,1]
  
  self.argv = argv.copy()
  geo_type = argv.setdefault('model','porous/square_lattice')

  if geo_type == 'geo':
   if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
    self.dim = 3   


    self.polygons=[]
    self.import_mesh()

    Lx = self.size[0]
    Ly = self.size[1]

    self.frame = []
    self.frame.append([-Lx/2,Ly/2])
    self.frame.append([Lx/2,Ly/2])
    self.frame.append([Lx/2,-Ly/2])
    self.frame.append([-Lx/2,-Ly/2])


    self.compute_mesh_data()
    data = {'state':self.state}
    if self.argv.setdefault('save',True):
      pickle.dump(self.state,open('geometry.p','wb'),protocol=pickle.HIGHEST_PROTOCOL)
    return
   else: data = None
   data = mpi4py.MPI.COMM_WORLD.bcast(data,root=0)
   self.state = data['state']

  if geo_type == 'load':
    self.state = pickle.load(open(argv.setdefault('filename_geometry','geometry.p'),'rb'))
    self._update_data()

  else:

   if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
    #porous-----
    polygons = []

    self.porous = False
    if  geo_type == 'porous/square_lattice' or\
        geo_type == 'porous/square_lattice_smooth' or\
        geo_type == 'porous/hexagonal_lattice' or\
        geo_type == 'porous/staggered_lattice' or\
        geo_type == 'porous/random_over_grid' or\
        geo_type == 'porous/random' or\
        geo_type == 'porous/custom':

     self.porous = True
     if geo_type == 'porous/square_lattice':
      self.frame,self.polygons = GenerateSquareLatticePores(argv)
     
     if geo_type == 'porous/square_lattice_smooth':
      self.frame,self.polygons = GenerateSquareLatticePoresSmooth(argv)

     if geo_type == 'porous/hexagonal_lattice':
      self.frame,self.polygons = GenerateHexagonalLatticePores(argv)

     if geo_type == 'porous/staggered_lattice':
      self.frame,self.polygons = GenerateStaggeredLatticePores(argv)

     if geo_type == 'porous/custom':
      self.frame,self.polygons = GenerateCustomPores(argv)

     if geo_type == 'porous/random':
      self.frame,self.polygons,tt,mind,porosity = GenerateRandomPoresOverlap(argv)
      self.tt = tt
      self.mind = mind
      self.porosity = porosity

     if geo_type == 'porous/random_over_grid':
      x,polygons = GenerateRandomPoresGrid(**argv)
      self.x = x

      argv['polygons'] = polygons
      argv['automatic_periodic'] = False
      self.frame,self.polygons = GenerateCustomPores(argv)


    self.Lz = float(argv.setdefault('lz',0.0))
    if geo_type == 'bulk':
      self.porous = True
      self.polygons = []
      Lx = float(argv['lx'])
      Ly = float(argv['ly'])

      self.frame = []
      self.frame.append([-Lx/2,Ly/2])
      self.frame.append([Lx/2,Ly/2])
      self.frame.append([Lx/2,-Ly/2])
      self.frame.append([-Lx/2,-Ly/2])

    #if geo_type == 'interface':
      #generate_interface(**argv)   
      #qui()

    if argv.setdefault('mesh',True):
     self.mesh(**argv)
     data = {'state':self.state}
   else:
     data = None
   data = mpi4py.MPI.COMM_WORLD.bcast(data,root=0)

   self.state = data['state']

   mpi4py.MPI.COMM_WORLD.Barrier()

   self._update_data()

  if argv.setdefault('savefig',False) or argv.setdefault('show',False) or argv.setdefault('store_rgb',False) :
   self.plot_polygons(**argv)
 '''
 def mesh(self,**argv):

   if self.Lz > 0.0:
    self.dim = 3
   else:
    self.dim = 2
   argv.update({'lz':self.Lz})
  
   if self.porous:
     
    if len(self.polygons) > 0 and self.dim == 3:
     #GenerateMesh3D.mesh(self.polygons,self.frame,argv)
     argv['polygons'] = self.polygons
     argv['frame'] = self.frame
     porous.Porous(**argv)

    if len(self.polygons) > 0 and self.dim == 2:
     GenerateMesh2D.mesh(self.polygons,self.frame,argv)
    if len(self.polygons) == 0 and self.dim == 2:
     GenerateBulk2D.mesh(argv)
    if len(self.polygons) == 0 and self.dim == 3:
     GenerateBulk3D.mesh(argv)
     #-----------------------------------


   if argv['model'] == 'porous/square_structured':

       self.compute_structured_mesh_new(**argv)   
       self.compute_mesh_data()

       #define delta--
       phi = argv['porosity']
       l = argv['l']
       delta = np.sqrt(phi)*l/2
       #-------------

       elem_mat_map = {}
       for ne in self.elem_list['active']:
        c,dummy = self.compute_general_centroid_and_side(ne)
        if not  ((c[0] > -delta and c[0] < delta) and (c[1] > -delta and c[1] < delta)):
         elem_mat_map.update({ne:0})

       data = self.add_patterning(elem_mat_map)
       self.state.update(data)

       return

  
   if argv['model'] == 'porous/random_correlated':

       self.compute_structured_mesh_new(**argv)   
       self.compute_mesh_data()

       if argv.setdefault('load_conf',False):
        #grid = np.loadtxt(argv['filename'],delimiter=',',dtype=int)
        #grid = np.loadtxt(argv['filename'],delimiter=',',dtype=int)
        grid = np.load(open('conf.dat','rb'),allow_pickle=True)   

       else:
        grid = generate_correlated_pores(**argv)
        if argv.setdefault('save_conf',False):
         grid.dump(open('conf.dat','wb'))   

       #Get elem mat map
       elem_mat_map = {}
       for ne in grid:
           elem_mat_map.update({ne:0})

       #----------------
       data = self.add_patterning(elem_mat_map)
        
       self.state.update(data)

       #self.eliminate_lone_pores() 

       return

   if argv['model'] == 'interface':

       generate_interface(**argv)   
       Lx = float(argv['lx'])
       Ly = float(argv['ly'])
       self.frame = []
       self.frame.append([-Lx/2,Ly/2])
       self.frame.append([Lx/2,Ly/2])
       self.frame.append([Lx/2,-Ly/2])
       self.frame.append([-Lx/2,-Ly/2])
       self.polygons = []


   if not argv.setdefault('only_geo',False):
    if argv['model'] == 'structured':
     self.compute_structured_mesh_new(**argv)   
    else:
     #Create mesh---
     subprocess.check_output(['gmsh','-optimize_netgen','-format','msh2','-' + str(self.dim),'mesh.geo','-o','mesh.msh'])
     self.import_mesh()
    self.compute_mesh_data()

    if argv['model'] == 'interface':
       self.build_interface(**argv)


   if self.argv.setdefault('save',True) and not argv.setdefault('only_geo',False):
     self.save(**argv)   

   return self.state

 def save(self,**argv):
     pickle.dump(self.state,open(argv.setdefault('filename_geometry','geometry.p'),'wb'),protocol=pickle.HIGHEST_PROTOCOL)


 def pattern(self,grid,lone=True):

   if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:

     elem_mat_map = {}
     for ne in grid:
      elem_mat_map.update({ne:0})

     data = self.add_patterning(elem_mat_map,lone=lone)
     data = {'state':data}
   else:
     data = None

   data = mpi4py.MPI.COMM_WORLD.bcast(data,root=0)
   self.state.update(data['state'])

   mpi4py.MPI.COMM_WORLD.Barrier()
   self._update_data()


 def update_island(self,covered):
   
     for j in self.elem_map[covered[-1]]:
         if j in self.elem_list['active'] and not j in covered: 
            covered.append(j)
            self.update_island(covered)
     return       

 def eliminate_lone_pores(self):

    islands = []
    for i in self.elem_list['active']:
      already = False  
      for s in islands:  
          if i in s:
            already = True
      if not already:
       islands.append([])  
       islands[-1].append(i) 
       self.update_island(islands[-1]) 
     
    #---------------------- 
    eliminated = []
    for island in islands:
      legit = False  
      for i in island:
          for ll in self.elem_side_map[i] :
           if ll in self.side_list['Periodic'] + self.side_list['Cold'] + self.side_list['Hot']:
              legit = True
              break
      #ELiminate region
      if not legit:
       for i in island:
         for side in self.elem_side_map[i]:
             if side in self.side_list['active']: 
              self.side_list['active'].remove(side) 
         self.elem_list['active'].remove(i)
         eliminated.append(i)
      #---------------------------------------
    return eliminated  
   




 # MPI.COMM_WORLD.Barrier()

 def get_repeated_size(self,argv):
    Nx = argv.setdefault('repeat_x',1)
    Ny = argv.setdefault('repeat_y',1)
    size = self.size
    Lx = size[0]*Nx
    Ly = size[1]*Ny
    return Lx,Ly

 def get_repeated(self,Nx,Ny):
    size = self.size
    Lx = size[0]*Nx
    Ly = size[1]*Ny
    return Lx,Ly

 def get_interface_point_couples(self,argv):

   Nx = argv.setdefault('repeat_x',1)
   Ny = argv.setdefault('repeat_y',1)


   for ll in self.side_list['Interface']:
    p1 = self.nodes[self.sides[ll][0]][:2]
    p2 = self.nodes[self.sides[ll][1]][:2]
    for nx in range(Nx):
      for ny in range(Ny):
       P = np.array([self.size[0]*(nx-(Nx-1)*0.5),\
                     self.size[1]*(ny-(Ny-1)*0.5)])
       p1 = np.array(p1) + P
       p2 = np.array(p2) + P
       plot([p1[0],p2[0]],[p1[1],p2[1]],color='w',ls='--',zorder=1)


 def compute_general_centroid_and_side(self,i,verbose=False):


  xmax = -1e4
  xmin = 1e4
  ymax = -1e4
  ymin = 1e4
  #print(' ')
  #plot([-2.5,-2.5,2.5,2.5,-2.5],[-2.5,2.5,2.5,-2.5,-2.5])
  #axis('equal')
  #axis('off')
  #print(len(self.elems[i]))

  for n in self.elems[i]:
    p = self.nodes[n]  
    xmin = min([xmin,p[0]])  
    xmax = max([xmax,p[0]])  
    ymin = min([ymin,p[1]])  
    ymax = max([ymax,p[1]]) 
  #  scatter(p[0],p[1])
  p0 = [xmin,ymin]
  p1 = [xmin,ymax]
  p2 = [xmax,ymax]
  p3 = [xmax,ymin]
 

  pp_final = -np.ones(4,dtype=int)
  for n in self.elems[i]:
      p = self.nodes[n][0:2] 
      if np.linalg.norm(p-p0) < 1e-4:
         pp_final[0] = n
      if np.linalg.norm(p-p1) < 1e-4:
         pp_final[1] = n
      if np.linalg.norm(p-p2) < 1e-4:
         pp_final[2] = n
      if np.linalg.norm(p-p3) < 1e-4:
         pp_final[3] = n 
 
  l = list(pp_final)
  
  ave = np.zeros(3)
  for p in l:
    ave +=self.nodes[p]
  ave = ave/4

  #text(ave[0],ave[1],str(i),color='black',va='center',ha='center')
  #show()

  l = np.linalg.norm(self.nodes[pp_final[1]]-self.nodes[pp_final[0]])

  return ave,l

  '''
   pm2 = self.elems[i][0]
   pm1 = self.elems[i][1]
   l = [pm2,pm1]
   delta = 1e-6
   for n in range(len(self.elems[i])-2):
     p = self.elems[i][(n+2)%len(self.elems[i])]
     
     if np.linalg.norm(np.cross(self.nodes[p]-self.nodes[pm1],self.nodes[pm1]-self.nodes[pm2])) < np.linalg.norm(self.nodes[pm1]-self.nodes[pm2])/100:
      l[-1] = p
     else: 
      l.append(p)
     pm2 = pm1
     pm1 = p
  
   #if len(l) == 2:
   # print(self.elems[i],l)
   # self.reorder(i,self.elems[i],verbose=verbose)
   # for u in self.elems[i]:
   #   scatter(self.nodes[u][0],self.nodes[u][1],color='g') 
   #   text(self.nodes[u][0],self.nodes[u][1],str(u))
   # show() 
   # quit()

   #check the last point---
   if np.linalg.norm(np.cross(self.nodes[l[0]]-self.nodes[l[-1]],self.nodes[l[-1]]-self.nodes[l[-2]])) < np.linalg.norm(self.nodes[l[-1]]-self.nodes[l[-2]])/100:

    del l[len(l)-1]

   #check the first point---
   #if not len(l) == 4:
   # print(self.elems[i])
   # quit()

   
   if np.linalg.norm(np.cross(self.nodes[l[0]]-self.nodes[l[-1]],self.nodes[l[1]]-self.nodes[l[0]])) < np.linalg.norm(self.nodes[l[1]]-self.nodes[l[0]])/100:
    del l[0]


   #----------------------- 
   #if i == 27:
   # print(self.elems[i],l)   
   # for u in self.elems[i]:
   #   scatter(self.nodes[u][0],self.nodes[u][1]) 
   #   text(self.nodes[u][0],self.nodes[u][1],str(u))
   # show()  
  
   ave = [0,0,0]
   for r in l:
     ave +=self.nodes[r]
   ave /=4  
   
   side = np.linalg.norm(self.nodes[l[1]]-self.nodes[l[2]])

   return ave,side
  '''

 def add_periodic_nodes(self,u,up):

     self.periodic_nodes.setdefault(u,[]).append(up)
     self.periodic_nodes.setdefault(up,[]).append(u)

 def update_elem_side_map(self,elems,sides): #first elements, then sides
  for elem in elems:
   for side in sides:
    self.elem_side_map.setdefault(elem,[]).append(side)
    if not elem in self.side_elem_map.setdefault(side,[]):   
     self.side_elem_map[side].append(elem)

 def remove_elem_side_map(self,elems,sides): #first elements, then sides
  for elem in elems:
   for side in sides:
       if side in self.elem_side_map[elem]:   
        self.elem_side_map[elem].remove(side)
       if elem in self.side_elem_map[side]:   
        self.side_elem_map[side].remove(elem)


 def update_elem_node_map(self,elems,nodes): #first elements, then sides

   for elem in elems:
    c =  self.compute_elem_centroid(elem)
    for node in nodes:

        #check is the are periodic versions, which are more appropriate 
        n_current = node
        if node in self.periodic_nodes.keys():
          dd = np.linalg.norm(c-self.nodes[node])
          for n2 in self.periodic_nodes[node]:
              if np.linalg.norm(c-self.nodes[n2]) < dd:
               dd = np.linalg.norm(c-self.nodes[n2])
               if n_current in self.elems[elem]:
                self.elems[elem].remove(n_current)
               n_current = n2
               if not n_current in self.elems[elem]:  
                self.elems[elem].append(n_current)
              else:  
               if n2 in self.elems[elem]:  
                self.elems[elem].remove(n2)
                
        #-------------------------------------------
        if not n_current in self.elems[elem]:  
            self.elems[elem].append(n_current)
        if not elem in self.node_elem_map.setdefault(n_current,[]):   
          self.node_elem_map[n_current].append(elem)

    self.elems[elem] = self.reorder(self.elems[elem])
          

    epsilon=1e-8 
    crossproduct = (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])

    # compare versus epsilon for floating point values, or != 0 if using integers
    if abs(crossproduct) > epsilon:
        return False

    dotproduct = (c[0] - a[0]) * (b[0] - a[0]) + (c[1] - a[1])*(b[1] - a[1])
    if dotproduct < 0:
        return False

    squaredlengthba = (b[0] - a[0])*(b[0] - a[0]) + (b[1] - a[1])*(b[1] - a[1])
    if dotproduct > squaredlengthba:
        return False

    return True



 def line_exist(self,i,s1,s2):
    #Check is there is periodicity
    per = -1 
    active_2 = -1 

    g1 = s1; g2=s2
    if s1 in self.periodic_nodes.keys() and s2 in self.periodic_nodes.keys():
     for side in self.elem_side_map[i] + self.exlude:
       tmp = self.sides[side]  
       if (tmp[0] in self.periodic_nodes[s1] and tmp[1] in self.periodic_nodes[s2]) or \
         (tmp[0] in self.periodic_nodes[s2] and tmp[1] in self.periodic_nodes[s1]) :
         per = side
         if tmp[1] in self.periodic_nodes[s1]:
          g2 = tmp[0]; g1=tmp[1]
          self.sides[side] = [g1,g2]
         else: 
          g1 = tmp[0]; g2=tmp[1]
          
         active_2 = per in self.side_list['active']
         break

    #print(i,self.elem_side_map[i])
    for side in self.elem_side_map[i] + self.exlude:
      tmp = self.sides[side]  
      if (tmp[0] == s1 and tmp[1] == s2) or (tmp[0] == s2 and tmp[1] == s1) :

         active_1 = side in self.side_list['active']

         #swaping order for convenience---
         if not s1 == self.sides[side][0]:
           self.sides[side] = [s1,s2]  
         #----  

         if per == -1: per = side
          
         if active_1:
            return side,per,self.get_neighbor_elem(i,side),g1,g2
         else:
            if per == side:
             return per,side,self.get_neighbor_elem(i,side),g1,g2
            else:
             return per,side,self.get_neighbor_elem(i,per),g1,g2

    #print('g')
    return -1


 def point_exist(self,i,c):

    for n in self.elems[i]:
     if np.linalg.norm(c-self.nodes[n]) < 1e-6:
       return n

    return -1
          
            

 def plot_structure(self,**argv):

   self.argv = argv   
   if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:

    lx = abs(self.frame[0][0])*2
    ly = abs(self.frame[0][1])*2

    #init_plotting()
    close()
    #fig = figure(num=" ", figsize=(8*lx/ly, 4), dpi=80, facecolor='w', edgecolor='k')
    fig = figure(num=" ", figsize=(4*lx/ly, 4), dpi=80, facecolor='w', edgecolor='k')
    axes([0,0,1.0,1.0])
    #axes([0,0,0.5,1.0])


    xlim([-lx/2.0,lx/2.0])
    ylim([-ly/2.0,ly/2.0])

    path = create_path(self.frame)
    patch = patches.PathPatch(path,linestyle=None,linewidth=0.1,color='#D4D4D4',zorder=1,joinstyle='miter')
    gca().add_patch(patch);

    #if self.argv.setdefault('inclusion',False):
    # color='g'
    #else:
    # color='white'
   
    for ne in range(len(self.data['elems'])):
      cc =  self.data['elem_mat_map'][ne]
      if cc == 1:
         color = 'gray'
      else:   
         color = 'green'
      self.plot_elem(ne,color=color)


    #for poly in self.polygons:
    # path = create_path(poly)
    # patch = patches.PathPatch(path,linestyle=None,linewidth=0.1,color=color,zorder=2,joinstyle='miter')
    # gca().add_patch(patch);
     
    
    #plot Boundary Conditions-----
    if argv.setdefault('plot_boundary',False):
     for side in self.side_list['Boundary'] + self.side_list['Interface'] :
      p1 = self.sides[side][0]
      p2 = self.sides[side][1]
      n1 = self.nodes[p1]
      n2 = self.nodes[p2]
      plot([n1[0],n2[0]],[n1[1],n2[1]],color='#f77f0e',lw=12)
      #plot([n1[0],n2[0]],[n1[1],n2[1]],color='k',lw=1)
     
     
      #plot Periodic Conditions-----
    if argv.setdefault('plot_Periodic',False):
     for side in self.side_list['Periodic'] + self.side_list['Inactive']  :
      p1 = self.sides[side][0]
      p2 = self.sides[side][1]
      n1 = self.nodes[p1]
      n2 = self.nodes[p2]
      plot([n1[0],n2[0]],[n1[1],n2[1]],color='#1f77b4',lw=12,zorder=1)
     
    if argv.setdefault('plot_interface',True):
     for side in self.side_list['Interface']:
      p1 = self.sides[side][0]
      p2 = self.sides[side][1]
      n1 = self.nodes[p1]
      n2 = self.nodes[p2]
      plot([n1[0],n2[0]],[n1[1],n2[1]],color='#1f77b4',lw=2,zorder=1)
     



    
    #----------------------------
    axis('off')
    #data = {}
    if self.argv.setdefault('show',False):
     show()
    #if self.argv.setdefault('savefig',False) :
    # savefig(argv.setdefault('fig_file','geometry.png'))

    #if self.argv.setdefault('store_rgb',False):
    # fig.canvas.draw()   
    # rgb = np.array(fig.canvas.renderer._renderer)
    # data = {'rgb':rgb}
    # clf()
   #else: data = None
   #data =  mpi4py.MPI.COMM_WORLD.bcast(data,root=0)

   #if self.argv.setdefault('store_rgb',False):
   # self.rgb = data['rgb']



 def compute_triangle_area(self,p1,p2,p):
   p = Polygon([(p1[0],p1[1]),(p2[0],p2[1]),(p[0],p[1])])
   return p.area

 def point_in_elem(self,elem,p):

  poly = []
  for n in self.elems[elem]:
   t1 = self.nodes[n][0]
   t2 = self.nodes[n][1]
   poly.append((t1,t2))
  polygon = Polygon(poly)
  if polygon.contains(Point(p[0],p[1])):
   return True
  else:
   return False


 def compute_line_data(self,p1,p2,data):

    #get the first element---
    x = [0]
    elem = self.get_elem_from_point(p1) 
    value = self.compute_2D_interpolation(data,p1,elem)
    line_data = [value]

    int_points = []
    N = 100
    delta = np.linalg.norm(p2-p1)/N
    gamma = 1e-4
    r_old = p1
    for n in range(1,N+1):
     neighbors = self.get_elem_extended_neighbors(elem)  
     r = p1 + n*(p2-p1)/N
     tmp = self.cross_interface(r_old,r)
     if len(tmp) > 0:
       #print(tmp) 
       x.append(n*delta-np.linalg.norm(r-tmp)-gamma)
       x.append(n*delta-np.linalg.norm(r-tmp)+gamma)
       int_points.append(n*delta-np.linalg.norm(r-tmp))
       versor = r - r_old;versor /= np.linalg.norm(versor)
       elem = self.get_elem_from_point(tmp-gamma*versor,guess = neighbors) 
       value = self.compute_2D_interpolation(data,tmp-gamma*versor,elem)
       elem = self.get_elem_from_point(tmp+gamma*versor,guess = neighbors) 
       value = self.compute_2D_interpolation(data,tmp+gamma*versor,elem)
       line_data.append(value)
       line_data.append(value)
       neighbors = self.get_elem_extended_neighbors(elem)  
     else:
      elem = self.get_elem_from_point(r,guess = neighbors) 
      if  elem == -1:
          value = 0
      else:        
       #print(elem)
       value = self.compute_2D_interpolation(data,r,elem)
      line_data.append(value)
      x.append(n*delta)
     r_old = r

    
    return x,line_data,int_points 


 def get_elem_from_point(self,r,guess = []): 
   
    for elem in guess:
     if self.point_in_elem(elem,r) :
       return elem
   
    for elem in range(len(self.elems)):
     if self.point_in_elem(elem,r) :
       #print('no guess')  
       return elem

    return -1

 def compute_2D_interpolation(self,data,p,elem):

    if self.point_in_elem(elem,p) :
       
      nodes = self.elems[elem]
      p1 = self.nodes[nodes[0]]
      p2 = self.nodes[nodes[1]]
      p3 = self.nodes[nodes[2]]

      v1 = data[nodes[0]]
      v2 = data[nodes[1]]
      v3 = data[nodes[2]]

      area_1 = self.compute_triangle_area(p2,p3,p)
      area_2 = self.compute_triangle_area(p1,p3,p)
      area_3 = self.compute_triangle_area(p1,p2,p)
      area = area_1 + area_2 + area_3
      v = v1*area_1 + v2*area_2 + v3*area_3
      v /= area

      return v
   
    print('ERROR: no elem found')
    quit()


 def update_all_maps_from_side(self):
    
    #update_maps
    self.elem_side_map = {}
    for side in self.side_elem_map.keys():
     for elem in self.side_elem_map[side]:   
      self.elem_side_map.setdefault(elem,[]).append(side)
       
    #------------------------------------------------------
    self.node_side_map = {}
    for ns,side in enumerate(self.sides):
     for node in side:   
      self.node_side_map.setdefault(node,[]).append(ns)
    #------------------------------------------------------

    #-------------------
    self.node_elem_map = {}
    for node in self.node_side_map.keys():
      for side in self.node_side_map[node]:
        for elem in self.side_elem_map[side]:
          self.node_elem_map.setdefault(node,set()).add(elem)

    self.side_node_map = self.sides


 def adjust_boundary_elements(self):

  for side in self.side_list['Boundary'] + self.side_list['Hot'] + self.side_list['Cold']:
   self.side_elem_map[side].append(self.side_elem_map[side][0])


 def plot_elem(self,ne,color='gray') :

     elem = self.data['elems'][ne]
     
     pp = []
     for e in elem:
      pp.append(self.nodes[e][:2])
     path = create_path(pp)
     patch = patches.PathPatch(path,linestyle=None,linewidth=0.5,color=color,zorder=1,joinstyle='miter',alpha=0.7)
     gca().add_patch(patch);
     ave = np.sum(np.array(pp),axis=0)/4
     ave,dummy = self.compute_general_centroid_and_side(ne)

     return ave



 def build_interface(self,**argv):

    p = argv['ly']
    l = argv.setdefault('l',1)
    scale = argv.setdefault('scale',1)
    variance = argv.setdefault('variance',1)
    f = generate_random_interface(p,l,variance,scale)

    elem_mat_map = {}
    for ne in range(len(self.data['elems'])):
      c = self.compute_elem_centroid(ne)
      dd = f(c[1])
      
      if c[0] < dd:
       elem_mat_map.update({ne:0})
      else:
       elem_mat_map.update({ne:1})

    self.data['elem_mat_map'] = elem_mat_map

    ll = []
    for side in self.side_list['active']:
     elems = self.side_elem_map[side]
     k1 = self.data['elem_mat_map'][elems[0]]       
     k2 = self.data['elem_mat_map'][elems[1]]       
     if not k1 == k2:
      ll.append(side)
    self.side_list['Interface'] = ll
    #----------------------------

    self.compute_volume_fraction()
    

 def compute_volume_fraction(self):   

   v1 = 0
   v2 = 0
   for e in range(len(self.elems)):
    v = self.compute_elem_volume(e)   
    if self.elem_mat_map[e] == 0:
      v1 +=v
    else:  
      v2 +=v

   #self.fraction = v1/v2
   #print(v1,v2)   


 def add_patterning(self,elem_mat_map,lone=True):

  if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
 
    self.elem_mat_map = elem_mat_map  
    
    #change active elements---
    grid = list(elem_mat_map.keys())

    self.elem_list['active']= grid.copy()

    #----Restore previous boundary connections---
    #self.side_elem_map = self.side_elem_map_global.copy()
    #for side in self.side_list['Boundary']:
    #  (i,dummy) = self.side_elem_map[side]
    #  for elem in self.elem_map[i]:
    #      for s2 in self.elem_side_map[elem]:
    #          if s2 == side:
    #            self.side_elem_map[side][1] = elem
    #            break
    self.side_list['Boundary'] = []       
    self.side_list['active'] = self.side_list['active_global'].copy()
    #self.side_list['Periodic'] = self.side_list['Periodic_global'].copy()
    #------------------------------------------ 


    #eliminate lone pores---
    if lone:
     self.eliminate_lone_pores()
    #-------------------

    n = int(sqrt(len(self.nodes)))-1

    #elem maps
    self.nle = len(self.elem_list['active'])
    self.l2g = []
    self.g2l = np.zeros(len(self.elems),dtype=int)
    for i in self.elem_list['active']:
     self.l2g.append(i)
     self.g2l[i] = len(self.l2g)-1
    #----------------------------------         

    #active side---
    active_sides = set()
    for i in self.elem_list['active']:
     for side in self.elem_side_map[i]:
       active_sides.add(side)
    self.side_list['active'] = list(active_sides)   
    #boundary side---
    boundary_side = []
    for side in self.side_list['active']:
      elems = self.side_elem_map[side]
      if not ((elems[0] in self.elem_list['active']) and (elems[1] in self.elem_list['active'])):
        boundary_side.append(side)
        if elems[0] in self.elem_list['active']:
           self.side_elem_map[side] = [elems[0],elems[0]]
        elif elems[1] in self.elem_list['active']:
           self.side_elem_map[side] = [elems[1],elems[1]]
        else:
           print('error')
    self.side_list['Boundary'] = boundary_side

    #------------------------------------
    ll = []
    for side in self.side_list['active']:
     elems = self.side_elem_map[side]
     k1 = self.elem_mat_map[elems[0]]       
     k2 = self.elem_mat_map[elems[1]]       
     if not k1 == k2:
      ll.append(side)
      
    self.side_list['Interface'] = ll
    #----------------------------

    #remove boundary in periodic----
    for ll in self.side_list['Boundary']:
        if ll in self.side_list['Periodic']:
          self.side_list['Periodic'].remove(ll)
    #-------------------------------

    self.compute_boundary_condition_data()
    self.compute_connecting_matrix()
    self.compute_connecting_matrix_new()
   
    data = {'nle':self.nle,'g2l':self.g2l,'l2g':self.l2g,'side_elem_map':self.side_elem_map,'elem_side_map':self.elem_side_map,\
            'side_list':self.side_list,'region_elem_map':self.region_elem_map,\
            'elem_region_map':self.elem_region_map,'A':self.A,'CM':self.CM,\
            'B':self.B,'B_with_area_old':self.B_with_area_old,'B_area':self.B_area,\
            'CP':self.CP,'N':self.N,'N_new':self.N_new,'IB':self.IB,'BN':self.BN}

   # self._update_data()

  else: data = None



 def compute_dists(self):
  self.dists= {}   
  for ll in self.side_list['active']:
   if not (ll in (self.side_list['Boundary'] + self.side_list['Hot'] + self.side_list['Cold'])) :
    elem_1 = self.side_elem_map[ll][0]
    elem_2 = self.side_elem_map[ll][1]
    c1 = self.get_elem_centroid(elem_1)
    if elem_1 == elem_2:
     c2 = self.get_side_centroid(ll)
    else:    
     c2 = self.get_next_elem_centroid(elem_1,ll)
    dist = c2 - c1
    self.dists.setdefault(elem_1,{}).update({elem_2:dist})





 def compute_mesh_data(self,argv):


    self.import_mesh(argv)


    #self.adjust_boundary_elements()e
    self.compute_elem_map()
    self.compute_elem_volumes()
    self.compute_side_areas()
    self.compute_side_normals()
    self.compute_elem_centroids()
    self.compute_side_centroids()
    self.compute_least_square_weigths()
    self.compute_connecting_matrix()
    self.compute_connecting_matrix_new()
    self.compute_interpolation_weigths()
    self.compute_contact_areas()
    self.compute_dists()
    self.compute_boundary_condition_data(argv)
    self.n_elems = len(self.elems)
    self.nle = len(self.l2g)

    data = {'side_list':self.side_list,\
          'n_elems':np.array([self.n_elems]),\
          'elem_side_map':self.elem_side_map,\
          'side_elem_map':self.side_elem_map,\
          'elem_list':self.elem_list,\
          'elems':self.elems,\
          'dim':np.array([self.dim]),\
          'weigths':self.weigths,\
          'kappa_factor':np.array([self.kappa_factor]),\
          'areas':self.side_areas,\
          'periodic_values':self.periodic_values,\
          'interp_weigths':self.interp_weigths,\
          'centroids':self.elem_centroids,\
          'side_centroids':self.side_centroids,\
          'volumes':self.elem_volumes,\
          'B_with_area_old':self.B_with_area_old.todense(),\
          'B':self.B.todense(),\
          'normals':self.new_normals,\
          'dists':self.dists,\
          'flux_sides':self.flux_sides,\
          'i':self.i,\
          'j':self.j,\
          'k':self.k,\
          'ip':self.ip,\
          'jp':self.jp,\
          'dp':self.dp,\
          'eb':self.eb,\
          'sb':self.sb,\
          'db':self.db,\
          'dbp':self.dbp,\
          'pv':self.pv,\
          'n_side_per_elem':np.array([len(self.elems[0])])}

    return data


 def compute_connecting_matrix(self):

   A  = dok_matrix((self.nle,self.nle), dtype=float32)
   for ll in self.side_list['active'] :
    if not ll in self.side_list['Hot'] and\
     not ll in self.side_list['Cold'] and not ll in self.side_list['Boundary']:
     elems = self.get_elems_from_side(ll)
     kc1 = self.g2l[elems[0]]
     kc2 = self.g2l[elems[1]]
     A[kc1,kc2] = 1
     A[kc2,kc1] = 1
   self.A = A.tocsc()



 def compute_connecting_matrix_new(self):

   nc = len(self.l2g)
   N  = sparse.DOK((nc,nc,3), dtype=float32)

   ns = len(self.elem_side_map[0])
   CP  = sparse.DOK((nc,ns,3), dtype=float32) #this is a elem X n_side map of normal
   CM  = sparse.DOK((nc,ns,3), dtype=float32) #this is a elem X n_side map of normal

   N_new = sparse.DOK((nc,nc,3), dtype=float32)

   #self.CPB = np.zeros((nc,3))
   self.i = [];self.j = [];self.k = []
   self.eb = [];self.sb = []; self.db = []; self.dbp = []

   data = []

   
   for ll in self.side_list['active'] :
     elems = self.get_elems_from_side(ll)
     kc1 = elems[0]
     kc2 = elems[1]
     l1 = self.g2l[kc1]
     l2 = self.g2l[kc2]
     vol1 = self.get_elem_volume(kc1)
     vol2 = self.get_elem_volume(kc2)
     area = self.compute_side_area(ll)
     normal = self.compute_side_normal(kc1,ll)
     if not kc1 == kc2:
       N_new[l1,l2] = normal
       N_new[l2,l1] = -normal
       N[l1,l2] = normal*area/vol1
       N[l2,l1] = -normal*area/vol2
       
       #New development-----------------
       self.i.append(l1)
       self.j.append(l2)
       self.k.append(normal*area/vol1)
       self.j.append(l1)
       self.i.append(l2)
       self.k.append(-normal*area/vol2)
       #---------------------------------


       #modify in case of interfaces---
       #if ll in self.side_list['Interface']:
       # s1 = np.where(np.array(self.elem_side_map[kc1])==ll)[0][0]
       # s2 = np.where(np.array(self.elem_side_map[kc2])==ll)[0][0]
       # CM[l1,s1]  = normal*area/vol1
       # CM[l2,s2]  = -normal*area/vol1

        #new development---
       # self.eb.append(l1)
       # self.sb.append(s1)
       # self.db.append(normal*area/vol1)
       # self.eb.append(l2)
       # self.sb.append(s2)
       # self.db.append(-normal*area/vol1) #CHECK THIS
        #----------------------------

       #else: 
       # N[l1,l2] = normal*area/vol1
       # N[l2,l1] = -normal*area/vol2
       #--------------------------------

     else:
       s = np.where(np.array(self.elem_side_map[kc1])==ll)[0][0]
       CM[l1,s]  = normal*area/vol1
       self.eb.append(l1)
       self.sb.append(s)
       self.db.append(normal*area/vol1)

       self.dbp.append(normal)

       CP[l1,s] = normal
       a = self.get_elem_centroid(l1)
       #if ll in self.side_list['Boundary']:
       # scatter(a[0],a[1])

   #show()
   self.CP = CP.to_coo()
   self.CM = CM.to_coo()
   self.N = N.to_coo()
   self.N_new = N_new.to_coo()
   self.k = np.array(self.k).T
   self.db = np.array(self.db).T
   self.dbp = np.array(self.dbp).T



 def compute_elem_map(self):

  self.elem_map = {}
  for elem1 in self.elem_side_map:
    for side in self.elem_side_map[elem1]:
     elem2 = self.get_neighbor_elem(elem1,side)
     self.elem_map.setdefault(elem1,[]).append(elem2)

  #self.elem_list = {'active':list(range(len(self.elems)))}

 def get_side_orthognal_direction(self,side):

   elem = self.side_elem_map[side][0]
   c1 = self.get_elem_centroid(elem)
   c2 = self.get_side_centroid(side)
   normal = self.compute_side_normal(elem,side)
   area = self.compute_side_area(side)
   Af = area*normal
   dist = c2-c1
   v_orth = np.dot(Af,Af)/np.dot(Af,dist)
   return v_orth


 def get_distance_between_centroids_of_two_elements_from_side(self,ll):


   (elem_1,elem_2) = self.side_elem_map[ll]
   c1 = self.get_elem_centroid(elem_1)
   c2 = self.get_next_elem_centroid(elem_1,ll)
   dist = np.linalg.norm(c2-c1)

   return dist

 def get_decomposed_directions(self,elem_1,elem_2,rot = np.eye(3)):

    side = self.get_side_between_two_elements(elem_1,elem_2)
    normal = self.compute_side_normal(elem_1,side)
    area = self.compute_side_area(side)
    Af = area*normal
    c1 = self.get_elem_centroid(elem_1)

    if elem_1 == elem_2:
     c2 = self.get_side_centroid(side)
    else:    
     c2 = self.get_next_elem_centroid(elem_1,side)

    dist = c2 - c1

    v_orth = np.dot(normal,np.dot(rot,normal))/np.dot(normal,dist)
    v_non_orth = np.dot(rot,normal) - dist*v_orth
    return area*v_orth,area*v_non_orth


 def get_side_coeff(self,side):

  elem = self.side_elem_map[side][0]
  vol = self.get_elem_volume(elem)
  normal = self.compute_side_normal(elem,side)
  area = self.get_side_area(side)
  return normal*area/vol #This is specific to boundaries

 #def get_coeff(self,elem_1,elem_2):

 #  side = self.get_side_between_two_elements(elem_1,elem_2)
 #  vol = self.get_elem_volume(elem_1)
 #  normal = self.compute_side_normal(elem_1,side)
 #  area = self.compute_side_area(side)

 #  return normal*area/vol

 def compute_2D_adjusment(self,normal,phi_s,dphi):

  tmp =  np.arctan2(normal[1],normal[0])
  if not tmp == 0.0:
   tmp = (tmp + np.pi*(1.0-np.sign(tmp)))%(2.0*np.pi)

  phi_normal = (np.pi/2.0-tmp)%(2.0*np.pi)

  phi_right = (phi_normal+np.pi/2.0)%(2.0*np.pi)
  phi_left = (phi_normal-np.pi/2.0)%(2.0*np.pi)

  phi_1 = 0.0
  phi_2 = 0.0
  phi_left = round(phi_left,8)
  phi_right = round(phi_right,8)
  phi_normal = round(phi_normal,8)
  phi_s = round(phi_s,8)


  delta = 1e-7
  case = 0
  if phi_s <= phi_right +delta and phi_s > (phi_right - dphi/2.0)%(2.0*np.pi):
   phi_1 = phi_right
   phi_2 = (phi_s + dphi/2.0)%(2.0*np.pi)
   case = 1

  elif phi_s > phi_right + delta and phi_s < (phi_right + dphi/2.0)%(2.0*np.pi):
   phi_1 = (phi_s-dphi/2.0)%(2.0*np.pi)
   phi_2 = phi_right
   case = 2

  elif phi_s < phi_left - delta and phi_s > (phi_left - dphi/2.0)%(2.0*np.pi):
   phi_1 = phi_left
   phi_2 = (phi_s + dphi/2.0)%(2.0*np.pi)
   case = 3
   #print(phi_1 * 180.0/np.pi)
   #print(phi_2 * 180.0/np.pi)
   #print(phi_normal * 180.0/np.pi)
   #print(phi_s * 180.0/np.pi)



  elif phi_s >= phi_left - delta and phi_s < (phi_left + dphi/2.0)%(2.0*np.pi):
   phi_1 = (phi_s-dphi/2.0)%(2.0*np.pi)
   phi_2 = phi_left
   case = 4


  phi_dir_adj = [np.cos(phi_1) - np.cos(phi_2),np.sin(phi_2) - np.sin(phi_1),0.0]
  return np.array(phi_dir_adj),case


 def get_angular_coeff_old(self,elem_1,elem_2,angle):

  vol = self.get_elem_volume(elem_1)
  side = self.get_side_between_two_elements(elem_1,elem_2)
  normal = self.compute_side_normal(elem_1,side)
  area = self.compute_side_area(side)
  tmp = np.dot(angle,normal)
  cm = 0
  cp = 0
  cmb = 0
  cpb = 0

   
  if tmp < 0:
   if elem_1 == elem_2:  
    cmb = tmp*area/vol
   else:
    cm = tmp*area/vol
  else:
   cp = tmp*area/vol
   if elem_1 == elem_2:  
    cpb = tmp
  

  return cm,cp,cmb,cpb


 def get_angular_coeff(self,elem_1,elem_2,index):

  side = self.get_side_between_two_elements(elem_1,elem_2)
  vol = self.get_elem_volume(elem_1)
  normal = self.compute_side_normal(elem_1,side)
  for i in range(3):
   normal[i] = round(normal[i],8)
  area = self.compute_side_area(side)


  if self.dim == 2:
      
   p = index
   
   polar_int = self.dom['polar_dir'][p]*self.dom['polar_factor']
   tmp = np.dot(polar_int,normal)
   coeff  = tmp*area/vol
   #coeff = np.dot(angle_factor,normal)*area/vol
   #control  = np.dot(self.dom['polar_dir'][p],normal)
  else :
   t = int(index/self.dom['self.n_phi'])
   p = index%self.dom['n_phi']
   angle_factor = self.dom['S'][t][p]/self.dom['d_omega'][t][p]
   coeff = np.dot(angle_factor,normal)*area/vol

  #coeff = round(coeff,8)
  #anti aliasing---
  extra_coeff = 0.0
  extra_angle = 0.0
  case = 0
  if self.dim == 2:
   (phi_dir_adj,case) = self.compute_2D_adjusment(normal,round(self.dom['phi_vec'][p],8),self.dom['d_phi'])
   #extra_coeff = np.dot(phi_dir_adj,normal)*area/vol/self.dom['d_phi_vec'][p]
   #extra_angle = np.dot(phi_dir_adj,normal)*self.dom['d_phi']

  cm = 0.0
  cp = 0.0
  cmb = 0.0
  cpb = 0.0
  if coeff >= 0:
   cp = coeff - extra_coeff
   if not elem_1 == elem_2:
    cm = extra_coeff
   else:
    cpb = tmp*self.dom['d_phi'] - extra_angle
    cmb = extra_coeff
  else :
   cp = extra_coeff
   if not elem_1 == elem_2: #In this case the incoming part is from the boundary
    cm = coeff  - extra_coeff
   else:
    cmb = coeff - extra_coeff
    cpb = extra_angle


  if cpb < 0.0:
   print('cpb')
   print(cpb)

  if cmb > 0.0:
   print('cmb')
   print(cmb)

  if cm > 0.0:
   print('cm')
   print(cm)

  if cp < 0.0:#
     print('cp')
     print(cp)
     print(extra_coeff)
     print(case)
     print(coeff)
#     quit()

  return cm,cp,cmb,cpb


# def get_aa(self,elem_1,elem_2,phi_dir,dphi):

#   side = self.get_side_between_two_elements(elem_1,elem_2)
#   vol = self.get_elem_volume(elem_1)
#   normal = self.compute_side_normal(elem_1,side)

#   gamma = arcsin(np.dot(phi_dir,normal[0:2]))

#   if gamma >= 0.0:
     #p_plus = 0.5 + min([gamma,dphi/2.0])/dphi
     #p_minus = 1.0-p_plus


   #if gamma < 0.0:
#     p_minus = 0.5 + min([-gamma,dphi/2.0])/dphi
     #p_plus = 1.0-p_minus

  # return p_minus,p_plus



 def get_af(self,elem_1,elem_2):

   side = self.get_side_between_two_elements(elem_1,elem_2)
   normal = self.compute_side_normal(elem_1,side)
   area = self.compute_side_area(side)
   return normal*area


 def get_normal_between_elems(self,elem_1,elem_2):

   side = self.get_side_between_two_elements(elem_1,elem_2)
   normal = self.compute_side_normal(elem_1,side)
   return normal


 def get_side_between_two_elements(self,elem_1,elem_2):

   if elem_1 == elem_2: #Boundary side
    for side in self.elem_side_map[elem_1]:
     if side in self.side_list['Boundary'] + self.side_list['Hot'] + self.side_list['Cold']:
      return side
    print('No boundary side')
    quit()
   else:

    for side_1 in self.elem_side_map[elem_1]:
     for side_2 in self.elem_side_map[elem_2]:
      if side_1 == side_2:
       return side_1

    print('no adjacents elems')
    assert(1==0)
    quit()



 def compute_gradient_on_side(self,x,ll,side_periodic_value):

   #Diff in the temp
  
   elems = self.side_elem_map[ll]
   kc1 = elems[0]
   temp_1 = x[kc1]
   #Build tempreature matrix--------------------------
   diff_temp = np.zeros(self.dim + 1)
   grad = np.zeros(3)
   for s in self.elem_side_map[kc1]:
    if s in self.side_list['Boundary']:
     temp_2 = temp_1
    else:
     kc2 = self.get_neighbor_elem(kc1,s)
     temp_2 = x[kc2] + self.get_side_periodic_value(s,kc2,side_periodic_value)

    ind1 = self.elem_side_map[kc1].index(s)
    diff_t = temp_2 - temp_1
    diff_temp[ind1] = diff_t
    #-----------------------------------------------
    #COMPUTE GRADIENT
   tmp = np.dot(self.weigths[kc1],diff_temp)
   grad[0] = tmp[0] #THESE HAS TO BE POSITIVE
   grad[1] = tmp[1]
   if self.dim == 3:
    grad[2] = 0.0

   return grad



 def compute_thermal_conductivity_from_scalar(self,x,flux_sides,side_periodic_value,factor = np.eye(3)):

   #Diff in the temp
   kappa = 0.0
   for ll in flux_sides :
    normal = self.get_side_normal(0,ll)
    area_loc = self.get_side_area(ll)
    elems = self.side_elem_map[ll]
    kc1 = elems[0]
    temp_1 = x[kc1]
    #Build tempreature matrix--------------------------
    diff_temp = np.zeros(self.dim + 1)
    grad = np.zeros(self.dim)
    for s in self.elem_side_map[kc1]:
     if s in self.side_list['Boundary']:
      temp_2 = temp_1
     else:
      kc2 = self.get_neighbor_elem(kc1,s)
      temp_2 = x[kc2] + self.get_side_periodic_value(s,kc2,side_periodic_value)

     ind1 = self.elem_side_map[kc1].index(s)
     diff_t = temp_2 - temp_1
     diff_temp[ind1] = diff_t
    #-----------------------------------------------
    #COMPUTE GRADIENT
    tmp = np.dot(self.weigths[kc1],diff_temp)
    grad[0] = tmp[0] #THESE HAS TO BE POSITIVE
    grad[1] = tmp[1]
    if self.dim == 3:
     grad[2] = tmp[2]

    #COMPUTE THERMAL CONDUCTIVITY------------------
    for i in range(self.dim):
     for j in range(self.dim):
      kappa -= normal[i]*factor[i][j]*grad[j]*area_loc

   return kappa


 def compute_thermal_conductivity_outer(self,vector,scalar,flux_sides):

   #Here we simply invert Fourier's law
   factor = np.eye(3)
   power = 0.0
   for ns in flux_sides:
    ne = self.side_elem_map[ns][0]
    normal = self.get_side_normal(0,ns)
    area_loc = self.get_side_area(ns)
    power_loc = 0.0
    for i in range(self.dim):
     for j in range(self.dim):
      power_loc -= normal[i]*factor[i][j]*vector[j]*scalar[ne]*area_loc
    power += power_loc
   return power

 def compute_thermal_conductivity(self,grad,flux_sides,factor=np.eye(3)):

   #Here we simply invert Fourier's law
   power = 0.0
   for ns in flux_sides:
    ne = self.side_elem_map[ns][0]
    normal = self.get_side_normal(0,ns)
    area_loc = self.get_side_area(ns)
    power_loc = 0.0
    for i in range(3):
     for j in range(3):
      power_loc -= normal[i]* factor[i][j] * grad[ne][j]*area_loc
    power += power_loc

   return power



 def compute_average_temperature(self,x,flux_sides):

   #Here we simply invert Fourier's law
   temp = 0.0
   for ns in flux_sides:
    ne = self.side_elem_map[ns][0]
    area_loc = self.get_side_area(ns)
    temp += area_loc * x[ne]

   return temp


 def compute_contact_areas(self):

  self.c_areas = np.zeros(3)

  nodesT = self.nodes.copy().T

  minx = min(nodesT[0])
  maxx = max(nodesT[0])
  miny = min(nodesT[1])
  maxy = max(nodesT[1])
  minz = min(nodesT[2])
  maxz = max(nodesT[2])
  if self.dim == 3:
   self.c_areas[0] = (maxy - miny) * (maxz - minz)
   self.c_areas[1] = (maxz - minz) * (maxx - minx)
   self.c_areas[2] = (maxy - miny) * (maxx - minx)
  else :
   self.c_areas[0] = maxy - miny
   self.c_areas[1] = maxx - minx
   self.c_areas[2] = 0


 def get_side_area(self,side):
  return self.side_areas[side]

 def get_elem_volume(self,elem):
  return self.elem_volumes[elem]

 def compute_elem_volumes(self):

  self.elem_volumes = np.zeros(len(self.elems))
  for k in range(len(self.elems)):
   self.elem_volumes[k] = self.compute_elem_volume(k)


 def compute_side_areas(self):
  self.side_areas = np.zeros(len(self.sides))
  for k in range(len(self.sides)):
   self.side_areas[k] = self.compute_side_area(k)


 def get_dim(self):
   return self.dim


 def import_mesh(self,argv):


  #-------------

  f = open('mesh.msh','r')
  #Read boundary conditions------------------
  f.readline()
  f.readline()
  f.readline()
  f.readline()
  self.blabels = {}
  nb =  int(f.readline())
  for n in range(nb):
   tmp = f.readline().split()
   l = tmp[2].replace('"',r'')
   self.blabels.update({int(tmp[1]):l})

  #------------------------------------------
  self.elem_region_map = {}
  self.region_elem_map = {}

  #import nodes------------------------
  #f.readline()
  f.readline()
  f.readline()
  n_nodes = int(f.readline())
  nodes = []
  for n in range(n_nodes):
   tmp = f.readline().split()
   nodes.append([float(tmp[1]),float(tmp[2]),float(tmp[3])])

  nodes = np.array(nodes)
  if max(nodes[:,2])-min(nodes[:,2]) < 1e-4:
     self.dim = 2
  else:   
     self.dim = 3

  if self.dim == 3:
   self.size = np.array([ max(nodes[:,0]) - min(nodes[:,0]),\
                  max(nodes[:,1]) - min(nodes[:,1]),\
                  max(nodes[:,2]) - min(nodes[:,2])])
  else:
   self.size = np.array([ max(nodes[:,0]) - min(nodes[:,0]),\
                  max(nodes[:,1]) - min(nodes[:,1]),0])

  if (max(nodes[:,2]) - min(nodes[:,2])) < 1e-3:
     self.dim = 2
  else:   
     self.dim = 3


  self.nodes = nodes
  #import elements and create map
  self.side_elem_map = {}
  self.elem_side_map = {}
  f.readline()
  f.readline()
  n_tot = int(f.readline())
  self.sides = []
  self.elems = []
  b_sides = []
  nr = []

  self.node_side_map = {}
  self.side_node_map = {}
  self.node_elem_map = {}
  self.side_list = {}


  for n in range(n_tot):
   tmp = f.readline().split()

   #Get sides------------------------------------------------------------
   if self.dim == 3 and int(tmp[1]) == 2: #2D area
     n = sorted([int(tmp[5])-1,int(tmp[6])-1,int(tmp[7])-1])

   if self.dim == 2 and int(tmp[1]) == 1: #1D area
     n = sorted([int(tmp[5])-1,int(tmp[6])-1])

   b_sides.append(n)
   nr.append(int(tmp[3]))

   if self.dim == 3 and int(tmp[1]) == 4: #3D Elem
     node_indexes = [int(tmp[5])-1,int(tmp[6])-1,int(tmp[7])-1,int(tmp[8])-1]
     n = sorted(node_indexes)
     self.elems.append(n)
     perm_n =[[n[0],n[1],n[2]],\
             [n[0],n[1],n[3]],\
             [n[1],n[2],n[3]],\
             [n[0],n[2],n[3]]]
     self._update_map(perm_n,b_sides,nr,node_indexes)
     self.elem_region_map.update({len(self.elems)-1:self.blabels[int(tmp[3])]})
     self.region_elem_map.setdefault(self.blabels[int(tmp[3])],[]).append(len(self.elems)-1)

   if self.dim == 2 and int(tmp[1]) == 2: #2D Elem (triangle)
     node_indexes = [int(tmp[5])-1,int(tmp[6])-1,int(tmp[7])-1]
     n = sorted(node_indexes)
     self.elems.append(n)
     perm_n =[[n[0],n[1]],\
             [n[0],n[2]],\
             [n[1],n[2]]]
     self.elem_region_map.update({len(self.elems)-1:self.blabels[int(tmp[3])]})
     self.region_elem_map.setdefault(self.blabels[int(tmp[3])],[]).append(len(self.elems)-1)

     self._update_map(perm_n,b_sides,nr,node_indexes)

   if self.dim == 2 and int(tmp[1]) == 3: #2D Elem (quadrangle)
     n = [int(tmp[5])-1,int(tmp[6])-1,int(tmp[7])-1,int(tmp[8])-1]

     #n = sorted(node_indexes)
     self.elems.append(n)
     perm_n =[ sorted([n[0],n[1]]),\
              sorted([n[1],n[2]]),\
              sorted([n[2],n[3]]),\
              sorted([n[3],n[0]])]
     self._update_map(perm_n,b_sides,nr,n)

     self.elem_region_map.update({len(self.elems)-1:self.blabels[int(tmp[3])]})
     self.region_elem_map.setdefault(self.blabels[int(tmp[3])],[]).append(len(self.elems)-1)

  #------------------------------------------------------------
  #Set default for hot and cold
  self.side_list.setdefault('Hot',[])
  self.side_list.setdefault('Cold',[])


  #Apply Periodic Boundary Conditions
  self.side_list.update({'active':list(range(len(self.sides)))})
  self.side_periodicity = np.zeros((len(self.sides),2,3))
  group_1 = []
  group_2 = []
  self.exlude = []

  self.pairs = [] #global (all periodic pairs)

  self.side_list.setdefault('Boundary',[])
  self.side_list.setdefault('Interface',[])
  self.periodic_nodes = {}

  if argv.setdefault('delete_gmsh_files',False):
    os.remove(os.getcwd() + '/mesh.msh')
    os.remove(os.getcwd() + '/mesh.geo')

  for label in list(self.side_list.keys()):

   if str(label.split('_')[0]) == 'Periodic':
    if not int(label.split('_')[1])%2==0:
     contact_1 = label
     contact_2 = 'Periodic_' + str(int(label.split('_')[1])+1)
     group_1 = self.side_list[contact_1]
     group_2 = self.side_list[contact_2]
     pairs = []

     #compute tangential unity vector
     tmp = self.nodes[self.sides[group_2[0]][0]] - self.nodes[self.sides[group_2[0]][1]]
     t = tmp/np.linalg.norm(tmp)
     n = len(group_1)
     for s1 in group_1:
      d_min = 1e6
      for s in group_2:
       c1 = self.compute_side_centroid(s1)
       c2 = self.compute_side_centroid(s)
       d = np.linalg.norm(c2-c1)
       if d < d_min:
        d_min = d
        pp = c1-c2
        s2 = s

      pairs.append([s1,s2])
      self.side_periodicity[s1][1] = pp
      self.side_periodicity[s2][1] = -pp

      if np.linalg.norm(self.nodes[self.sides[s1][0]] - self.nodes[self.sides[s2][0]]) == np.linalg.norm(pp):
          self.periodic_nodes.update({self.sides[s1][0]:self.sides[s2][0]})
          self.periodic_nodes.update({self.sides[s1][1]:self.sides[s2][1]}) 
      else:
          self.periodic_nodes.update({self.sides[s1][0]:self.sides[s2][1]})
          self.periodic_nodes.update({self.sides[s1][1]:self.sides[s2][0]})

     plot_sides = False
     if plot_sides:
      for s in pairs:
       c1 = self.compute_side_centroid(s[0])
       c2 = self.compute_side_centroid(s[1])
       #plot([c1[0],c2[0]],[c1[1],c2[1]],color='r')
      #show()
     #Amend map
     for s in pairs:
      s1 = s[0]
      s2 = s[1]

      #Change side in elem 2
      elem2 = self.side_elem_map[s2][0]
      index = self.elem_side_map[elem2].index(s2)
      self.elem_side_map[elem2][index] = s1
      self.side_elem_map[s1].append(elem2)

      #To get hflux sides right-----
      self.side_elem_map[s2].append(self.side_elem_map[s1][0])

      #-------------------------
      #Delete s2 from active list
      self.side_list['active'].remove(s2)

      self.exlude.append(s2)
      tmp = self.side_list.setdefault('Inactive',[]) + [s2]
      self.side_list['Inactive'] = tmp

     #Polish sides
     tmp = self.side_list.setdefault('Periodic',[])+self.side_list[contact_1]
     self.side_list['Periodic'] = tmp

     del self.side_list[contact_1]
     del self.side_list[contact_2]
     self.pairs += pairs

  #Create boundary_elements--------------------

  #self.boundary_elements = []
  #self.interface_elements = []

  self.node_list = {}
  boundary_nodes = []
  for ll in self.side_list['Boundary']:
   #self.boundary_elements.append(self.side_elem_map[ll][0])
   for node in self.sides[ll]:
    if not node in boundary_nodes:
     boundary_nodes.append(node)
  self.node_list.update({'Boundary':boundary_nodes})

  interface_nodes = []
  #self.interface_elements = []
  for ll in self.side_list['Interface']:
   #self.interface_elements.append(self.side_elem_map[ll][0])
   #self.interface_elements.append(self.side_elem_map[ll][1])
   for node in self.sides[ll]:
    if not node in interface_nodes:
     interface_nodes.append(node)
  self.node_list.update({'Interface':interface_nodes})

  self.side_list.update({'active_global':self.side_list['active']})
  self.adjust_boundary_elements()

  self.elem_kappa_map = {}
  self.l2g=range(len(self.elems))
  self.g2l=range(len(self.elems))
  self.nle = len(self.elems)
  self.elem_list = {'active':list(range(len(self.elems)))}
  self.elem_mat_map = { ne:0 for ne in list(range(len(self.elems)))}

  #print(self.node_list['Interface'])
  #delete MESH-----
  #a=subprocess.check_output(['rm','-f','mesh.geo'])
  #a=subprocess.check_output(['rm','-f','mesh.msh'])

  #----------------


 def get_elems_from_side(self,ll):

  return self.side_elem_map[ll]


 def _update_map(self,perm_n,b_sides,nr,node_indexes):


    #UPDATE NODE_ELEM_MAP-------------------
    for n in node_indexes:
     self.node_elem_map.setdefault(n,[]).append(len(self.elems)-1)
    #---------------------------------------

    for new_n in perm_n :
     #Use the node_side_map---------
     index = -1
     sides = []
     for k in new_n:
      if index == -1:
       if k in self.node_side_map.keys():
        for s in self.node_side_map[k]:
         if np.allclose(self.sides[s],new_n):
          index = s
          break

     if index == -1:
      self.sides.append(new_n)
      index = len(self.sides)-1

      #add boundary--------------------------------------
      try:
       k = b_sides.index(new_n)

       self.side_list.setdefault(self.blabels[nr[k]],[]).append(index)
      except:
       a = 1
      #------------------------------------------------------
      for i in range(len(new_n)):
       self.node_side_map.setdefault(new_n[i],[]).append(index)
       self.side_node_map.setdefault(index,[]).append(new_n[i])

     #------------------------------------
     self.side_elem_map.setdefault(index,[]).append(len(self.elems)-1)
     self.elem_side_map.setdefault(len(self.elems)-1,[]).append(index)





 def get_elem_boundary_area_normal(self,elem) :

    sides = self.elem_side_map[elem]
    is_here = 0
    for s in sides:
     if s in self.side_list['Boundary'] :
      normal = self.compute_side_normal(elem,s)
      is_here = 1
      break
    if is_here == 0 : print('ERROR: side is not a boundary')
    return normal


 def compute_side_normals(self):

  self.new_normals = {}   
  self.side_normals = np.zeros((len(self.sides),2,3))
  for s in range(len(self.sides)):
   elems = self.side_elem_map[s]
   self.side_normals[s][0] = self.compute_side_normal(elems[0],s)
   self.new_normals.setdefault(elems[0],{}).update({elems[1]:self.compute_side_normal(elems[0],s)})
   if len(elems)>1:
    self.side_normals[s][1] = self.compute_side_normal(elems[1],s)
    self.new_normals.setdefault(elems[1],{}).update({elems[0]:self.compute_side_normal(elems[1],s)})

 def compute_side_normal(self,ne,ns):

  #Get generic normal--------------------
  v1 = self.nodes[self.sides[ns][1]]-self.nodes[self.sides[ns][0]]
  if self.dim == 3:
   v2 = self.nodes[self.sides[ns][2]]-self.nodes[self.sides[ns][1]]
  else :
   v2 = np.array([0,0,1])
  v = np.cross(v1,v2)
  normal = v/np.linalg.norm(v)
  #-------------------------------------

  #Check sign
  ind = self.side_elem_map[ns].index(ne)
  c_el   = self.compute_centroid(self.elems[ne])
  c_side = self.compute_centroid(self.sides[ns]) -  self.side_periodicity[ns][ind]

  c = (c_side - c_el)
  if np.dot(normal,c) < 0: normal = - normal


  return normal

 def get_n_elems(self):
  return len(self.elems)

 def compute_side_area(self,ns):

  p = self.nodes[self.sides[ns]]
  if self.dim == 2:
    return np.linalg.norm(p[1]-p[0])
  else:
   v = np.cross(p[1]-p[0],p[1]-p[2])
   normal = v/np.linalg.norm(v)
   tmp = np.zeros(3)
   for i in range(len(p)):
    vi1 = p[i]
    vi2 = p[(i+1)%len(p)]
    tmp += np.cross(vi1, vi2)

   result = np.dot(tmp,normal)
   return abs(result/2)


 def write_vtk(self,file_name,data) :

   points = self.nodes
   el = self.elems
   cells = []
   for k in range(len(el)):
    if self.dim == 3:
     cells.append(el[k][0:4])
    else:
     cells.append(el[k][0:3])

   if self.dim == 3:
    vtk = VtkData(UnstructuredGrid(points,tetra=cells),data)
   else:
    vtk = VtkData(UnstructuredGrid(points,triangle=cells),data)
   vtk.tofile(file_name,'ascii')


 def get_side_normal(self,ne,ns):
  return self.side_normals[ns,ne]


 def get_elem_centroid(self,ne):
  return self.elem_centroids[ne]

 def get_side_centroid(self,ns):
  return self.side_centroids[ns]



 def compute_side_centroids(self):

  self.side_centroids = np.zeros((len(self.sides),3))
  for s in range(len(self.sides)):
   self.side_centroids[s] = self.compute_side_centroid(s)

 def compute_side_centroid(self,ns):

  nodes = self.nodes[self.sides[ns]]

  centroid = np.zeros(3)
  for p in nodes:
   centroid += p
  return centroid/len(nodes)



 def compute_elem_centroids(self):

  self.elem_centroids = np.zeros((len(self.elems),3))
  for elem in range(len(self.elems)):
   self.elem_centroids[elem] = self.compute_elem_centroid(elem)


 def compute_elem_centroid(self,ne):
  nodes = self.nodes[self.elems[ne]]
  centroid = np.zeros(3)
  for p in nodes:
   centroid += p

  return centroid/len(nodes)

 def compute_elem_volume(self,kc1):

  if self.dim == 3: #Assuming Tetraedron
   ns = self.elems[kc1]
   m = np.ones((4,4))
   m[0,0:3] = self.nodes[ns[0]]
   m[1,0:3] = self.nodes[ns[1]]
   m[2,0:3] = self.nodes[ns[2]]
   m[3,0:3] = self.nodes[ns[3]]
   return abs(1.0/6.0 * np.linalg.det(m))

  if self.dim == 2: #Assuming Tetraedron

    
   points = self.nodes[self.elems[kc1]]
   x = []; y= []
   for p in points:
    x.append(p[0])
    y.append(p[1])

   return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


   return volume


 def get_side_periodic_value(self,elem,elem_p) :

    ll = self.get_side_between_two_elements(elem,elem_p)
    ind = self.side_elem_map[ll].index(elem)
    return self.side_periodic_value[ll][ind]


 def get_next_elem_centroid(self,elem,side):

  centroid1 = self.get_elem_centroid(elem)
  elem2 = self.get_neighbor_elem(elem,side)
  centroid2 = self.get_elem_centroid(elem2)
  ind1 = self.side_elem_map[side].index(elem)
  ind2 = self.side_elem_map[side].index(elem2)
  centroid = centroid2 - self.side_periodicity[side][ind1] + self.side_periodicity[side][ind2]
  return centroid

 def cross_interface(self,p1,p2):

  line1 = LineString([p1,p2])
  for ll in self.side_list['Interface']:
   p3 = self.nodes[self.sides[ll][0]][:2]
   p4 = self.nodes[self.sides[ll][1]][:2]
   line2 = LineString([p3,p4])
   tmp = line1.intersection(line2)
   if isinstance(tmp,shapely.geometry.Point):
     return np.array([tmp.x,tmp.y])
  return []


 def compute_centroid(self,side):

  node = np.zeros(3)
  for p in side:
   node += self.nodes[p]

  return node/len(side)

 def get_elem_extended_neighbors(self,elem):
  #This also includes the element get_distance_between_centroids_of_two_elements_from_side
  neighbors = []
  for n in self.elems[elem]:
   for elem2 in self.node_elem_map[n]:
    if not elem2 in neighbors: neighbors.append(elem2)
    #for m in self.elems[elem2]:
    # for elem3 in self.node_elem_map[m]:
    #    if not elem3 in neighbors: neighbors.append(elem3)

  return neighbors

 def get_elem_neighbors(self,elem):

  neighbors = []
  for ll in self.elem_side_map[elem]:
   neighbors.append(self.get_neighbor_elem(elem,ll))
  return neighbors

 def get_neighbor_elem(self,elem,ll) :

    if not (elem in self.side_elem_map[ll]) : print('error, no neighbor',ll,elem)

    for tmp in self.side_elem_map[ll] :
       if not (tmp == elem) :
         return tmp


 def compute_interpolation_weigths(self):

  self.interp_weigths = {}# np.zeros(len(self.sides))
  for ll in self.side_list['active']:
   if not (ll in (self.side_list['Boundary'] + self.side_list['Hot'] + self.side_list['Cold'])) :
    e0 = self.side_elem_map[ll][0]
    e1 = self.side_elem_map[ll][1]
    
    P0 = self.get_elem_centroid(e0)
    P1 = self.get_next_elem_centroid(e0,ll)
    #---------------------------------------------------------------
    if self.dim == 3:
     #from here: http://geomalgorithms.com/a05-_intersect-1.html
     u = P1 - P0
     n = self.get_side_normal(1,ll)
     node = self.nodes[self.sides[ll][0]]
     w = P0 - node
     s = -np.dot(n,w)/np.dot(n,u)
    else: #dim = 2
     P2 = self.nodes[self.sides[ll][0]]
     P3 = self.nodes[self.sides[ll][1]]
     den = (P3[1] - P2[1])*(P1[0]-P0[0])-(P3[0]-P2[0])*(P1[1]-P0[1])
     num = (P3[0] - P2[0])*(P0[1]-P2[1])-(P3[1]-P2[1])*(P0[0]-P2[0])
     a = num/den
     P = P0 + a * (P1-P0)
     if a < 0.0 or a > 1.0 :
      print(ll)
      print(P0)
      print(P1)
      print(e0)
      print(e1)
      print(P2)
      print(P3)
      print('ERROR in the skew parameter')
      return
     dist = np.linalg.norm(P1-P0)
     d = np.linalg.norm(P - P1)
     s = d/dist
     #---------------------------------------------------------------
    self.interp_weigths.update({ll:[s]})
   else:
    self.interp_weigths.update({ll:[1.0]})


 def get_region_from_elem(self,elem):
  return self.elem_region_map[elem]


 def get_interpolation_weigths(self,ll,elem):

  #return self.interp_weigths[ll]
  if self.side_elem_map[ll][0] == elem:
   return self.interp_weigths[ll]
  else:
   return 1.0-self.interp_weigths[ll]

 def compute_boundary_condition_data(self,argv):

    direction = argv.setdefault('direction','x')
    if direction == 'x':
     gradir = 0
     applied_grad = [1,0,0]
    if direction == 'y': 
     applied_grad = [0,1,0]
     gradir = 1
    if direction == 'z':
     applied_grad = [0,0,1]
     gradir = 2

    if gradir == 0:
     flux_dir = [1,0,0]
     length = self.size[0]

    if gradir == 1:
     flux_dir = [0,1,0]
     length = self.size[1]

    if gradir == 2:
     flux_dir = [0,0,1]
     length = self.size[2]

    delta = 1e-2
    nsides = len(self.sides)
    side_value = np.zeros(nsides)

    tmp = self.side_list.setdefault('Periodic',[]) + self.exlude + self.side_list.setdefault('Hot',[]) + self.side_list.setdefault('Cold',[])

    for kl,ll in enumerate(tmp) :
     normal = self.compute_side_normal(self.side_elem_map[ll][0],ll)
     tmp = np.dot(normal,flux_dir)
     #if tmp < -delta : #either negative or positive
     # flux_sides.append(ll)

     if tmp < - delta :
        side_value[ll] = -1.0
     if tmp > delta :
        side_value[ll] = +1.0
     
     if ll in self.side_list['Hot']:
       side_value[ll] = -0.5

     if ll in self.side_list['Cold']:
       side_value[ll] = +0.5


    side_periodic_value = np.zeros((nsides,2))
    self.periodic_values = {}

    n_el = self.nle
    B = sparse.DOK((n_el,n_el),dtype=float32)

    B_with_area_old = sparse.DOK((n_el,n_el),dtype=float32)
    self.B_area = np.zeros(n_el,dtype=float32)
    
    self.ip = []; self.jp = []; self.dp = []; self.pv = [] 
    if len(self.side_list.setdefault('Periodic',[])) > 0:
     for side in self.pairs:

      area = self.get_side_area(side[0])
      
      side_periodic_value[side[0]][0] = side_value[side[0]]
      side_periodic_value[side[0]][1] = side_value[side[1]]

      elem1,elem2 = self.side_elem_map[side[0]]

      self.periodic_values.update({elem1:{elem2:[side_value[side[0]]]}})
      self.periodic_values.update({elem2:{elem1:[side_value[side[1]]]}})

      i = self.g2l[elem1]
      j = self.g2l[elem2]

      normal = self.compute_side_normal(i,side[0])
      
      voli = self.get_elem_volume(i)       
      volj = self.get_elem_volume(j)       
 

      B[i,j] = side_value[side[0]]
      B[j,i] = side_value[side[1]]
      self.ip.append(i); self.jp.append(j); self.dp.append(side_value[side[0]]); self.pv.append(normal*area/voli)
      self.ip.append(j); self.jp.append(i); self.dp.append(side_value[side[1]]); self.pv.append(-normal*area/volj)

      if np.linalg.norm(np.cross(self.get_side_normal(1,side[0]),applied_grad)) < 1e-5:
       B_with_area_old[i,j] = abs(side_value[side[0]]*area)
       
   
    #select flux sides-------------------------------
    self.flux_sides = [] #where flux goes
    for ll in self.side_list['Periodic']:
     normal = self.compute_side_normal(self.side_elem_map[ll][0],ll)
     tmp = np.abs(np.dot(normal,flux_dir))
     if tmp > delta : #either negative or positive
       self.flux_sides.append(ll)
    #-----------------------------------------------


    #In case of fixed temperature----
    self.BN = np.zeros((n_el,len(self.elems[0]),3))
    self.IB = np.zeros((n_el,len(self.elems[0])))
    self.eft = []; self.sft = []; self.dft = []
    for ll in self.side_list['Hot'] + self.side_list['Cold']:
      elems = self.side_elem_map[ll]
      normal = self.get_side_normal(0,ll)
      ind = self.elem_side_map[elems[0]].index(ll)
      self.BN[elems[0],ind] = normal * self.get_side_area(ll)
      area = self.get_side_area(ll)
      vol = self.get_elem_volume(elems[0])

      self.eft.append(elems[0])
      self.sft.append(ind)
      self.dft.append(normal*area/vol)

      #This is treated similarly to boundary data

      if ll in self.side_list['Hot']:
        self.IB[elems[0],ind] =  0.5
       # self.dft.append(0.5*normal*area/vol)
      else :
        self.IB[elems[0],ind] = -0.5
        self.B_area[elems[0]] = area
       # self.dft.append(-0.5*normal*area/vol)
        
    #----------------------------------------------------------------
    self.area_flux = abs(np.dot(flux_dir,self.c_areas))
    self.side_periodic_value = side_periodic_value
    self.B = B.to_coo()
    self.B_with_area_old = B_with_area_old.to_coo()
    self.pv = np.array(self.pv).T
    self.dft = np.array(self.dft).T
    self.kappa_factor = self.size[gradir]/self.area_flux
  

 def compute_least_square_weigths(self):

   nd = len(self.elems[0])
   #diff_dist = np.zeros((len(self.elems),nd,self.dim))
   diff_dist = {}

   for ll in self.side_list['active_global'] :
    elems = self.side_elem_map[ll]
    kc1 = elems[0]
    c1 = self.compute_elem_centroid(kc1)
    ind1 = self.elem_side_map[kc1].index(ll)
    if not ll in (self.side_list['Boundary'] + self.side_list['Hot'] + self.side_list['Cold'] + self.side_list['Interface']):
     #Diff in the distance
     kc2 = elems[1]
     c2 = self.get_next_elem_centroid(kc1,ll)
     
     ind2 = self.elem_side_map[kc2].index(ll)
     dist = c2-c1

     for i in range(self.dim):
      diff_dist.setdefault(kc1,np.zeros((len(self.elem_side_map[kc1]),self.dim)))[ind1][i] = dist[i]   
      diff_dist.setdefault(kc2,np.zeros((len(self.elem_side_map[kc2]),self.dim)))[ind2][i] = -dist[i]   
      #diff_dist[kc1][ind1][i] = dist[i]
      #diff_dist[kc2][ind2][i] = -dist[i]
    else :
     dist = self.compute_side_centroid(ll) - c1
     for i in range(self.dim):
      #diff_dist[kc1][ind1][i] = dist[i]
      diff_dist.setdefault(kc1,np.zeros((len(self.elem_side_map[kc1]),self.dim)))[ind1][i] = dist[i]   
      
     if ll in self.side_list['Interface']:# or self.side_list['Boundary']:
      kc2 = elems[1]
      c2 = self.get_next_elem_centroid(kc1,ll)
      dist = self.compute_side_centroid(ll) - c2
      ind2 = self.elem_side_map[kc2].index(ll)
      for i in range(self.dim):
        #diff_dist[kc2][ind2][i] = dist[i]
        diff_dist.setdefault(np.zeros((len(self.elem_side_map[kc2]),self.dim)))[ind2][i] = dist[i]   


   #Compute weights
   self.weigths = {}
   for h in diff_dist.keys() :
    tmp = diff_dist[h]   
    self.weigths[h] = np.dot(np.linalg.inv(np.dot(np.transpose(tmp),tmp)),np.transpose(tmp)  )



   #self.weigths = []
   #for tmp in diff_dist :
   # print(tmp)   
   # self.weigths.append(np.dot(np.linalg.inv(np.dot(np.transpose(tmp),tmp)),np.transpose(tmp)  ))


 def compute_interfacial_node_temperature(self,temp,kappa):
     
     int_temp_matrix = {}
     int_temp_inclusion = {}
     for ll in self.side_list['Interface'] :

       (i,j) = self.side_elem_map[ll]
       w = self.get_interpolation_weigths(ll,i)
       Ti = temp[i]
       Tj = temp[j]
       ki = kappa[i]
       kj = kappa[j]
       Tb = (kj*w*Tj + ki*(1-w)*Ti)/(kj*w + ki*(1-w))

       (n1,n2) = self.side_node_map[ll]

       int_temp_matrix[n1] = int_temp_matrix.setdefault(n1,0) + Tb/2
       int_temp_matrix[n2] = int_temp_matrix.setdefault(n2,0) + Tb/2

       int_temp_inclusion[n1] = int_temp_inclusion.setdefault(n1,0) + Tb/2
       int_temp_inclusion[n2] = int_temp_inclusion.setdefault(n2,0) + Tb/2

     return {'inclusion':int_temp_inclusion,'matrix':int_temp_matrix}

 def compute_interfacial_temperature_side(self,temp,kappa):
     
     int_temp = {}
     
     for ll in self.side_list['Interface'] :

       (i,j) = self.side_elem_map[ll]
       w = self.get_interpolation_weigths(ll,i)
       Ti = temp[i]
       Tj = temp[j]
       ki = kappa[i]
       kj = kappa[j]
       Tb = (kj*w*Tj + ki*(1-w)*Ti)/(kj*w + ki*(1-w))

       int_temp[ll] = Tb

     return int_temp


 def compute_divergence(self,data,add_jump=True,verbose=0) :

  div = np.zeros(len(self.elems))     
  for n,d in enumerate(data.T):
    div += self.compute_grad(d,add_jump = add_jump,verbose=verbose).T[n]

  return div


 def compute_grad(self,temp,lattice_temp =[],add_jump=True,verbose=0,pbcs=None,interfacial_temperature = None) :

   if len(lattice_temp) == 0: lattice_temp = temp

   nd = len(self.elems[0])

   #if self.dim == 2:
   # diff_temp = np.zeros((self.nle,nd))
   #else:
   # diff_temp = np.zeros((self.nle,4))

  # diff_temp = self.elems.copy()


   diff_temp = len(self.elems)*[None]
   for i in range(len(diff_temp)):
      diff_temp[i] = len(self.elems[i])*[None] 

   gradT = np.zeros((self.nle,3))

   #Diff in the temp
   for ll in self.side_list['active'] :
   #for ll in range(len(self.sides))  :

    elems = self.side_elem_map[ll]
    kc1 = elems[0]
    c1 = self.get_elem_centroid(kc1)
    ind1 = self.elem_side_map[kc1].index(ll)

    if not ll in (self.side_list['Boundary'] + self.side_list['Hot'] + self.side_list['Cold'] + self.side_list['Interface']) :
     kc2 = elems[1]
     ind2 = self.elem_side_map[kc2].index(ll)
     temp_1 = temp[self.g2l[kc1]]
     temp_2 = temp[self.g2l[kc2]]
     if add_jump: 
       temp_2 += self.get_side_periodic_value(kc2,kc1) 
     else:
       temp_2 += pbcs[kc1,kc2]        

     diff_t = temp_2 - temp_1
     diff_temp[self.g2l[kc1]][ind1]  = diff_t
     diff_temp[self.g2l[kc2]][ind2]  = -diff_t
    else :
     if ll in self.side_list['Hot'] :
      diff_temp[kc1][ind1]  = 0.5-temp[kc1]

     if ll in self.side_list['Cold'] :
      diff_temp[kc1][ind1]  = -0.5-temp[kc1]

     if ll in self.side_list['Boundary'] : 
      diff_temp[self.g2l[kc1]][ind1]  = lattice_temp[self.g2l[kc1]]-temp[self.g2l[kc1]]

     if ll in self.side_list['Interface'] : 
      kc2 = elems[1]
      ind2 = self.elem_side_map[kc2].index(ll)
      if interfacial_temperature == None:
       Tb1 = temp[kc1]
       Tb2 = temp[kc2]
      else:    
       Tb1 = interfacial_temperature[ll][0]
       Tb2 = interfacial_temperature[ll][1]

      diff_temp[self.g2l[kc1]][ind1]  = Tb1 - temp[self.g2l[kc1]]
      diff_temp[self.g2l[kc2]][ind2]  = Tb2 - temp[self.g2l[kc2]]


   for k in range(self.nle) :
       
    tmp = np.dot(self.weigths[self.l2g[k]],diff_temp[k])
    gradT[k][0] = tmp[0] #THESE HAS TO BE POSITIVE
    gradT[k][1] = tmp[1]
    if self.dim == 3:
     gradT[k][2] = tmp[2]

   return gradT
