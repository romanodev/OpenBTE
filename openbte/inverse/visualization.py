from scipy.ndimage import gaussian_filter
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from sqlitedict import SqliteDict
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource
#from .publish import *
from itertools import product, combinations
import shelve
import plotly.graph_objs as go
from jax import numpy as jnp
import matplotlib as mpl
#from .plot_cubes import *

mpl.rcParams['toolbar'] = 'None'

def init_plot_3D(ax,N):

 #add axes
 add_axes(ax)
 #add_cube(ax,N)
 ax.set_xlim(-2, N)
 ax.set_ylim(-2, N)
 ax.set_zlim(-2, N)
 ax.axis('off')

 ax.set_box_aspect([1,1,1])
 plt.tight_layout()
 plt.draw()


def lighting(normals,faces):

 ls = LightSource(azdeg=30.0, altdeg=0.0)
 normalsarray = np.array([np.array((np.sum(normals[face[:], 0]/3), np.sum(normals[face[:], 1]/3), np.sum(normals[face[:], 2]/3))/np.sqrt(np.sum(normals[face[:], 0]/3)**2 + np.sum(normals[face[:], 1]/3)**2 + np.sum(normals[face[:], 2]/3)**2)) for face in faces])

 min = np.min(ls.shade_normals(normalsarray, fraction=1.0)) # min shade value
 max = np.max(ls.shade_normals(normalsarray, fraction=1.0)) # max shade value
 diff = max-min
 newMin = 0.3
 newMax = 0.95
 newdiff = newMax-newMin

 # Using a constant color, put in desired RGB values here.
 colourRGB = np.array((255.0/255.0, 54.0/255.0, 57/255.0, 1.0))

 # The correct shading for shadows are now applied. Use the face normals and light orientation to generate a shading value and apply to the RGB colors for each face.
 return np.array([colourRGB*(newMin + newdiff*((shade-min)/diff)) for shade in ls.shade_normals(normalsarray, fraction=1.0)])


def add_cube(ax,N):

 delta = 0.05   
 vertsc = (delta+1)*N*np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0],\
          [0,0,1],[0,1,1],[1,1,1],[1,0,1]])

 vertsc -=delta/2*N
 facesc = np.array([[0,4,7,3],[1,5,6,2],[0,1,5,4],[3,2,6,7],[4,7,6,5],[0,1,2,3]])

 mesh2 = Poly3DCollection(vertsc[facesc],alpha=0.1,linewidth=1,facecolor='w',edgecolor='w')
 ax.add_collection3d(mesh2)


def load_optimized(name='output.db'):
 """Load last structure"""

 with SqliteDict(name) as db:

      data = db['x'][-1]

      print(1-jnp.sum(data)/len(data))

      dim  = db['dim']
 
 return data,dim

def load_x():

    return np.load('output.db',allow_pickle=True),3



def plot_2D(x):

        plot_structure_2D(x,replicate=True,write=False)

def plot(filename='output.db'):

    structure,dim = load_optimized(filename)
    #structure,dim = load_x()
    if dim == 3: 
        plot_structure_3D(structure,replicate=True,plotly=True)
        #plot_cubes(structure)
    else: 
        plot_structure_2D(x,replicate=True,write=False)

    return np.array(structure)


def add_axes(ax):

 qx = -2
 qy = -2
 qz = -5
 d =  4
 ax.quiver(qx, qy, qz,d,0,0, length=d, normalize=True)
 ax.quiver(qx, qy, qz,0,d,0, length=d, normalize=True)
 ax.quiver(qx, qy, qz,0,0,d, length=d, normalize=True)
 ax.text(d+qx, qy, qz, 'x',fontsize=14)
 ax.text(qx, d+qy,qz, 'y',fontsize=14)
 ax.text(qx, qy,d+qz, 'z',fontsize=14)


def plot_structure_2D(x,**options_plot_structure):
 
 N  = len(x)   
 Ns = int(jnp.sqrt(len(x)))
 N2 = int(N/2)
 x  = np.array(x)
 x  = x.reshape((Ns,Ns)).T
 
 if not options_plot_structure.setdefault('headless',False):
    fig  = plt.figure(figsize=(6,6),num='Evolution',frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    #ax   = fig.add_subplot(111)


 if options_plot_structure.setdefault('transpose',False):
     x = x.T

 if options_plot_structure.setdefault('replicate',True):
  x = jnp.pad(x,Ns,mode='wrap')
 if options_plot_structure.setdefault('invert',True):
  x = 1-x   

 cmap =  options_plot_structure.setdefault('colormap','gray')
 #cmap =  options_plot_structure.setdefault('colormap','inferno')

 vmax = options_plot_structure.setdefault('max',np.max(x))

 #x = np.array(x)
 #x[x> 0.5] = 0.3
 #im = plt.imshow(x,cmap=cmap,vmin=0,vmax=1,animated=True)
 if options_plot_structure.setdefault('normalize','binary') == 'binary':
    im = plt.imshow(x,vmin=0,vmax=1,cmap=cmap,animated=True)
 else:   
    im = plt.imshow(x,vmin=np.min(x),vmax=np.max(x),cmap=cmap,animated=True)

 #Apply mask
 if 'mask' in options_plot_structure.keys():
    x = options_plot_structure['mask']
    x = x.reshape((Ns,Ns)).T
    x = jnp.pad(x,Ns,mode='wrap')
    masked = np.ma.masked_where(x > 0.5, 1-x)
    if options_plot_structure.setdefault('invert_mask',False): masked = 1-masked
    plt.imshow(masked,cmap='gray',vmin=0,vmax=1);

 if  options_plot_structure.setdefault('unitcell',True):
     dx = 0.5

     plt.plot([Ns-0.5,Ns-0.5,2*Ns-0.5,2*Ns-0.5,Ns-0.5],[Ns-0.5,2*Ns-0.5,2*Ns-0.5,Ns-0.5,Ns-0.5],color=options_plot_structure.setdefault('color_unitcell','c'),ls='--')

 plt.axis('off')

 if options_plot_structure.setdefault('save',False):
  ax.set_xlim([0,3*Ns])
  ax.set_ylim([0,3*Ns])
  plt.savefig('figure.png',dpi=600,bbox_inches='tight')   

 plt.tight_layout(pad=0,h_pad=0,w_pad=0)
 if options_plot_structure.setdefault('blocking',True):
  plt.ioff()
  plt.show()
  
 else: 
  plt.ion()
  plt.show()
  plt.pause(0.1)

 return im


def plot_structure_3D(x,ax=None,replicate=False,draw_cube=False,show=True,plotly=False):
  
 N= round(np.power(x.shape[0],1/3))

 x = x.reshape((N,N,N))
 x = np.swapaxes(x,0,2)
 x = np.swapaxes(x,0,1)

 c = 6
 x = x[c:-c,c:-c,c:-c]

 if replicate:
  x = np.pad(x,N,mode='wrap')

 
 x = np.pad(x,pad_width=1,mode='wrap')[1:,1:,1:]
 x = np.pad(x,pad_width=1,mode='constant',constant_values=(0))

 if replicate:
  x = gaussian_filter(x, sigma=0.1)
  verts, faces, normals, values = measure.marching_cubes(x, level = 0.98)
 else: 
  level = np.mean(x[x>0.5])-1e-3
  verts, faces, normals, values = measure.marching_cubes(x, level = level)
  #x     = jnp.where(x>0.5,1,0)
  #verts, faces, normals, values = measure.marching_cubes(x, level = 0.999)

 verts -= np.ones(3)[np.newaxis,:]

 if plotly:

     fig = go.Figure()
     fig.add_trace(go.Mesh3d(x=verts[:,0],
                             y=verts[:,1],
                             z=verts[:,2],
                             color='red',
                             intensitymode='vertex',i=faces[:,0],j=faces[:,1],k=faces[:,2],
                              ))

     fig.update_layout(
     paper_bgcolor='rgba(0,0,0,0)',
     plot_bgcolor='rgba(0,0,0,0)')
    

     axis = dict(
                         backgroundcolor="rgb(0,0,0)",
                         gridcolor="white",
                         showbackground=False,
                         visible=False,
                         showticklabels=False,
                         zerolinecolor="white",)

     fig.update_xaxes(range=[0, N])
     fig.update_yaxes(range=[0, N])
     fig.update_layout(scene = dict(
                    xaxis = axis,yaxis = axis,
                    zaxis = axis))


     fig.show(config= dict(
            displayModeBar = False))
 else:

  if ax==None:
   fig = plt.figure(figsize=(10, 10))
   ax  = fig.add_subplot(111, projection='3d')
   init_plot_3D(ax,N)
   #ax.axis('off')

  mesh = Poly3DCollection(verts[faces],alpha=1.0,linewidth=0,edgecolor='white')
  if len(ax.collections) == 5:
    ax.collections.pop()
    ax.collections.pop()
  ax.add_collection3d(mesh)

  if not replicate:
    add_cube(ax,N)

  mesh.set_facecolor(lighting(normals,faces))

  #ADD ARROW in 3D

  if replicate:
   ax.set_xlim(0, 3*N)  
   ax.set_ylim(0, 3*N)  
   ax.set_zlim(0, 3*N)  


  if show:
      plt.show()
  else:    
      plt.draw()

