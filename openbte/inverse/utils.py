from scipy.ndimage import gaussian_filter
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource
from itertools import product, combinations

import matplotlib as mpl

mpl.rcParams['toolbar'] = 'None'
#x = np.load('x',allow_pickle=True)

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

def add_cube(ax,N):

 delta = 0.05   
 vertsc = (delta+1)*N*np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0],\
          [0,0,1],[0,1,1],[1,1,1],[1,0,1]])

 vertsc -=delta/2*N
 facesc = np.array([[0,4,7,3],[1,5,6,2],[0,1,5,4],[3,2,6,7],[4,7,6,5],[0,1,2,3]])

 mesh2 = Poly3DCollection(vertsc[facesc],alpha=0.1,linewidth=1,facecolor='w',edgecolor='w')
 ax.add_collection3d(mesh2)



def init_plot_3D(ax,N):

 #add axes
 add_axes(ax)
 #add_cube(ax,N)
 ax.set_xlim(-6, N)  
 ax.set_ylim(-6, N)  
 ax.set_zlim(-6, N)  
 ax.axis('off')
 
 ax.set_box_aspect([1,1,1])
 plt.tight_layout()
 plt.draw()
 #plt.ion()
 #plt.show()

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




def plot_structure_3D(x,ax,replicate=False,draw_cube=False):

 #ax   = fig.add_subplot(111, projection='3d')
 #binarize
 #x = np.where(x>0.5,1.0,0.0)

 #Set an automatic level

 #print(level)

 N  = round(np.power(x.shape[0],1/3))

 x = x.reshape((N,N,N))
 x = np.swapaxes(x,0,2)
 x = np.swapaxes(x,0,1)
 if replicate:
  x = np.pad(x,N,mode='wrap')

 x = np.pad(x,pad_width=1,mode='wrap')[1:,1:,1:]
 x = np.pad(x,pad_width=1,mode='constant',constant_values=(0))


 if replicate:
  x = gaussian_filter(x, sigma=0.5)
  verts, faces, normals, values = measure.marching_cubes(x, level = 0.93)
 else: 
  level = np.mean(x[x>0.5])-1e-3
  verts, faces, normals, values = measure.marching_cubes(x, level = level)

 verts -= np.ones(3)[np.newaxis,:]


 # Fancy indexing: `verts[faces]` to generate a collection of triangles
 mesh = Poly3DCollection(verts[faces],alpha=1.0,linewidth=0,edgecolor='white')
 #print(len(ax.collections))
 if len(ax.collections) == 5:
    ax.collections.pop()
    ax.collections.pop()
 ax.add_collection3d(mesh)

 if not replicate:
    add_cube(ax,N)

 #print(x) 
 #ax.voxels(x,facecolors='gray')

 mesh.set_facecolor(lighting(normals,faces))


 #ADD ARROW in 3D

 if replicate:
  ax.set_xlim(0, 3*N)  
  ax.set_ylim(0, 3*N)  
  ax.set_zlim(0, 3*N)  

 plt.draw()
 #plt.pause(0.2)


def plot3D(x,ax,**options):

    try :
     plot_structure_3D(x,ax,**options)
    except ValueError:
        pass
    except RuntimeError:
        pass



