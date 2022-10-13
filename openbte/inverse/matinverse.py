from jax import custom_vjp
from jax.tree_util import tree_structure
from sqlitedict import SqliteDict
import nlopt
import numpy as np
#import shelve
import os
import time
import nlopt
from functools import partial,update_wrapper
import jax
from jax import custom_jvp
from jax import value_and_grad as value_and_grad
from jax import  jit
import scipy
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
import scipy
import scipy.sparse.linalg as spla
import scipy.sparse as sp
import warnings
import matplotlib.pylab as plt
from pathlib import Path
import shutil
#from quart import websocket, json
import json
from jax import custom_vjp
import os.path
import asyncio
import matplotlib
#from utils import * 
import warnings
from functools import lru_cache
import matplotlib as mpl
#from .visualization import *

from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
    csr_matrix, isspmatrix, dok_matrix, lil_matrix, bsr_matrix)
warnings.simplefilter('ignore',SparseEfficiencyWarning)

mpl.interactive(True)

mpl.rcParams['toolbar'] = 'None'

#jax.config.update('jax_platform_name', 'gpu')
jax.config.update("jax_log_compiles",0)
jax.config.update("jax_enable_x64",1)

#@jax.jit
def reflect_2D(a):
 N  = jnp.sqrt(a.shape[0]).astype(int)
 a = a.reshape((N,N))
 c = jnp.concatenate((a,jnp.fliplr(a)),axis=1)
 return jnp.concatenate((c,jnp.flipud(c)),axis=0).flatten()

def reflect_3D(x):

 N  = round(np.power(x.shape[0],1/3))
 x = x.reshape((N,N,N))

 x = jnp.concatenate((x,jnp.fliplr(x)),axis=1)
 x = jnp.concatenate((x,jnp.flipud(x)),axis=0)
 x = jnp.concatenate((x,jnp.flip(x,axis=2)),axis=2)

 return x.flatten()




@jax.jit
def LineSpace(gsquare,filtered_field,projected_field,options_LS):

  c     = options_LS['c']

  eta_d = options_LS['eta_d']

  indicator =  (1-projected_field)*jnp.exp(-c*gsquare)

  s = indicator.shape         

  return jnp.power((eta_d-filtered_field).clip(a_max=0),2)


@jax.jit
def convolve_3D(x,y):

   N  = round(np.power(x.shape[0],1/3))
   x = x.reshape((N,N,N))

   a1 = jnp.pad(x,N,mode='wrap')

   return jax.scipy.signal.convolve(a1,y,mode='same')[N:2*N,N:2*N,N:2*N].flatten()

   #quit()

   #return convolve2d(a1,y,mode='same')[N:2*N,N:2*N].flatten()

@jax.jit
def convolve_2D(x,y):

   N  = int(np.sqrt(x.shape[0]))
   x  = x.reshape((N,N))

   a1 = jnp.pad(x,N,mode='wrap')

   return convolve2d(a1,y,mode='same')[N:2*N,N:2*N].flatten()


@jax.jit
def filtering_2D(design_field,conic_kernel):

   return convolve_2D(design_field,conic_kernel)


@jax.jit
def filtering_3D(design_field,conic_kernel):

   return convolve_3D(design_field,conic_kernel)


@jax.jit
def projecting(filtered_field,beta):

   #Projection options---
   eta = 0.5

   return (jnp.tanh(beta*eta) + jnp.tanh(beta*(filtered_field-eta)))/(jnp.tanh(beta*eta) + jnp.tanh(beta*(1-eta)))



@jax.jit
def mapping_3D(design_field,beta,conic_kernel):
   #Filtering--
   filtered_field = filtering_3D(design_field,conic_kernel)

   return projecting(filtered_field,beta)

@jax.jit
def mapping_2D(design_field,beta,conic_kernel):

   #Filtering--
   filtered_field = filtering_2D(design_field,conic_kernel)

   return projecting(filtered_field,beta)

def conic_filter_3D(centroids,R):

    N = round(np.power(len(centroids),1/3))

    tmp = jnp.sqrt(jnp.power(centroids[:,0],2) + jnp.power(centroids[:,1],2)+  jnp.power(centroids[:,2],2))/R

    conic_kernel = jnp.where(tmp<1,1-tmp,0).reshape((N,N,N))

    if not (jnp.sum(conic_kernel) == 0):
      conic_kernel /= jnp.sum(conic_kernel)

    return conic_kernel

def conic_filter_2D(centroids,R):

    N = jnp.array(jnp.sqrt(len(centroids)),int)

    tmp = jnp.sqrt(jnp.power(centroids[:,0],2) + jnp.power(centroids[:,1],2))/R

    conic_kernel = jnp.where(tmp<1,1-tmp,0).reshape((N,N))

    if not (jnp.sum(conic_kernel) == 0):
      conic_kernel /= jnp.sum(conic_kernel)

    return conic_kernel


def generate_correlated_pores(**argv):

 #---------------READ Parameters-------
 N = argv['N']
 #L = argv['l']
 #d = L/N
 #p = L
 le  = argv['length']
 phi = argv['porosity']
 #------------------------------------
 #NOTE: Variance is 1 because we only take the smallest number up to a certain point
 #gen = [np.exp(-2/le/le * np.sin(d*np.pi*i/p) ** 2) for i in range(N)]
 gen    = [np.exp(-2/le/le * np.sin(np.pi*i/N) ** 2) for i in range(N)]
 kernel = np.array([[  gen[abs(int(s/N) - int(t/N))]*gen[abs(s%N-t%N)]  for s in range(N*N)] for t in range(N*N)])
 y = np.random.multivariate_normal(np.zeros(N*N), kernel)
 h = y.argsort()
 idxl = h[0:int(N*N*(1-phi))]

 x = np.ones(N*N)

 x[idxl] = 0

 return x



def import_evolution(name):

   name = os.getcwd() + '/' + name 
   data = []
   with shelve.open(name, 'r') as shelf:
     for key,value in shelf.items(): 
       data.append(value)  

   return data


def get_guess(**options):

 grid  = options['grid']
 L  = options['L']
 dim  = options.setdefault('dim',2)
 N    = int(grid**dim)
 #Number of elements--- 
 centroids = get_grid(grid,dim)['centroids']*L
 #---------------------
 model = options.setdefault('guess','random')

 if model == 'load':
    final = np.load('x',allow_pickle=True)

 elif model =='evolution':

    data =  import_evolution(options['name'])
    final = data[-1][0]

 elif model =='solid':
    final = np.ones(N)

 elif model =='gaussian':

  #options = {'porosity':0.2,'N':N,'length':2}
  

  options['N'] = grid
  
  final = 1-generate_correlated_pores(**options)

 elif model =='random':
  
    if dim == 3:
     final = np.random.rand(int(N/8))
     final = reflect_3D(final)
    if dim == 2: 
     final = np.random.rand(int(N/4))
     final = reflect_2D(final)
    #np.array(final).dump('x')
    #quit()
    #final = np.random.rand(N)




 #elif model =='random_smoothed':

 #   x = np.random.rand(int(M/4))
 #   x = reflect(x)
 #   final =  transform(x,1e8)

 elif model in ['staggered','aligned']:

    if options.setdefault('dim',2) == 3:

      C = L/2*np.array([[ 0,0 ,0],\
           [-1,-1,-1],\
           [-1, 1, 1],\
           [ 1,-1, 1],\
           [ 1, 1,-1],\
           [-1,-1, 1],\
           [-1, 1,-1],\
           [ 1,-1,-1],\
           [ 1, 1, 1]])

      phi = options['phi']

      r = L*(phi*3/8/np.pi)**(1.0/3.0) 

      vec = np.zeros(len(centroids))
      for c in C:
       tmp  =  jnp.power((centroids[:,0]-c[0])/r,2) + jnp.power((centroids[:,1]-c[1])/r,2) + jnp.power((centroids[:,2]-c[2])/r,2)
       vec +=  jnp.where(tmp<1,1,0)
      
      final = 1-vec

      return final


    else:    

      phi = options['phi']
      vec = np.zeros(len(centroids))
      radii_ratio = options.setdefault('radii_ratio',1)
   
      C = [[0,0],[L/2,L/2],[L/2,-L/2],[-L/2,-L/2],[-L/2,L/2]]
      r = L*np.sqrt(phi/np.pi/radii_ratio/2)

      vec = np.zeros(len(centroids))
      for c in C:
       tmp = jnp.power((centroids[:,0]-c[0])/r,2) + jnp.power((centroids[:,1]-c[1])/r,2)
       vec +=  jnp.where(tmp<1,1,0)

      final = 1-vec

      return final

 if options.setdefault('show',False):
     plot_structure(final,**{'replicate':True,'invert':True,'unitcell':True,'color_unitcell':'r','write':True})

 #if options.setdefault('save',False):
 #    final.dump('x')

 return final


def plot_2D(x,**options_plot_structure):
 
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

 vmax = options_plot_structure.setdefault('max',np.max(x))

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



def init_state(name,N,monitor,dim):

   def remove_file(filename):
       filepath = filename
       if os.path.exists(filepath):
         os.remove(filepath)

   #init database---------
   remove_file(name)
   with SqliteDict(name,autocommit=True) as db:
        db['dim'] = dim
        db['x']   = []

   if monitor:
    fig  = plt.figure(figsize=(6,6),num='Evolution')
    if dim == 2:
     ax   = fig.add_subplot(111)
     #im = plot_structure(np.ones(N*N),**{'blocking':False,'invert':True,'replicate':True,'unitcell':True,'color_unitcell':'c'})  
     im = plot_structure_2D(np.ones(N*N),blocking=False,invert=True,replicate=True,unitcel=True,color_unitcell='c',headless=True)  

    if dim == 3:
      plt.ion()
      ax   = fig.add_subplot(111, projection='3d')
      init_plot_3D(ax,N)

   def update(data):
       if monitor:
        if dim == 3:
         plot3D(data,ax,draw_cube=True)

        if dim == 2:
         x = data.reshape((N,N)).T
         x = np.pad(x,N,mode='wrap')
         im.set_data(1-x)

        plt.show()
        plt.pause(0.2)

       #async def send_message():
       #  await asyncio.sleep(0.1)
       #  await websocket.send(json.dumps(x.tolist()))
       #asyncio.run(send_message())  
       #print(data)

       #with SqliteDict(name,autocommit=True) as db:
       # db[str(len(db))] = data

   return update


def init_optimizer(x,objective,N,n_betas,tol,max_iter,maps_regular,update,name,filtering,resolution,**kwargs):

    #N = int(N/4)

    min_beta = 1
    betas = [2**(n+min_beta) for n in range(n_betas)]
    betas.append(1e24)
    
    #betas = [2]
    #TESTING
    #Prepare functions----
    maps = maps_regular

    #betas[-1] =  1e17
    M = len(x)
    for k,beta in enumerate(betas):

      if k == len(betas) -1 and len(betas) > 1:
           max_iter = 1
      print('beta: ',beta)   
      #if beta == -1:#last step  
      #   maps = maps_binary
      #else:   
      #   maps = maps_regular

      opt = nlopt.opt(nlopt.LD_CCSAQ,N)
      opt.set_lower_bounds(np.zeros(N))
      opt.set_upper_bounds(np.ones(N))


      #Make this special for dumping
      objective.objective=True
      #----
      #Set up the mapping
      opt.set_min_objective(enhance_function(objective,maps,update,beta,name))

      #user-defined inequalities
      for inequality in kwargs.setdefault('inequality_constraints',[]):
        inequality.objective=False
        opt.add_inequality_constraint(enhance_function(inequality,maps,update,beta,name),tol)

      #Add commonly-used inequalities
      if 'min_porosity' in kwargs.keys():
          func = partial(min_porosity,phi=kwargs['min_porosity'],N=N)
          func.objective=False
          update_wrapper(func,min_porosity)
          opt.add_inequality_constraint(enhance_function(func,maps,update,beta,name),tol)
        
      #Add commonly-used inequalities
      if 'max_porosity' in kwargs.keys():
          func = partial(max_porosity,phi=kwargs['max_porosity'],N=N)
          func.objective=False
          update_wrapper(func,max_porosity)
          opt.add_inequality_constraint(enhance_function(func,maps,update,beta,name),tol)

      #Add commonly-used inequalities
      if 'minimum_linewidth' in kwargs.keys():
          func = partial(minimum_linewidth,c=200,eta_e = 0.75,beta=beta,resolution = resolution)
          func.objective=False
          update_wrapper(func,minimum_linewidth)
          #Here the transformation is only filtering. Note that y is passed to mimic beta
          opt.add_inequality_constraint(enhance_function(func,lambda x,y: filtering(x),update,beta,name),tol)

      print(max_iter,beta)
      opt.set_maxeval(max_iter)
      #opt.set_stopval(1e-3) #Objective function
      #opt.set_ftol_abs(1e-4) #Relative on the objective function
      #opt.set_xtol_rel(1e-5) #Relative tol on x
      x = opt.optimize(x)

    print(' ')
    print("TOP OPT DONE")
    print(' ')
    plt.ioff()
    plt.show()
    return maps(x,beta)  


#@jax.jit
def minimum_linewidth(x,c,eta_e,beta,resolution):

 #x is filtered field  
 gsquare = (resolution*jnp.linalg.norm(jnp.gradient(x),axis=0))**2

 Is = projecting(x,beta)*jnp.exp(-c*gsquare)

 Lw = jnp.mean(Is * jnp.minimum(x - eta_e, 0)**2)

 return Lw,Lw


def min_porosity(x,phi,N):

    Vd = (1-phi)*N

    V = jnp.sum(x)
    return -(Vd-V),(1-V/N,)

def max_porosity(x,phi,N):

    Vd = (1-phi)*N

    V = jnp.sum(x)

    return Vd-V,(1-V/N,)

def enhance_function(func,transform,update,beta,name):
    """Make func compatible with NlOpt"""

    #@lru_cache
    def objective_optimizer(x,grad):
          
          def wrapper(x):
              
           x = transform(x,beta)
          
           return func(x)

          (g,aux),grad[:]    = jax.value_and_grad(wrapper,has_aux=True)(x)

          mapped_x = transform(x,beta)

          #Write evolution
          with SqliteDict(name,autocommit=True) as db:
                  x = db['x']
                  x.append(mapped_x)
                  db['x'] = x

                  tmp = db.setdefault(func.__name__,[])
                  tmp.append(aux)
                  db[func.__name__] = tmp
          update(mapped_x)        
                    

          print(func.__name__,'  ',aux[0])

          return float(g)

    return objective_optimizer  


def get_2D(N):


    def maps(i,j):

        return (i%N)*N + j%N
    #Compute indices for adiacent elements
    i_mat = []
    j_mat = []
    normals = []
    k  = 0
    for i in range(N):
     for j in range(N):

         k1,k2 = maps(i,j),maps(i+1,j)

         i_mat.append(k1)    
         j_mat.append(k2)    
         normals.append([1,0])
         k +=1

         k2 = maps(i,j+1)
         i_mat.append(k1)    
         j_mat.append(k2)    
         normals.append([0,-1])
         k +=1

         k2 = maps(i,j-1)
         i_mat.append(k1)    
         j_mat.append(k2)    
         normals.append([0,1])
         k +=1

         k2 = maps(i-1,j)
         i_mat.append(k1)    
         j_mat.append(k2)    
         normals.append([-1,0])
         k +=1
         
    #These relationships are found heuristically      

    kr = 4*N*N-4*N + 4*np.arange(N)  
    ku = 2+4*N*np.arange(N)
    kd = 4*N-3 +4*N*np.arange(N)
    kl = 3+4*np.arange(N)

    #calculate centroids--
    x = np.linspace(-1/2+1/(2*N),1/2-1/(2*N),N)
    centroids = np.array([ [i,j] for i in x for j in x[::-1]])
    #--------------------
    ind_extremes = [[kl,kr],[ku,kd]]

    return {'i':np.array(i_mat),'j':np.array(j_mat),'ind_extremes':np.array(ind_extremes,dtype=int),'normals':np.array(normals),'centroids':centroids}

def get_3D(N):

    #calculate centroids--
    x = np.linspace(-1/2+1/(2*N),1/2-1/(2*N),N)
    centroids = np.array([ [i,j,k] for k in x for i in x for j in x[::-1]])

    #plot centroids to double check
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #for c in centroids:
    #    ax.scatter(c[0],c[1],c[2],marker='o')
    #ax.axis('off')
    #plt.show()    

    def maps(i,j,k):

        return  (k%N)*N*N  +  (i%N)*N + j%N 

    #Compute indices for adiacent elements
    i_mat = []
    j_mat = []
    normals = []
    l  = 0
    for i in range(N):
     for j in range(N):
      for k in range(N):

         k0 = maps(i,j,k)

         kp = maps(i+1,j,k)
         i_mat.append(k0);j_mat.append(kp); l+=1
         normals.append([1,0,0])

         kp = maps(i-1,j,k)
         i_mat.append(k0);j_mat.append(kp); l+=1
         normals.append([-1,0,0])

         kp = maps(i,j+1,k)
         i_mat.append(k0);j_mat.append(kp); l+=1
         normals.append([0,-1,0])

         kp = maps(i,j-1,k)
         i_mat.append(k0);j_mat.append(kp); l+=1
         normals.append([0,1,0])

         kp = maps(i,j,k+1)
         i_mat.append(k0);j_mat.append(kp); l+=1
         normals.append([0,0,1])

         kp = maps(i,j,k-1)
         i_mat.append(k0);j_mat.append(kp); l+=1
         normals.append([0,0,-1])



    kr  = 6*N*N*N - 6*N*N + 6 * np.arange(N*N)
    kl  = 1 + 6 * np.arange(N*N)

    kd  = np.zeros(0,dtype=int)
    for i in range(N):
        kd = np.hstack((kd, (i+1)*6*N*N - 6*N + 6 * np.arange(N) + 2))

    ku  = np.zeros(0,dtype=int)
    for i in range(N):
        ku = np.hstack((ku,i*6*N*N+3+6*np.arange(N)))
    
    ka  = -2+6*N*(np.arange(N*N) +1)

    kb  = 5+6*N*(np.arange(N*N))


    ind_extremes = [[kl,kr],[ku,kd],[ka,kb]]

    #return {'i':np.array(i_mat),'j':np.array(j_mat),'kr':kr,'ku':ku,'kd':kd,'kl':kl,'ka':ka,'kb':kb,'normals':np.array(normals),'centroids':centroids}
    return {'i':np.array(i_mat),'j':np.array(j_mat),'ind_extremes':np.array(ind_extremes,dtype=int),'normals':np.array(normals),'centroids':centroids}

def get_grid(N,dimension):
    """Provides facilities for matrix assembly"""
 
    if dimension == 2:
       return get_2D(N)
    else:
       return get_3D(N)


def cachable(func):
    "This functions is a cache with size 1"

    def wrapper(x):
 
      if not wrapper.current_x == None:
        if jnp.allclose(x,wrapper.current_x):
         return wrapper.cached_results

      wrapper.current_x = x

      results = func(x)

      wrapper.cached_results = results

      return results

    wrapper.current_x = None

    return wrapper



def compose(func):
    #A function that adds a vustom vjp to a solver

    fdiff = custom_vjp(cachable(func))

    def f_fwd(pt):

     return fdiff(pt)

    def f_bwd(jac, v):

     return (jnp.dot(v[0],jac),) #vJp

    fdiff.defvjp(f_fwd, f_bwd)

    return fdiff


def optimize(objective,**kwargs):
   
    #Patse options---
    grid = kwargs['grid']
    dim  = kwargs.setdefault('dim',2)
    L    = kwargs['L']
    centroids = get_grid(grid,dim)['centroids']*L
    R    = kwargs['R']
    n_betas      = kwargs.setdefault('n_betas',10)
    max_iter     = kwargs.setdefault('maxiter',25)
    tol          = kwargs.setdefault('tol',1e-3)
    name         = kwargs.setdefault('output_file','output.db')
    monitor      = kwargs.setdefault('monitor',False)
    resolution = grid/L #1/Delta X


    if dim == 2:
     N    = int(grid**2.0)
     conic_kernel         = conic_filter_2D(centroids,R)
     filtering_map        = partial(filtering_2D,conic_kernel=conic_kernel)
     transform            = jax.jit(partial(mapping_2D,conic_kernel=conic_kernel))
    else: 
     N    = int(grid**3.0)
     conic_kernel         = conic_filter_3D(centroids,R)
     filtering_map        = partial(filtering_3D,conic_kernel=conic_kernel)
     transform            = jax.jit(partial(mapping_3D,conic_kernel=conic_kernel))


    #-------------------------

    #Get the updater
    update = init_state(name,grid,monitor,dim)
    #----------------

    #Init optimizer---------------------------------
    optimizer = partial(init_optimizer,N=N,n_betas=n_betas,tol=tol,max_iter=max_iter,\
                        maps_regular=transform,\
                        update=update,\
                        filtering = filtering_map,\
                        resolution = resolution,\
                        name=name)
    #-----------------------------------------------

    #Init guess generator--
    guess = partial(get_guess,grid=grid,L=L,dim=dim)
    #-----------------------

    x = guess(**kwargs)
    
    #x = guess(model='staggered',save=False,phi=0.2,show=False,dim=dim)
    #x = guess(model='random',save=False,show=False)
    #np.array(x).dump('x')
    #x = np.load('x',allow_pickle=True)
    #x = guess(model='solid',save=False,show=False)

    #Launch simulations---
    return optimizer(x,objective,**kwargs)


def load_projected_evolution(name):

   name = os.getcwd() + '/' + name +'/bte'
   output = []
   with shelve.open(name, 'r') as shelf:
     for key,value in shelf.items(): 
        output.append(np.where(np.asarray(value[0])>0.5,1,0))

   return output

def load_evolution(name):

   name = os.getcwd() + '/' + name +'/bte'
   output = []
   with shelve.open(name, 'r') as shelf:
     for key,value in shelf.items(): 
        output.append(value[0])

   return output



