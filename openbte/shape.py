import numpy as np
import math
from .utils import *


def get_shape(argv):

    
    #Vectorize shape if needed
    shape = argv.setdefault('shape','square') 
    base = argv.setdefault('base',[[0,0]]) 
    if type(shape) == list:
     assert(len(shape)==len(base))
     shapes_str = shape
    else:
     shapes_str = len(base)*[shape]   

    #Vectorize custom shapes if needed
    n_custom = 0
    for shape in shapes_str:
        if shape == 'custom':
           n_custom +=1

    if n_custom > 0:
      if type(argv['shape_function']) == list:
       assert(len(argv['shape_function'])==len(argv['base']))
      else:
       argv['shape_function'] = len(argv['base'])*[argv['shape_function']]   
    #-----------------------
           

    #compute area weigths
    argv.setdefault('area_ratio',np.ones(len(argv['base'])))

    if argv.setdefault('relative',True):
     areas =  [argv['porosity']* i /sum(argv['area_ratio'])  for i in argv['area_ratio']]
    else: 
     areas = argv['area_ratio']

    #-------------------
    shapes = []
    n_custom = 0
    for n,shape in enumerate(shapes_str):

     area= areas[n]

     if shape == 'square':
       shapes.append(np.array(make_polygon(4,area)))

     if shape == 'triangle':
       shapes.append(np.array(make_polygon(3,area)))

     elif shape == 'circle':
       shapes.append(np.array(make_polygon(24,area)))

     elif shape == 'custom':

      options = {key:value[n_custom] if isinstance(value,list) else value for key,value in argv.setdefault('shape_options',{}).items()}

      options.update({'area':area})

      shapes.append(argv['shape_function'][n_custom](**options))
      n_custom +=1


    return shapes  






def get_smoothed_square(**argv):

     
     smooth = argv['smooth']
     area = argv['area']
     Na = argv['Na']
     L = np.sqrt(area+smooth*smooth*(4-np.pi))

     p = []
     dphi = math.pi/Na/2;
     for ka in range(Na+1):
      ph =  dphi * ka
      px  = L/2 - smooth + smooth * math.cos(ph)
      py  = L/2 - smooth + smooth * math.sin(ph)
      p.append([px,py])

     dphi = math.pi/Na/2;
     for ka in range(Na+1):
      ph = np.pi/2 +  dphi * ka
      px  = - L/2 + smooth + smooth * math.cos(ph)
      py  = + L/2 - smooth + smooth * math.sin(ph)
      p.append([px,py])

     dphi = math.pi/Na/2;
     for ka in range(Na+1):
      ph = np.pi +  dphi * ka
      px  = - L/2 + smooth + smooth * math.cos(ph)
      py  = - L/2 + smooth + smooth * math.sin(ph)
      p.append([px,py])

     dphi = math.pi/Na/2;
     for ka in range(Na+1):
      ph = 1.5*np.pi +  dphi * ka
      px  = + L/2 - smooth + smooth * math.cos(ph)
      py  = - L/2 + smooth + smooth * math.sin(ph)
      p.append([px,py])

     return p







