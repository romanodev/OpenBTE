import numpy as np
import math



def get_shape(argv):


    area  = argv.setdefault('porosity')/len(argv['base'])
    
    shape = argv.setdefault('shape','square')
    if shape == 'square':
       return np.array(make_polygon(4,area))

    elif shape == 'circle':
       return np.array(make_polygon(24,area))

    elif shape == 'custom':
      options = argv.setdefault('shape_options',{})
      options.update({'area':area})
      return argv['shape_function'](options)



def make_polygon(Na,A):
 
   dphi = 2.0*math.pi/Na;
   r = math.sqrt(2.0*A/Na/math.sin(2.0 * math.pi/Na))
   poly_clip = []
   for ka in range(Na):
     ph =  dphi/2 + (ka-1) * dphi
     px  = r * math.cos(ph) 
     py  = r * math.sin(ph) 
     poly_clip.append([px,py])

   return poly_clip  



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







