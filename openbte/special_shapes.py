import numpy as np
import math



def make_polygon(x,y,**options):
 
   Na = options['Na']
   A = options['area']
   dphi = 2.0*math.pi/Na;
   r = math.sqrt(2.0*A/Na/math.sin(2.0 * math.pi/Na))
   poly_clip = []
   for ka in range(Na):
     ph =  dphi/2 + (ka-1) * dphi
     px  = x + r * math.cos(ph) 
     py  = y + r * math.sin(ph) 
     poly_clip.append([px,py])

   return poly_clip  



def get_smoothed_square(cx,cy,**argv):

     
     smooth = argv['smooth']
     area = argv['area']
     Na = argv['Na']
     L = np.sqrt(area+smooth*smooth*(4-np.pi))

     p = []
     dphi = math.pi/Na/2;
     for ka in range(Na+1):
      ph =  dphi * ka
      px  = cx + L/2 - smooth + smooth * math.cos(ph)
      py  = cy + L/2 - smooth + smooth * math.sin(ph)
      p.append([px,py])

     dphi = math.pi/Na/2;
     for ka in range(Na+1):
      ph = np.pi/2 +  dphi * ka
      px  = cx - L/2 + smooth + smooth * math.cos(ph)
      py  = cy + L/2 - smooth + smooth * math.sin(ph)
      p.append([px,py])

     dphi = math.pi/Na/2;
     for ka in range(Na+1):
      ph = np.pi +  dphi * ka
      px  = cx - L/2 + smooth + smooth * math.cos(ph)
      py  = cy - L/2 + smooth + smooth * math.sin(ph)
      p.append([px,py])

     dphi = math.pi/Na/2;
     for ka in range(Na+1):
      ph = 1.5*np.pi +  dphi * ka
      px  = cx + L/2 - smooth + smooth * math.cos(ph)
      py  = cy - L/2 + smooth + smooth * math.sin(ph)
      p.append([px,py])

     return p







