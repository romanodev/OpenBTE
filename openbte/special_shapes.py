import numpy as np
import math

def get_smoothed_square(cx,cy,area,**argv):

     smooth = argv['smooth']
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

