import numpy as np
import math
from matplotlib.pylab import *
from matplotlib.path import Path
import matplotlib.patches as patches
from pyclipper import *
from scipy import stats


def from_vector_to_periodic_polygons(vector,options):

 #Transform from vector to polygons and add periodic ones

 #Get radius------------------------------
 Np = 16
 Na = options['Na']
 Lx = options['Lx']*options['Nx']
 Ly = options['Ly']*options['Ny']
 area = Lx*Ly*options['porosity']/Np
 r = math.sqrt(2.0*area/Na/math.sin(2.0*math.pi/options['Na']))


 #frame-----------------------------------
 frame = []
 frame.append([float(-Lx)/2,float(Ly)/2])
 frame.append([float(Lx)/2,float(Ly)/2])
 frame.append([float(Lx)/2,float(-Ly)/2])
 frame.append([float(-Lx)/2,float(-Ly)/2])


 #Get pores---------------------------------------------
 polygons = []
 for n in range(len(vector)/2):
  polygons.append(get_clip(vector[2*n],vector[2*n+1],r,Na))
 #------------------------------------------------------  


 #Get inflated---------------------------------------------
 polygons_inflated = []
 for n in range(len(vector)/2):
  polygons_inflated.append(get_clip(vector[2*n],vector[2*n+1],r,Na))
 #------------------------------------------------------  


 #Periodic Vectors
 pbc = []
 pbc.append([Lx,0])
 pbc.append([-Lx,0])
 pbc.append([0,Ly])
 pbc.append([0,-Ly])
 pbc.append([Lx,Ly])
 pbc.append([-Lx,-Ly])
 pbc.append([-Lx,Ly])
 pbc.append([Lx,-Ly])
 #---------------------

 #Add periodic pores-----------------
 polys = np.array(polygons_inflated).copy()
 for poly in polys :
  for p in pbc :
   tmp = poly.copy()
   translate_poly(tmp,p)
   if check_collision(tmp,frame):
    x = np.mean(np.array(tmp)[:,0])
    y = np.mean(np.array(tmp)[:,1])
    polygons.append(get_clip(x,y,r,Na))
  
 #plot_pores(frame,polygons)
 #show()

 return polygons

def get_clip(x,y,r,Na,delta_pore = 0.0):
 #We consider inflated pores-----
 dphi = 2.0*math.pi/Na; 
 r3 = r*(1.0+delta_pore)
 poly_clip = []
 for ka in range(Na):
  ph =  dphi/2 + (ka-1) * dphi
  px  = x + r3 * math.cos(ph)
  py  = y + r3 * math.sin(ph)
  poly_clip.append([px,py])
 return poly_clip

def CheckCompatibility(vector,options):

  #THOSE ARE NOT PERIODIC

  #Adapt vector to info------------------------
  data = []
  for n in range(len(vector)/2):
   data.append([vector[2*n],vector[2*n+1]])
  #-------------------------------------------  

  #Get radius------------------------------
  Np = 16
  Na = options['Na']
  Lx = options['Lx']*options['Nx']
  Ly = options['Ly']*options['Ny']
  area = Lx*Ly*options['porosity']/Np
  r = math.sqrt(2.0*area/Na/math.sin(2.0*math.pi/options['Na']))  
  #------------------------------------------

  #Periodic Vectors----------------------
  pbc = []
  pbc.append([Lx,0])
  pbc.append([-Lx,0])
  pbc.append([0,Ly])
  pbc.append([0,-Ly])
  pbc.append([Lx,Ly])
  pbc.append([-Lx,-Ly])
  pbc.append([-Lx,Ly])
  pbc.append([Lx,-Ly])
  #---------------------------------------
  for np1 in range(Np):
   x = data[np1][0]
   y = data[np1][1]
   clip_1 = get_clip(x,y,r,Na)
   for np2 in range(Np-np1):
    for kp in pbc:
     x = data[np2][0] + kp[0]
     y = data[np2][1] + kp[1]
     clip_2 = get_clip(x,y,r,Na)
     c = Pyclipper()
     c.AddPath(scale_to_clipper(clip_1), PT_CLIP,True) 
     c.AddPath(scale_to_clipper(clip_2),PT_SUBJECT,True) 
     solution = scale_from_clipper(c.Execute(CT_INTERSECTION,PFT_EVENODD,PFT_EVENODD))
     if not solution == []  :
       return True
       break

  return False


def check_cycle_collision(pores,Lx,Ly): 

   Px = [-Lx,Lx]
   p = pores[0].copy()
   translate_poly(p,[0,Ly])
   pores.append(p)
   collision = False
   for p in range(len(pores)-1):
    p1 = pores[p].copy()
    if collision == False:
     for px in Px:
      p2 = pores[p+1].copy()
      translate_poly(p2,[px,0])  
      if check_collision(p1,p2.copy()):
       collision = True
       break
   return collision

def reorder_poly(cycle,polygons,length):

   nn_p = []
   Px = [-length,0,length] 
   poly_next = np.array(polygons[cycle[0]]).copy()
   for ii,n1 in enumerate(cycle):
    poly_current = np.array(poly_next).copy()
    n2 = int(cycle[(int(ii)+1)%len(cycle)])
    p2 = polygons[int(n2)]
    mind = 1e10
    ref = []
    for px in Px:
     for py in Px:
      tmp = np.array(p2).copy()
      ref = [px,py]
      translate_poly(tmp,ref)
      d = compute_distance_extended(poly_current,tmp)
      if d < mind:
       poly_next = tmp.copy()
       mind = d
       ref_ok = ref      
    nn_p.append(poly_next)

   return nn_p 

def plot_polygon(c,Na,ax,r,lim,phi0,col,y1):

 #-----------------
 x1 = lim[0]
 deltax = lim[1]
 x2 = lim[2]
 deltay = lim[3]
 #------------------
 
 p = []    
 dphi = 2.0*math.pi/Na;
 for ka in range(Na+1):
   ph =   (ka-1) * dphi + phi0/180.0*math.pi
   px  = x1 + c[0]*deltax + r * math.cos(ph)*deltax
   py  = y1 + c[1]*deltay + r * math.sin(ph)*deltay
   p.append([px,py])
 
 for ka in range(len(p)):
  p1 = p[ka]
  p2 = p[(ka+1)%len(p)]
  ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color=col,linewidth=3)

def get_distribution(data) :

 kde = stats.gaussian_kde(data)
 delta =  (data.max() -  data.min())*0.3
 xs = np.linspace(data.min()-delta, data.max()+delta, 200)

 return xs,kde


def plot_network(conn,polygons,L):

  Px = [-L,0,L]
  Py = [-L,0,L]
  #Px = [0]
  #Py = [0]
  for px in Px:
   for py in Py:
    for tmp1 in conn:
     g1 = tmp1
     c1 = compute_circumcenter(polygons[int(g1)]) + [px,py]
     for tmp2 in conn[tmp1]:
      g2 = tmp2[0]   
      deltax = tmp2[1]
      deltay = tmp2[2]
      c2 = c1 + [deltax,deltay]
      plot([c1[0],c2[0]],[c1[1],c2[1]],color='b',linewidth=2)  



def plot_index(conn,polygons,L,cycle,cc):

  Py = [-L,0,L]
  c1 = compute_circumcenter(polygons[int(cycle[0])])
  c1[1] -=5.0*L
  for i in range(10):
   #The beginning
   for n,node in enumerate(cycle):
    n2 = cycle[(n+1)%len(cycle)]
    tmp = conn[node]
    for g in tmp:
     if g[0] == n2:
      delta = [g[1],g[2]]
      c2 = c1 + delta
      break
    plot([c1[0],c2[0]],[c1[1],c2[1]],color=cc,linewidth=5)
    c1 = c2





def plot_network_and_index(conn,polygons,L,cycle):

  Px = [-L,0,L]
  Py = [-L,0,L]
  #Px = [0]
  #Py = [0]
  for px in Px:
   for py in Py:
    for tmp1 in conn:
     g1 = tmp1
     c1 = compute_circumcenter(polygons[int(g1)]) + [px,py]
     for tmp2 in conn[tmp1]:
      g2 = tmp2[0]   
      deltax = tmp2[1]
      deltay = tmp2[2]
      c2 = c1 + [deltax,deltay]
      plot([c1[0],c2[0]],[c1[1],c2[1]],color='b',linewidth=2)  

  Py = [-L,0,L]
  c1 = compute_circumcenter(polygons[int(cycle[0])])
  c1[1] -=5.0*L
  for i in range(10):
   #The beginning
   for n,node in enumerate(cycle):
    n2 = cycle[(n+1)%len(cycle)]
    tmp = conn[node]
    for g in tmp:
     if g[0] == n2:
      delta = [g[1],g[2]]
      c2 = c1 + delta
      break
    plot([c1[0],c2[0]],[c1[1],c2[1]],color='r',linewidth=5)
    c1 = c2




def check_collision(poly1,poly3):
    c = Pyclipper()
    c.AddPath(scale_to_clipper(poly1), PT_SUBJECT,True) 
    c.AddPaths(scale_to_clipper([poly3]),PT_CLIP,True) 
    solution = scale_from_clipper(c.Execute(CT_INTERSECTION,PFT_EVENODD,PFT_EVENODD))
    if len(solution)>0:
     return True
    return False


def compute_distance_extended(poly1,poly2):

 N = 30
 c1 = compute_circumcenter(poly1)
 c2 = compute_circumcenter(poly2)
 dis = c2-c1
 dp = np.array(dis)/N
 collision = False
 poly = np.array(poly1).copy()
 d = 0
 while not collision:
  translate_poly(poly,dp)
  #plot_pores(pol1,'black',3)
  d += np.linalg.norm(dp)
  collision = check_collision(poly,poly2)
 return d


def compute_distance(polygons,conn,p1,p2):

 N = 30
 poly1 = polygons[int(p1)]
 c1 = compute_circumcenter(poly1)
 tmp = conn[p1]
 for g in tmp:
  if g[0] == p2:
   dis = [g[1],g[2]]
   c2 = c1 + dis
   #Translate polygon p2---
   poly2 = np.array(polygons[p2]).copy()  
   c2tmp = compute_circumcenter(poly2)
   poly2 -=c2tmp
   poly2 += c2
   #-------------------------
   break

 dp = np.array(dis)/N
 collision = False
 poly = np.array(poly1).copy()
 d = 0
 while not collision:
  translate_poly(poly,dp)
  d += np.linalg.norm(dp)
  collision = check_collision(poly,poly2)

 return d

def get_frame(Lx,Ly):

 #frame-----------------------------------
 frame = []
 frame.append([float(-Lx)/2,float(Ly)/2])
 frame.append([float(Lx)/2,float(Ly)/2])
 frame.append([float(Lx)/2,float(-Ly)/2])
 frame.append([float(-Lx)/2,float(-Ly)/2])

 return frame


def plot_pores(frame,polygons,cc='r',lw=3):

 axis('equal')
 for f,p in enumerate(frame):
  plot([p[0],frame[(f+1)%len(frame)][0]],[p[1],frame[(f+1)%len(frame)][1]],linestyle='--',color='black',linewidth=2)

 for n,poly in enumerate(polygons):
  c = compute_circumcenter(poly)
  text(c[0],c[1],str(n))
  for f,p in enumerate(poly):
   plot([p[0],poly[(f+1)%len(poly)][0]],[p[1],poly[(f+1)%len(poly)][1]],color=cc,linewidth=lw)
   #text(c[0],c[1],str(n))


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

def plot_cycles(cycle,conn,polygons,periodic_polygons,corr,frame,L,ax):

  #fig =figure(num=None, figsize=(10, 10.0), dpi=80, facecolor='w', edgecolor='k')


  Py = [-L,0,L]
  c1 = compute_circumcenter(polygons[int(cycle[0])])
  c1[1] -=5.0*L
  for i in range(10):
   #The beginning
   for n,node in enumerate(cycle):
    n2 = cycle[(n+1)%len(cycle)]
    tmp = conn[node]
    for g in tmp:
     if g[0] == n2:
      delta = [g[1],g[2]]
      c2 = c1 + delta
      break
    plot([c1[0],c2[0]],[c1[1],c2[1]],color='r',linewidth=5)
    c1 = c2

  #small frame
  for f,p in enumerate(frame):
   plot([p[0],frame[(f+1)%len(frame)][0]],[p[1],frame[(f+1)%len(frame)][1]],linestyle='--',color='b',linewidth=4)



  #Create big frame----------------------------------------
  frame_big = np.zeros((4,2))
  for n1 in range(4):
   for n2 in range(2):
    frame_big[n1,n2] = frame[n1][n2]*2.0
  path = create_path(frame_big)
  patch = patches.PathPatch(path, facecolor='orange', lw=2,alpha=0.5)
  ax.add_patch(patch)  
  #--------------------------------------------------------------
  #plot polygons---
  for poly in periodic_polygons:
   p = np.array(poly).copy()
   path = create_path(p)
   patch = patches.PathPatch(path, facecolor='white', lw=2)
   ax.add_patch(patch)  

  axis('off')
  ylim([-1*L,1*L])
  xlim([-1*L,1*L])
  #show()

def pores_already_included(pp,polygons):

 c1 = compute_circumcenter(pp)
 for poly in polygons:
  c2 = compute_circumcenter(poly)
  if np.linalg.norm(c2-c1)<1e-10:
   return True
 return False




def translate_poly(poly,dp):
    for p in poly:
      p += dp

def get_periodic_polygons(polygons_tmp,L):

  Px = [0,-L,L]
  Py = [0,-L,L]
  corr = []
  polygons = []
  l = 0
  for px in Px:
   for py in Py:
    for n,poly in enumerate(polygons_tmp):
     pp = np.array(poly).copy()
     translate_poly(pp,[px,py])
     polygons.append(pp)
     corr.append([n,l])
    l +=1

  return polygons,corr 
     

def get_polygon(p):
 
 cx   = p[0]
 cy   = p[1]
 r    = p[2]
 Na   = p[3]
 phi0 = p[4]

 dphi = 2.0*math.pi/Na;
 poly = []
 for ka in range(Na):
  ph =  dphi/2 + (ka-1) * dphi + phi0*math.pi/180.0;
  px  = cx + r * math.cos(ph)
  py  = cy + r * math.sin(ph)
  poly.append([px,py])

 return poly


def import_pickle(namefile):

 L = 40.0
 frame = [[-L/2.0,-L/2.0],[-L/2.0,L/2.0],[L/2.0,L/2.0],[L/2.0,-L/2,0]]
 #frame = []
 polygons_info = np.load(namefile)
 polygons = []
 for p in polygons_info:
  polygons.append(get_polygon(p))

 return np.array(frame),polygons,L
 


def compute_circumcenter(poly):
   cx = 0.0
   cy = 0.0
   for p in poly:
    cx += p[0]
    cy += p[1]
   cx /=len(poly)
   cy /=len(poly)

   return np.array([cx,cy])


def prune_polygons(polygons,length):


 L = length
 #Px = [-L,0.0,L]
 #Py = [-L,0.0,L]

 polygons_prune = []
 #disc_list = []
 for p,poly in enumerate(polygons):
  c = compute_circumcenter(poly)
  if c[0] < -L/2.0 : continue
  if c[0] > L/2.0 : continue
  if c[1] < -L/2.0 : continue
  if c[1] > L/2.0 : continue
  polygons_prune.append(poly)
  #discard = False
  #for p2,poly2 in enumerate(polygons):
  # if discard == False: 
  #  if (not p1 == p2) and (not p2 in disc_list):
  #   for px in Px:
  #    for py in Py:
  #     c2 = compute_circumcenter(poly2)
  #     c2 += [px,py]  
  #     d = np.linalg.norm(c2-c1)
  #     if d < 1e-5:
  #      discard = True
  #      disc_list.append(p1)

  #if discard == False:
  # polygons_prune.append(poly)   

 return polygons_prune
