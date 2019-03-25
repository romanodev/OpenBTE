import numpy as np
from shapely.ops import cascaded_union
from shapely.ops import unary_union
from shapely.geometry import Point
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
from shapely.geometry import shape, JOIN_STYLE
import math


def is_neighbor(pos1,pos2,nx,ny):
 delta = 1e-3
 if abs(pos1[0] - pos2[0])<delta and abs(abs(pos1[1]-pos2[1]) -1) < delta:
  return True
 if abs(abs(pos1[0] - pos2[0]) -1)  < delta and abs(pos1[1] - pos2[1]) < delta:
  return True
 #if abs(abs(pos1[0] - pos2[0]) - (nx-1))  < delta and abs(pos1[1] - pos2[1]) < delta:
 # return True
 #if abs(pos1[0] - pos2[0])<delta and abs(abs(pos1[1]-pos2[1]) -(ny-1)) < delta:
 # return True

 return False


#Get an island of connecting pores----
def get_merged_pores(output,pos,total,nx,ny):
 for n,p in enumerate(pos):
  if not n in output and not n in total:
   for o in output:
    if is_neighbor(p,pos[o],nx,ny):
     output.append(n)
     get_merged_pores(output,pos,total,nx,ny)
     break
 #return output



def get_square(x,y,area):

 Na = 4
 r = math.sqrt(2.0*area/Na/math.sin(2.0 * math.pi/Na))
 dphi = 2.0*math.pi/Na;
 poly_clip = []
 for ka in range(Na+1):
      ph =  dphi/2 + (ka-1)*dphi
      px  = x + r * math.cos(ph)
      py  = y + r * math.sin(ph)
      poly_clip.append([px,py])
 return poly_clip

def merge_positions(pos,nx,ny):

 n_pores = len(pos)
 flat_merged_pores = []
 total_merged_pores = []
 n_filled = 0
 start = 0
 while n_filled < n_pores:
  pores = [start]
  get_merged_pores(pores,pos,flat_merged_pores,nx,ny)
  total_merged_pores.append(pores)
  flat_merged_pores = [element for tupl in total_merged_pores for element in tupl]
  n_filled = len(flat_merged_pores)
  #print(n_filled)
  #Get next available pores-------------
  while True:
    start +=1
    if not start in flat_merged_pores:
     break

 return total_merged_pores

def complement_positions(positions,nx,ny):

  compl = []
  for x in range(nx):
   for y in range(ny):
    included = False
    for tmp in positions:
     if tmp[0] == x and tmp[1] == y:
      included = True
      break
    if not included:
     compl.append([x,y])

  return compl



def  delete_unnecessary_points(polygons):

  #ELIMINATE DUPLICATE POINTS------
  n_del = 1
  n_iter = 0
  while n_del > 0:
   n_del = 0
   delete_list = []
   for n,poly in enumerate(polygons):
    tmp = []
    for p in range(len(poly)):
     p1 = np.array(poly[p])
     p2 = np.array(poly[(p+1)%(len(poly))])
     p3 = np.array(poly[(p+2)%(len(poly))])
     d1 = p2-p1
     d2 = p3-p2
     a = abs(np.dot(d1,d2)/np.linalg.norm(d1)/np.linalg.norm(d2)) #if they are collinear
     if a > 0.9: #if they are collinear
      tmp.append((p+1)%(len(poly)))
      n_del += 1
    delete_list.append(tmp)

   #print('Unnecessary_points: ' + str(n_del))
   #--------------------------------------
   new_polygons = []
   for d1 in range(len(delete_list)):
    tmp = []
    for p in range(len(polygons[d1])):
     if not p in delete_list[d1]:
      tmp.append(polygons[d1][p])
    new_polygons.append(tmp)
   polygons = new_polygons
   return polygons


def  delete_duplicate_points(polygons):

  #ELIMINATE DUPLICATE POINTS------
  n_del = 1
  while n_del > 0:
   n_del = 0
   delete_list = []
   for n,poly in enumerate(polygons):
    tmp = []
    for p in range(len(poly)):
     p1 = np.array(poly[p])
     p2 =np.array(poly[(p+1)%(len(poly))])
     if np.linalg.norm(p1-p2)<1e-2:
      tmp.append(p)
      n_del += 1
    delete_list.append(tmp)
   #print('Duplicate_points:' + str(n_del))
   #--------------------------------------
   for d1 in range(len(delete_list)):
    n = 0
    for d2 in delete_list[d1]:
     del polygons[d1][d2-n]
     n += 1

  return polygons




def compute_inverse_polygons(positions,nx,ny,area,dx,dy,Lx,Ly,frame):

   compl = complement_positions(positions,nx+1,ny+1)
   merged_material = merge_positions(compl,nx,ny)
   coords = []
   for m1 in merged_material:
    p = []
    for m2 in m1:
     poly = get_square(compl[m2][0]*dx-Lx*0.5,compl[m2][1]*dy-Ly*0.5,area)
     p.append(Polygon(poly).buffer(0))
    m = MultiPolygon(p)
    #total = cascaded_union(m)

    total = cascaded_union(m).intersection(frame)
    #quit()
    external = list(total.exterior.coords)[0:-1]
    external = delete_duplicate_points(list([external]))
    #external = delete_unnecessary_points(list(external))


    internal = []
    for k in total.interiors:
     tmp = list(k.coords)[0:-1]
     tmp = delete_duplicate_points([tmp])
     tmp = delete_unnecessary_points(tmp)
     internal.append(tmp[0])
    coords.append({'external':external[0],'internal':internal})

   return coords



def compute_polygons_from_grid(values):
  #Convert--
  (nx,ny)=np.shape(values)
  (ix,iy) = values.nonzero()
  positions_old = []
  for a,b in zip(ix,iy):
   positions_old.append([a,b])


  #positions_old = argv['x']
  #positions_old = np.reshape(argv['x'],(len(argv['x'])/2,2))

  #Lx =  argv.setdefault('Lx',40.0)
  Lx = 1.0
  Ly = 1.0
  #Ly =  argv.setdefault('Ly',40.0)
  #phi =  argv.setdefault('porosity',0.3)
  Na = 4
  #nx =  argv.setdefault('nx',10)
  #ny =  argv.setdefault('ny',10)
  dx = Lx/nx
  dy = Ly/ny
  area = dx*dy
  frame_tmp = []
  frame_tmp.append([float(-Lx)/2,float(Ly)/2])
  frame_tmp.append([float(Lx)/2,float(Ly)/2])
  frame_tmp.append([float(Lx)/2,float(-Ly)/2])
  frame_tmp.append([float(-Lx)/2,float(-Ly)/2])
  frame = Polygon(frame_tmp)



  #quit()
  delta = 1e-3

  positions = []
  polygons = []
  for pos in positions_old:
    xr = pos[0]
    yr = pos[1]
    polygons.append(get_square(xr*dx-Lx*0.5,\
                               yr*dy-Lx*0.5,\
                               area))
    positions.append(pos)

    #if argv.setdefault('add_periodic',True):
    if 1 ==1:
    #ADD PERIODIC PORES--------------------------
     if xr == 0:
       polygons.append(get_square(xr*dx-Lx*0.5 + Lx, yr*dy-Lx*0.5,area))
       positions.append([nx,yr])

     if yr == 0:
      polygons.append(get_square(xr*dx-Lx*0.5     , yr*dy-Lx*0.5 + Ly ,area))
      positions.append([xr,ny])

     if yr == 0 and xr == 0:
      polygons.append(get_square(xr*dx-Lx*0.5 + Lx, yr*dy-Lx*0.5 + Ly ,area))
      positions.append([nx,ny])

    #-----------------------------------------------------

  #print('START')
  lone_pores = []
  #if argv.setdefault('eliminate_lone_pores',True):
  if 1 == 1:
  #if argv.setdefault('eliminate_lone_pores',True):
   #Eliminate holes-----
   th = 15
   compl = complement_positions(np.array(positions),nx,ny)
   merged_material = merge_positions(compl,nx,ny)
   #print('Number of holes:')
   for hole in merged_material:
    if len(hole) < th:
     for pos in hole:
      #fill missing pores
      positions.append(compl[pos])
      polygons.append(get_square(compl[pos][0]*dx-Lx*0.5,\
                                compl[pos][1]*dy-Ly*0.5,\
                                area))
      lone_pores.append(len(polygons)-1)


      if 1 == 1:
      #if argv.setdefault('add_periodic',True):

       if compl[pos][0] == 0:
         polygons.append(get_square(compl[pos][0]*dx-Lx*0.5+Lx,\
                                    compl[pos][1]*dy-Ly*0.5,\
                                    area))
         lone_pores.append(len(polygons)-1)

         positions.append([nx,compl[pos][1]])

       if compl[pos][1] == 0:
        polygons.append(get_square(compl[pos][0]*dx-Lx*0.5,\
                                   compl[pos][1]*dy-Ly*0.5+Ly,\
                                   area))
        lone_pores.append(len(polygons)-1)
        positions.append([compl[pos][0],ny])

       if compl[pos][0] == 0 and compl[pos][1] == 0:
        lone_pores.append(len(polygons)-1)
        polygons.append(get_square(compl[pos][0]*dx-Lx*0.5 + Lx,\
                                   compl[pos][1]*dy-Ly*0.5 + Ly,\
                                   area))
        lone_pores.append(len(polygons)-1)
        positions.append([nx,ny])
  #print('Added lone pores:' + str(len(lone_pores)))
  #quit()
  #MERGE POLYGONS:#---------
  merge_poly = True
  if merge_poly:
   polygons = np.array(polygons)
   merged = merge_positions(positions,nx,ny)

   npp = 0
   polygons_final = []
   for m1 in merged:
    p = []
    for m2 in m1:
     #p.append(Polygon(polygons[m2]).buffer(1e-6))
     tt = Polygon(polygons[m2]).buffer(1e-6, resolution=1,mitre_limit=20.0,join_style=2)
     p.append(tt)
    m = MultiPolygon(p)
    npp += len(p)
    total = cascaded_union(m)

    polygons_final.append(list(total.exterior.coords)[0:-1])

  else:
   polygons_final = polygons
   merged = []
   for p in range(len(polygons_final)):
    merged.append([p])

  #print(npp)
  #print(len(positions))
  #polygons_final = polygons
  #type it up--------
  polygons_final = delete_duplicate_points(list(polygons_final))
  #[print(len(n)) for n in polygons_final] 
  polygons_final = delete_unnecessary_points(list(polygons_final))
  #print(' ')
  #[print(len(n)) for n in polygons_final] 

  #------------------

  #inverse_polygons = compute_inverse_polygons(positions,nx,ny,area,dx,dy,Lx,Ly,frame)
  #inverse_polygons = delete_duplicate_points(list(inverse_polygons))
  #inverse_polygons = delete_unnecessary_points(list(inverse_polygons))
  #--------------------------------------------------
  #----------------------------
  area = 0
  for p in polygons_final:
   area += Lx*Ly - frame.difference(Polygon(p)).area
  porosity = area/Lx/Ly

  delta = 1e-3
  #output = {'polygons':polygons_final,'Lx':Lx*(1.0-delta),'Ly':Ly*(1.0-delta),'test':merged,'lone':lone_pores,'porosity':porosity,'positions':positions,'inverse_polygons':inverse_polygons}

  polygons = []
  for poly in polygons_final:
   tmp = []
   for p in poly:
    tmp.append(p[0])
    tmp.append(p[1])
   polygons.append(tmp)


  return polygons



def GenerateRandomPoresGrid(**options):

 nx = options['nx']
 ny = options['ny']
 Np = options['np']

 values = np.zeros((nx,ny))

 if options.setdefault('manual',False):
  values = options['grid']   
 else: 
  pos = []
  while len(pos) < Np :
   x = np.random.randint(0,nx)
   y = np.random.randint(0,ny)
   tmp = [x,y]
   if not tmp in pos:
    pos.append(tmp)
    values[x,y] = 1

 polygons = compute_polygons_from_grid(values)
 return values,polygons



def generate_first_guess(options):

 Lx = options['Lx']
 Ly = options['Ly']
 Np = options['Np']

 pos = []
 for n in range(Np):
  x = np.random.uniform(-Lx/2.0,Lx/2.0)
  y = np.random.uniform(-Ly/2.0,Ly/2.0)
  pos.append(x)
  pos.append(y)

 return pos





def compute_cost(options,pos):

 output = adapt_data(options,pos)

 return output['polygons']



def plot_sample(options,pos):

 output = adapt_data(options,pos)


 #positions = prepare_data(pos,options)
 #options.update({'eliminate_lone_pores':True})
 #options.update({'add_periodic':True})
 Lx = options['Lx']
 Ly = options['Ly']
 #output = adapt_data(options,positions)
 coords = output['polygons']


 #test = output['test']
 #lone = output['lone']



 #r = lambda: float(random.randint(0,255))/255.0
 #colors = []
 #for g in test:
 # colors.append((r(),r(),r()))



 fig = figure()
 ax = fig.add_subplot(111, aspect='equal')
 frame = []
 frame.append([float(-Lx)/2,float(Ly)/2])
 frame.append([float(Lx)/2,float(Ly)/2])
 frame.append([float(Lx)/2,float(-Ly)/2])
 frame.append([float(-Lx)/2,float(-Ly)/2])
 frame.append([float(-Lx)/2,float(Ly)/2])

 for f,p in enumerate(frame):
   plot([p[0],frame[(f+1)%len(frame)][0]],[p[1],frame[(f+1)%len(frame)][1]],color='g',linewidth=4)

 path = create_path(frame)
 patch = patches.PathPatch(path, facecolor='orange', lw=2,alpha=0.5)
 ax.add_patch(patch)
 #--------------------------------------------------------------
 #plot polygons---i
 for n,poly in enumerate(coords):
  color = 'white'
  #for t,tt in enumerate(test):
  # if n in tt:
  #  color = colors[t]
  #  break
  #if n in lone:
  # color = 'black'
  #for p in range(len(poly)):
  # p1 = np.array(poly[p])
  # p2 = np.array(poly[(p+1)%(len(poly)-1)])
  # if np.linalg.norm(p1-p2)<1e-4:
  #  print(p1)
  #  print(p2)
  #  print(" ")

  p = np.array(poly).copy()
  path = create_path(p)
  patch = patches.PathPatch(path, facecolor=color, lw=2)
  ax.add_patch(patch)



 axis('off')
 axis('equal')
 ylim([-Lx/2,Lx/2])
 xlim([-Ly/2,Ly/2])

 show()
 return fig


def plot_coordinates(output):

 coords = output['coords']
 Lx = output['Lx']
 Ly = output['Ly']

 #P = [[0,0],[0,Ly],[Lx,Ly],[Lx,0],[0,-Ly],[-Lx,0],[-Lx,-Ly],[-Lx,Ly],[Lx,-Ly]]
 P = [[0,0]]
 for coord in coords:
  Np=len(coord)
  for tr in P:
   for ka in range(len(coord)-1):
    plot([coord[ka][0]+tr[0],coord[(ka+1)][0] + tr[0]],\
         [coord[ka][1]+tr[1],coord[(ka+1)][1] + tr[1]],color='b')
    #if Np > 4:
    #print(Np)
    #text(coord[ka][0]+tr[0],coord[ka][1]+tr[1],str(ka))

 plot([-Lx/2,-Ly/2],[-Lx/2,Ly/2],color='r',ls='--',lw=3)
 plot([-Lx/2,Ly/2],[Lx/2,Ly/2],color='r',ls='--',lw=3)
 plot([Lx/2,Ly/2],[Lx/2,-Ly/2],color='r',ls='--',lw=3)
 plot([Lx/2,-Ly/2],[-Lx/2,-Ly/2],color='r',ls='--',lw=3)

 axis('equal')
 axis('off')
 xlim([-Lx,Lx])
 ylim([-Ly,Ly])
 show()


def plot_configuration(output):

 vector = output['vector']


 L = 40
 P = [[0,0],[0,L],[L,L],[L,0],[0,-L],[-L,0],[-L,-L],[-L,L],[L,-L]]
 r = 5.0*sqrt(0.4)
 pairs = []
 N = len(vector)/2
 for n in range(N):
  x = vector[2*n]
  y = vector[2*n+1]
  for tr in P:
   pairs = []
   for ka in range(4):
    ph =  (ka-1) * np.pi/2.0
    px  = x + tr[0] + r * math.cos(ph+np.pi/4.0)
    py  = y  + tr[1] + r * math.sin(ph+np.pi/4.0)
    pairs.append([px,py])
   for ka in range(4):
    plot([pairs[ka][0],pairs[(ka+1)%4][0]],[pairs[ka][1],pairs[(ka+1)%4][1]],color='b')
 plot([-L/2,-L/2],[-L/2,L/2],color='r',ls='--',lw=3)
 plot([-L/2,L/2],[L/2,L/2],color='r',ls='--',lw=3)
 plot([L/2,L/2],[L/2,-L/2],color='r',ls='--',lw=3)
 plot([L/2,-L/2],[-L/2,-L/2],color='r',ls='--',lw=3)

 axis('equal')
 axis('off')
 xlim([-L,L])
 ylim([-L,L])
 show()



if __name__ == "__main__":


 vector = compute_first_guess()

 kappa = compute_kappa(vector)





