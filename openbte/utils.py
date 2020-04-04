import h5py
from google_drive_downloader import GoogleDriveDownloader as gdd
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
import shapely
from shapely.ops import cascaded_union


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

def create_layer(data,f):

  for i in data.keys():
         if not isinstance(i,str): 
                ind = str(i)
         else:  
                ind = i
         if isinstance(data[i],dict):
            g = f.create_group(ind)
            create_layer(data[i],g)
         else:
             f.create_dataset(ind,data=data[i],compression="gzip",compression_opts=9) 


def save_dictionary(data,filename):

 with h5py.File(filename, "w") as f:
      create_layer(data,f)

def load_layer(f):
     tmp = {}
     for i in f.keys():
        if i.isdigit():
               ind = int(i)
        else:   
               ind = i 
        if isinstance(f[i],h5py._hl.group.Group):
            name = f[i].name.split('/')[-1]
            if name.isdigit():
               ind2 = int(name)
            else:   
               ind2 = name 
            tmp.update({ind2:load_layer(f[i])}) 
        else:
            tmp.update({ind:np.array(f[i])}) 
     return tmp 



def load_dictionary(filename):

 data = {}
 with h5py.File(filename, "r") as f:
     data.update({'root':load_layer(f)})

 return data['root']



def download_file(file_id,filename):
      gdd.download_file_from_google_drive(file_id=file_id,
                                           dest_path='./' + filename,showsize=True,overwrite=True)


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


def generate_frame(**argv):

    Lx = float(argv['lx'])
    Ly = float(argv['ly'])
    frame = []
    frame.append([-Lx/2,Ly/2])
    frame.append([Lx/2,Ly/2])
    frame.append([Lx/2,-Ly/2])
    frame.append([-Lx/2,-Ly/2])

    return frame


def translate_shape(shape,base):

  out = []
  for p in shape:
    tmp = [p[0] + base[0],p[1] + base[1]]
    out.append(tmp)

  return out

def repeat_merge_scale(argv):

  polygons = argv['polygons']

  
  pbc = []
  pbc.append([0,0])
  pbc.append([1,0])
  pbc.append([-1,0])
  pbc.append([0,1])
  pbc.append([0,-1])
  pbc.append([1,1])
  pbc.append([-1,-1])
  pbc.append([-1,1])
  pbc.append([1,-1])
  lx = argv['lx']   
  ly = argv['ly']   


  frame = Polygon([[-0.5,-0.5],[-0.5,0.5],[0.5,0.5],[0.5,-0.5]])

   #---------------------
  #Store only the intersecting polygons
  final = []
  for poly in polygons:
    for kp in range(len(pbc)):
     tmp = []
     for p in poly:
      cx = p[0] + pbc[kp][0]
      cy = p[1] + pbc[kp][1]
      tmp.append([cx,cy])
     p1 = Polygon(tmp)
     if p1.intersects(frame):
      thin = Polygon(p1).intersection(frame)
      if isinstance(thin, shapely.geometry.multipolygon.MultiPolygon):
       tmp = list(thin)
       for t in tmp:
        final.append(t)
      else:
        final.append(thin)

  #Create bulk surface---get only the exterior to avoid holes
  MP = MultiPolygon(final) 

  conso = cascaded_union(MP)

  polygons_final = []
  if isinstance(conso, shapely.geometry.multipolygon.MultiPolygon):
      for i in conso: 
       polygons_final.append(list(i.exterior.coords))
  else: 
       polygons_final.append(list(conso.exterior.coords))
  
  #-------------------------------------------
  #scale-----------------------
  polygons = []
  for poly in polygons_final:
   new_poly = []
   for p in poly:
    g = list(p)
    g[0] *=lx
    g[1] *=ly
    new_poly.append(g)
   polygons.append(new_poly) 

  #cut redundant points (probably unnecessary)
  new_poly = []
  for poly in polygons:
   N = len(poly)
   tmp = []
   for n in range(N):
    p1 = poly[n]
    p2 = poly[(n+1)%N]
    if np.linalg.norm(np.array(p1)-np.array(p2)) >1e-4:
     tmp.append(p1)
   new_poly.append(tmp)

  argv['polygons'] = new_poly


