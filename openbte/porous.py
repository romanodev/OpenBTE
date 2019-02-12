from shapely.ops import cascaded_union
from shapely.geometry import Point,LineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
import shapely
import numpy as np

class Porous(object):

 def __init__(self,**argv):

  self.step = argv['step']
  self.points = []
  self.lines = []
  self.loops = []
  self.surfaces = []
  self.argv = argv
 
    
  self.common(z=-1)
  self.common(z=1)
  self.merge() 
  
  self.write_geo()

  quit()


 def merge(self):

   #Create lines
   Nh = int(len(self.points)/2)
   for n in range(Nh):
     self.lines.append([n,n+Nh])  
     

   #Create new loops 
   N = int(len(self.loops)/2)
   for ll in range(N):
    for l in self.loops[ll]:         
     
     i1 = self.lines[l-1][0];  i2 = self.lines[l-1][1]
              
     j1 = i1 + Nh; j2 = i2 + Nh
     tmp = []
     tmp.append(self.line_exists_ordered(i1,i2))
     tmp.append(self.line_exists_ordered(i2,j2))   
     tmp.append(self.line_exists_ordered(j2,j1))
     tmp.append(self.line_exists_ordered(j1,i1))
     
     self.loops.append(tmp)
     
 
 


 def write_geo(self):
     
   store = open('mesh.geo', 'w+')
   
   #Write points-----
   for p,point in enumerate(self.points):
    store.write( 'Point('+str(p) +') = {' + str(point[0]) +','+ str(point[1])+',' + str(point[2])+','+ str(self.step) +'};\n')
   
   #Write lines----
   for l,line in enumerate(self.lines):
     store.write( 'Line('+str(l+1) +') = {' + str(line[0]) +','+ str(line[1])+'};\n')
      
   #Write loops------
   nl = len(self.lines)
   for ll,loop in enumerate(self.loops):  
    strc = 'Line Loop(' + str(ll+nl)+ ') = {'
    for n,l in enumerate(loop):
     strc +=str(l)
     if n == len(loop)-1:
      strc += '};\n'
     else:
      strc += ','
    store.write(strc)
   
    #Write surface------
    #strc = '''Plane Surface(''' +str(nl) + ''')= {'''
    
    #if s == len(self.loops)-1:
    # strc += '};\n'
    #else:
    # strc += ','
   #store.write(strc)
   
   #------------------
  strc = r'''Surface Loop(''' + str(ss) + ') = {'
  for n,p in enumerate(surfaces) :
   strc +=str(p)
   if n == len(surfaces)-1:
    strc += '};\n'
   else :
    strc += ','
  store.write(strc)
   
   
   
  store.close()

 def line_exists_ordered(self,p1,p2):
     
  l = [p1,p2]   
  for n,line in enumerate(self.lines) :
   if (line[0] == l[0] and line[1] == l[1]) :
    return n
   if (line[0] == l[1] and line[1] == l[0]) :
    return -n
  return 0

 def already_included(self,p):

  for n,point in enumerate(self.points):
   d = np.linalg.norm(np.array(point)-np.array(p))
   if d < 1e-12:
    return n
  return -1


 def create_lines_and_loop_from_points(self,pp,z):
     
   p_list = []  
   for p in pp:
    p = [p[0],p[1],z]   
    tmp = self.already_included(p) 
    if tmp == -1:
      self.points.append(p)
      p_list.append(len(self.points)-1)
    else:
      p_list.append(tmp)


   line_list = []
   for l in range(len(p_list)):
    p1 = p_list[l]
    p2 = p_list[(l+1)%len(p_list)]
    if not p1 == p2:
     tmp = self.line_exists_ordered(p1,p2)
     if tmp == 0 : #craete line
      self.lines.append([p1,p2])
      line_list.append(len(self.lines))
     else:
      line_list.append(tmp)
      
   self.loops.append(line_list)  
      



 def common(self,z):

  Frame = Polygon(self.argv['frame'])
  polygons = self.argv['polygons']

  polypores = []
  for poly in polygons:
   thin = Polygon(poly).intersection(Frame)
   if isinstance(thin, shapely.geometry.multipolygon.MultiPolygon):
     tmp = list(thin)
     for t in tmp:
      polypores.append(t)
   else:
    polypores.append(thin)
  MP = MultiPolygon(polypores)
  bulk = Frame.difference(cascaded_union(MP))
  if not (isinstance(bulk, shapely.geometry.multipolygon.MultiPolygon)):
   bulk = [bulk]

  
  for region in bulk:
   pp = list(region.exterior.coords)[:-1]
   self.create_lines_and_loop_from_points(pp,z)
  
   #From pores 
   for interior in region.interiors:
    pp = list(interior.coords)[:-1]
    self.create_lines_and_loop_from_points(pp,z)
    
    







 

  




