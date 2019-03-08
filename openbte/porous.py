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
 
    
  self.common(z=-self.argv['lz']/2)
  self.common(z=self.argv['lz']/2)
  self.merge() 
  self.points = np.array(self.points)
  self.lines = np.array(self.lines)
  self.loops = np.array(self.loops)

  
  self.apply_periodic_mesh()
  
  self.write_geo()



 #def get_surface_normal(self,ss):
     
  # s = self.surfaces[ss][0]  
   #p1 = self.points[self.lines[self.loops[s,0]]]
  # p1 = self.points[self.lines[abs(self.loops[s,0])-1]]
  # p2 = self.points[self.lines[abs(self.loops[s,1])-1]]
  # normal = np.cross(p1[1]-p1[0],p2[1]-p2[0])
  # normal /= np.linalg.norm(normal)
   #if ss==2:
   #    print(normal)
   #    print(p1)
  # return normal
     
 def get_surface_centroid(self,ss):
     
  
   s = self.surfaces[ss][0]  
    
   pts = []
   for n in self.loops[s]:
    if n > 0:
        i = 0
    else:
        i = 1
    pts.append(self.points[self.lines[abs(n)-1][i]])
   pts = np.array(pts) 
    
   
   
   return np.mean(pts,axis=0)
   
     

 def isperiodic(self,s1,s2):

   #n = self.get_surface_normal(s1)
   #n2 = self.get_surface_normal(s2)
   c1 = self.get_surface_centroid(s1)
   c2 = self.get_surface_centroid(s2)
   
   
   
   if abs(np.linalg.norm(c1-c2) - self.argv['lx'])<1e-4 and \
      np.linalg.norm(np.cross(c1-c2,[1,0,0])) < 1e-4: 
      return 0,c1-c2
   

   if abs(np.linalg.norm(c1-c2) - self.argv['ly'])<1e-4 and \
      np.linalg.norm(np.cross(c1-c2,[0,1,0])) < 1e-4: 
      return 1,c1-c2
   
   if abs(np.linalg.norm(c1-c2) - self.argv['lz'])<1e-4 and \
      np.linalg.norm(np.cross(c1-c2,[0,0,1])) < 1e-4: 
      return 2,c1-c2
   
  
    
   return None,None
   
 
   
    
 def get_points_from_surface(self,s1):
      
   ll = self.surfaces[s1][0]
   pts = []
   ind = []
   for n in self.loops[ll]:
    if n > 0:
        i = 0
    else:
        i = 1
    pts.append(self.points[self.lines[abs(n)-1][i]])
    
    ind.append(self.lines[abs(n)-1][i])


   return np.array(pts),ind
     
     
 def apply_periodic_mesh(self):
     self.argv.setdefault('Periodic',[True,True,False])
     #Find periodic surfaces----
     self.periodic = {}
     for s1 in range(len(self.surfaces)):
      for s2 in range(s1+1,len(self.surfaces)):
         (a,per) = self.isperiodic(s1,s2)
         if not a == None and self.argv['Periodic'][a]:
           
          
           corr = {}
           pts1,ind1 = self.get_points_from_surface(s1)
           pts2,ind2 = self.get_points_from_surface(s2)

           #if a == 1:
           #  print(pts1)

           for n1,p1 in zip(ind1,pts1):
            for n2,p2 in zip(ind2,pts2):
             if np.linalg.norm(p1-p2-per) < 1e-3:
                corr[n1] = n2
                break            
           #-----------------------------------------
           loop_1 = self.loops[self.surfaces[s1][0]] 
           loop_2 = []
           for ll in loop_1 :
             pts = self.lines[abs(ll)-1]
             if ll > 0:
              p1 = pts[0]; p2 = pts[1] 
             else:
              p1 = pts[1]; p2 = pts[0]   
             loop_2.append(self.line_exists_ordered(corr[p1],corr[p2]))             
           self.periodic[s1] = [s2,loop_1,loop_2,np.array(per/np.linalg.norm(per))]  
           
          
           
           break  
     
      
        
       

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
     self.surfaces.append([len(self.loops)-1])
     
 
 
 #def apply_periodic_mesh(self):

 def write_geo(self):

   
   store = open('mesh.geo', 'w+')

   
   #Write points-----
   for p,point in enumerate(self.points):
    store.write( 'Point('+str(p) +') = {' + str(point[0]) +','+ str(point[1])+',' + str(point[2])+','+ str(self.step) +'};\n')
   
   #Write lines----
   for l,line in enumerate(self.lines):
     store.write( 'Line('+str(l+1) +') = {' + str(line[0]) +','+ str(line[1])+'};\n')
      
   #Write loops------
   nl = len(self.lines)+1
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
   nll = len(self.loops)
   for s,surface in enumerate(self.surfaces):
    strc = '''Plane Surface(''' +str(s+nll+nl) + ''')= {''' 
    for n,ll in enumerate(surface):
      strc += str(ll+nl)  
      if n == len(surface)-1:
       strc += '};\n'
      else:
       strc += ','    
    store.write(strc)
    

   boundary = list(range(len(self.surfaces)))
   #Apply periodicity
   vec = {}
   for key, value in self.periodic.items():
   
     boundary.remove(key)  
     boundary.remove(value[0])  

     value[3] = np.round(value[3],4)     
     store.write('Periodic Surface ' + str(key+nl+nll) + ' {') 
     
     for n,k in enumerate(value[1]):
      store.write(str(k))
      if n < len(value[1])-1:
       store.write(',')
     store.write('} = ' + str(value[0]+nl+nll) + ' {')
     
     for n,k in enumerate(value[2]):
      store.write(str(k))
      if n < len(value[2])-1:
       store.write(',')
     store.write('};\n')
    

     value[0] += nl+nll
     tt = key +  nl + nll
     if np.array_equal(np.array([1,0,0]),value[3]) : 
        vec.setdefault(1,[]).append(value[0]) 
        vec.setdefault(2,[]).append(tt) 
     
     if np.array_equal(np.array([-1,0,0]),value[3]): 
        vec.setdefault(1,[]).append(tt) 
        vec.setdefault(2,[]).append(value) 
        
     if np.array_equal(np.array([0,1,0]),value[3]) : 
        vec.setdefault(3,[]).append(value[0]) 
        vec.setdefault(4,[]).append(tt)
        
     if np.array_equal(np.array([0,-1,0]),value[3]) : 
        vec.setdefault(4,[]).append(value[0]) 
        vec.setdefault(3,[]).append(tt)
        
     if np.array_equal(np.array([0,0,1]),value[3]) : 
        vec.setdefault(5,[]).append(value[0]) 
        vec.setdefault(6,[]).append(tt)   
        
     if np.array_equal(np.array([0,0,-1]),value[3]) : 
        vec.setdefault(6,[]).append(value[0]) 
        vec.setdefault(5,[]).append(tt) 
        
   for key, value in vec.items():
    
     strc = r'''Physical Surface("Periodic_''' + str(key) + '''") = {'''
     for n,p in enumerate(value) :
      strc +=str(p)
      if n == len(value)-1:
       strc += '};\n'
      else :
       strc += ','
     store.write(strc)
         
   strc = r'''Physical Surface("Boundary") = {'''
   for n,p in enumerate(boundary) :
      strc +=str(p+nl+nll)
      if n == len(boundary)-1:
       strc += '};\n'
      else :
       strc += ','
   store.write(strc)
    
    
   if len(self.surfaces) > 0:
    #------------------
    ss  = len(self.surfaces)
    strc = r'''Surface Loop(''' + str(ss+nll+nl) + ') = {'
    for s in range(len(self.surfaces)) :
     strc +=str(s+nl+nll)
     if s == len(self.surfaces)-1:
      strc += '};\n'
     else :
      strc += ','
    store.write(strc)
    
    strc = r'''Volume(0) = {''' + str(ss+nll+nl) + '};\n'
    store.write(strc)
    strc = r'''Physical Volume('Bulk') = {0};'''
    store.write(strc) 

   store.close()

 def line_exists_ordered(self,p1,p2):
     
  l = [p1,p2]   
  for n,line in enumerate(self.lines) :
   if (line[0] == l[0] and line[1] == l[1]) :
    return n+1
   if (line[0] == l[1] and line[1] == l[0]) :
    return -(n+1)
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

  
  loop_start = len(self.loops)  
  for region in bulk:
   pp = list(region.exterior.coords)[:-1]
   self.create_lines_and_loop_from_points(pp,z)
   
    
   #From pores 
   for interior in region.interiors:
    pp = list(interior.coords)[:-1]
    self.create_lines_and_loop_from_points(pp,z)
  loop_end = len(self.loops)  
  
  self.surfaces.append(range(loop_start,loop_end))

    







 

  




