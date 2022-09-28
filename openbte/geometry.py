import numpy as np
from shapely.geometry import Polygon,MultiPolygon 
from openbte.objects import f64,Array,List,NamedTuple
from shapely.affinity import translate
from shapely.ops import unary_union
import os,subprocess


def circle(area :f64 = 1.0 ,\
           x    :f64 = 0.0 ,
           y    :f64 = 0.0)->Polygon:

    Na   = 24
    dphi = 2.0*np.pi/Na
    r = np.sqrt(2.0*area/Na/np.sin(dphi))
    phase =  dphi/2 + (np.arange(Na)-1) * dphi
    points = r*np.stack((np.cos(phase),np.sin(phase))).T + np.array([x,y])[np.newaxis,:]

    return Polygon(points)

def rectangle(area          :f64 = 1,\
              aspect_ratio  :f64 = 1,\
              x             :f64 = 0,\
              y             :f64 = 0):

   Lx = np.sqrt(area/aspect_ratio)

   Ly = aspect_ratio*Lx

   return Polygon(np.array([[-Lx/2,-Ly/2],[-Lx/2,Ly/2],[Lx/2,Ly/2],[Lx/2,-Ly/2]])+np.array([x,y])[np.newaxis,:])


def triangle(side             :f64 = 1,\
             x                :f64 = 0,\
             y                :f64 = 0):

    area = np.sqrt(3)/4*side*side
    Na   = 3
    dphi = 2.0*np.pi/Na
    r = np.sqrt(2.0*area/Na/np.sin(dphi))
    phase =  dphi/2 + (np.arange(Na)-1) * dphi
    points = r*np.stack((np.cos(phase),np.sin(phase))).T + np.array([x,y])[np.newaxis,:]

    return Polygon(points)





class Geometry(object):


     def __init__(self,step,lz:f64 = 0):

         self.step = step
         self.geometry = Polygon()
         self.regions    : List= []
         self.boundaries : dict =  {}
         self.periodic_sides  : dict =  {}


     def add_shape(self,polygon:Polygon):    

          self.geometry =   unary_union([self.geometry,polygon])


     def add_hole(self,polygon: Polygon,name: str = 'dummy'):

        previous_n_interior_regions = len(self.geometry.interiors)  

        self.geometry = self.geometry.difference(polygon) 

        next_n_interior_regions = len(self.geometry.interiors)  
 
        #Rationale: if the hole intersect the border, then a new outer region is created, instead of a new internal ones
        if next_n_interior_regions == previous_n_interior_regions + 1:
          self.regions.append(name)

     def set_boundary_region(self,selector:str,region:str,custom:List = []):
         """Set the name of specified boundaries"""

         if selector == 'outer':
            indices =  self.exterior_sides()

         if selector == 'inner':
            indices = self.inner_sides()

         if selector == 'all':
            indices = self.all_sides()

         if selector == 'top':
            indices = self.top_sides()

         if selector == 'bottom':
            indices = self.bottom_sides()

         if selector == 'left':
            indices = self.left_sides()

         if selector == 'right':
            indices = self.right_sides()

         if selector == 'custom':
            indices = custom


         #Delete the selected sides in previously defined boundaries
         for key,value in self.boundaries.items():
            for i in indices:
              if i in value:
               value.remove(i)
         #----------------

         #Set boundaries-----------------------------------
         for i in indices:
          self.boundaries.setdefault(region,[]).append(i) 
         #--------------------------------------------------

     def set_periodicity(self,direction : str,\
                                    region      : str):

         #Find periodicity vector--------
         if direction == 'y':
           points,lines = self.points_and_lines()
           minv = np.min(points[:,1])
           maxv = np.max(points[:,1])
           periodicity = np.array([0,maxv-minv])
         #-------------------------------- 

         #Find periodicity vector--------
         if direction == 'x':
           points,lines = self.points_and_lines()
           minv = np.min(points[:,0])
           maxv = np.max(points[:,0])
           periodicity = np.array([maxv-minv,0])

         points,lines = self.points_and_lines()

         outer_lines = self.exterior_sides() - 1 #We go from gmsh representation to program represetation

         output = []
         for l1,line in enumerate(outer_lines):
          b1  = np.mean(points[lines[line]],axis=0)   
          for l2 in np.arange(l1+1,len(outer_lines)):
                b2  = np.mean(points[lines[outer_lines[l2]]],axis=0)
                if np.allclose(periodicity,b2-b1):
                   if np.allclose(points[lines[l2]][0] - points[lines[l1]][0],b2-b1,atol=1e-3): #check the order of the line
                     output.append([l1+1,l2+1]) #+1 because of gmsh
                   else:  
                     output.append([l1+1,-(l2+1)]) #+1 because of gmsh
                if np.allclose(-periodicity,b2-b1):
                   if np.allclose(points[lines[l2]][0] - points[lines[l1]][0],b2-b1,atol=1e-3): #Check the order of line
                    output.append([l2+1,l1+1])
                   else:  
                    output.append([l2+1,-(l1+1)]) #+1 because of gmsh

         self.periodic_sides[region] = np.array(output).T

         #Delete the selected sides in previously defined boundaries
         for key,value in self.boundaries.items():
            for i in np.array(output).flatten():
              if abs(i) in value:
               value.remove(abs(i))

        
     def points_and_lines(self):    
         """Get all points and lines on the boundary"""

         points  = []
         lines   = []

         for i,point in enumerate(self.geometry.exterior.coords[:-1]):
                 points.append(point)
                 lines.append([len(points)-1,0 if i == len(self.geometry.exterior.coords) -2 else len(points)])
     
             
         for h,interior in enumerate(self.geometry.interiors):
             #An internal region named "dummy" means that it is indeed a hole
             if self.regions[h] == 'dummy':
              for i,point in enumerate(interior.coords[:-1]):
                 points.append(point)
                 lines.append([len(points)-1,0 if i == len(interior.coords) -2 else len(points)])


         return np.array(points),np.array(lines)

     def exterior_sides(self):
         """retrieve the indices of the outer lines"""

         indices = list(np.arange(0,len(self.geometry.exterior.coords) - 1)+1)

         return np.array(indices)


     def all_sides(self):
         """retrieve all hte sides"""
         points,lines = self.points_and_lines()

         return np.arange(len(lines)) +1

     def inner_sides(self):
         """retrieve the indices of the inner lines. It works only with rectangular 2D domain"""
        
         #Rationale: Subtract external sides from all sides
         points,lines = self.points_and_lines()

         indices = list(np.arange(len(lines))+1)
         for i in list(self.top_sides()) + list(self.bottom_sides())+ list(self.left_sides())  + list(self.right_sides()):
           indices.remove(i)
         return indices  
           
          
     def top_sides(self):
         """retrieve the indices of the top lines"""

         points,lines = self.points_and_lines()
         maxy = np.max(points[:,1])

         indices = []
         for n,i in enumerate(points[lines][:,:,1]):
             if np.allclose(i,[maxy,maxy]):
                 indices.append(n+1)

         return indices
   
     def bottom_sides(self):
         """retrieve the indices of the top lines"""

         points,lines = self.points_and_lines()
         maxy = np.min(points[:,1])

         indices = []
         for n,i in enumerate(points[lines][:,:,1]):
             if np.allclose(i,[maxy,maxy]):
                 indices.append(n+1)

         return indices
   
     def left_sides(self):
         """retrieve the indices of the top lines"""

         points,lines = self.points_and_lines()

         tmp = np.min(points[:,0])

         indices = []
         for n,i in enumerate(points[lines][:,:,0]):
             if np.allclose(i,[tmp,tmp]):
                 indices.append(n+1)

         return indices

     def right_sides(self):
         """retrieve the indices of the top lines"""

         points,lines = self.points_and_lines()
         maxy = np.max(points[:,0])

         indices = []
         for n,i in enumerate(points[lines][:,:,0]):
             if np.allclose(i,[maxy,maxy]):
                 indices.append(n+1)

         return indices

     def write_geo(self,**kwargs):
 
         if kwargs.setdefault('lz',0) == 0:
           strc = self.write_2D()
         else:  
          strc = self.write_3D(**kwargs)


     def write_3D(self,**kwargs):
         """Note this does not work with heat sources"""

         lz = kwargs['lz']
         n_points = 0
         n_lines  = 0
         n_loops  = 0
         n_surfaces  = 0
         z = -lz/2

         strc = ''
         strc +='h='+str(self.step) + ';\n'

         checkpoint_loops =  n_loops

         n_outer_points = len(self.geometry.exterior.coords)-1 
         checkpoint = n_points
         for k,p in enumerate(self.geometry.exterior.coords[:-1]) :  
            strc +='Point('+str(n_points+1) +') = {' + str(p[0]) +','+ str(p[1])+',' + str(z) + ',h};\n'
            n_points +=1
        
         #Outer Lines--
         for n in range(n_outer_points): 
            strc +='Line('+ str(n_lines+1) + ') = {' + str(n+1+checkpoint) +','+ str((n+1)%(n_outer_points)+1+checkpoint)+'};\n'
            n_lines +=1

         #Outer Loop 
         strc += 'Line Loop(' + str(n_loops + 1) + ') = {'
         for n in range(n_outer_points): 
            strc += str(n+1+checkpoint)
            strc += '};\n' if n == n_outer_points -1 else  ','
         n_loops +=1   

         #Interiors
         for h,hole in enumerate(self.geometry.interiors):

             #Inner points
             checkpoint = n_points
             for k,p in enumerate(hole.coords[:-1]):
                strc +='Point('+str(n_points+1) +') = {' + str(p[0]) +','+ str(p[1])+',' + str(z) + ',h};\n'
                n_points +=1

             #Inner lines
             for n in range(len(hole.coords)-1): 
                strc     +='Line('+ str(n_lines + 1) + ') = {' + str(n + checkpoint + 1) +','+ str((n + 1)%(len(hole.coords)-1)+checkpoint + 1)+'};\n'
                n_lines  +=1

             #Inner Loop 
             strc += 'Line Loop(' + str(n_loops+1) + ') = {'
             for n in range(len(hole.coords)-1): 
                strc += str(checkpoint+n+1)
                strc += '};\n' if n == len(hole.coords) - 2 else  ','
             n_loops +=1

         #Global surface
         strc += 'Plane Surface(' + str(n_surfaces+1) + ') = {' + str(checkpoint_loops + 1)
         for l in range(len(self.geometry.interiors)):
             strc +=  ',' + str(l+2 + checkpoint_loops) 
         strc += '};\n'  
         n_surfaces +=1

         #Perform extrusion--
         strc += 'Extrude {0,0,' + str(lz)  + '} { Surface{1};}\n'

         #Add physical volume
         strc += 'Physical Volume("Bulk") = {1};\n'

         #Mapping to 3D
         n_interior =0
         for i in self.geometry.interiors:
             n_interior += len(i.coords)-1
         n_exterior = len(self.geometry.exterior.coords) - 1      
         #------------------------------
         
         def maps(i):
             return 1 + 2*(n_exterior + n_interior ) +  abs(i)*4

         #Maps boundaries from 2D to 3D
         for key,value in self.boundaries.items():
             self.boundaries[key] =  [maps(i) for i in value]


         #-----------
         if not kwargs.setdefault('is_periodic_along_z',False):
             #If not periodic the name of top and bottom boundaries must be supplied
             self.boundaries.setdefault(kwargs['top_surface'],[]).append(maps(n_exterior+n_interior)+1)
             self.boundaries.setdefault(kwargs['bottom_surface'],[]).append(1)
         #-------------------


         #37-49 @1
         #45-73 @2
         for name,sides in self.boundaries.items():
           if len(sides) > 0:
             strc += r'''Physical Surface("''' + name + r'''") = {'''
             for s,side in enumerate(sides):
                 strc += str(side)
                 strc += '};\n' if s == len(sides) -1 else  ','


         points,lines = self.points_and_lines()
         #Periodic BCs
         for name,sides in self.periodic_sides.items():

             strc += r'''Physical Surface("''' + name + r'''_a") = {'''
             for s,side in enumerate(sides[0]):
                 tmp  = maps(side)
                 strc += str(tmp)
                 strc += '};\n' if s == len(sides[0]) -1 else  ','

             strc += r'''Physical Surface("''' + name  + r'''_b") = {'''
             for s,side in enumerate(sides[1]):
                 tmp  =  maps(side)
                 strc += str(tmp)
                 strc += '};\n' if s == len(sides[1]) -1 else  ','
             
             for s1,s2 in zip(*(sides[0],sides[1])):
                 ave1 = np.mean(points[lines[abs(s1)-1]],axis=0)
                 ave2 = np.mean(points[lines[abs(s2)-1]],axis=0)
                 ave_diff  = ave2-ave1 
                 strc += r'''Periodic Surface {''' + str(maps(s2)) + '} = {' + str(maps(s1)) + '} Translate {' + str(ave_diff[0]) + ',' + str(ave_diff[1]) + ',0};\n'

         if kwargs.setdefault('is_periodic_along_z',True):
             #Add periodic surface along z
             strc += r'''Physical Surface("''' + kwargs['name'] + r'''_a") = {''' + str(1) + '};\n'
             strc += r'''Physical Surface("''' + kwargs['name'] + r'''_b") = {''' + str(maps(n_exterior+n_interior)+1) + '};\n'
             strc += r'''Periodic Surface {''' + str(maps(n_exterior+n_interior)+1) + '} = {' + str(1) + '} Translate {0,0,' + str(lz) +  '};\n'

         #Create mesh
         with open("mesh.geo", 'w+') as f:
           f.write(strc)
    
         with open(os.devnull, 'w') as devnull:
           output = subprocess.check_output("gmsh -format msh2 -3 mesh.geo -o mesh.msh".split(), stderr=devnull)



     def write_2D(self):

         n_points = 0
         n_lines  = 0
         n_loops  = 0
         n_surfaces  = 0
         z = 0

         strc = ''
         strc +='h='+str(self.step) + ';\n'

         checkpoint_loops=  n_loops

         n_outer_points = len(self.geometry.exterior.coords)-1 
         checkpoint = n_points
         for k,p in enumerate(self.geometry.exterior.coords[:-1]) :  
            strc +='Point('+str(n_points+1) +') = {' + str(p[0]) +','+ str(p[1])+',' + str(z) + ',h};\n'
            n_points +=1
        
         #Outer Lines--
         for n in range(n_outer_points): 
            strc +='Line('+ str(n_lines+1) + ') = {' + str(n+1+checkpoint) +','+ str((n+1)%(n_outer_points)+1+checkpoint)+'};\n'
            n_lines +=1

         #Outer Loop 
         strc += 'Line Loop(' + str(n_loops + 1) + ') = {'
         for n in range(n_outer_points): 
            strc += str(n+1+checkpoint)
            strc += '};\n' if n == n_outer_points -1 else  ','
         n_loops +=1   

         #Interiors
         for h,hole in enumerate(self.geometry.interiors):

             #Inner points
             checkpoint = n_points
             for k,p in enumerate(hole.coords[:-1]):
                strc +='Point('+str(n_points+1) +') = {' + str(p[0]) +','+ str(p[1])+',' + str(z) + ',h};\n'
                n_points +=1

             #Inner lines
             for n in range(len(hole.coords)-1): 
                strc     +='Line('+ str(n_lines + 1) + ') = {' + str(n + checkpoint + 1) +','+ str((n + 1)%(len(hole.coords)-1)+checkpoint + 1)+'};\n'
                n_lines  +=1

             #Inner Loop 
             strc += 'Line Loop(' + str(n_loops+1) + ') = {'
             for n in range(len(hole.coords)-1): 
                strc += str(checkpoint+n+1)
                strc += '};\n' if n == len(hole.coords) - 2 else  ','
             n_loops +=1
     
             #Inner Surfaces
             if not self.regions[h] == 'dummy':
                #Elementary entity 
                strc += 'Plane Surface(' + str(n_surfaces+1) + ') = {'  + str(n_loops) + '};\n'   #this is the external one
                n_surfaces +=1

                #Physical entity 
                strc += r'''Physical Surface("''' + self.regions[h] + r'''") = {''' + str(n_surfaces) + '};\n'
                

         #Global surface
         strc += 'Plane Surface(' + str(n_surfaces+1) + ') = {' + str(checkpoint_loops + 1)
         for l in range(len(self.geometry.interiors)):
             strc +=  ',' + str(l+2 + checkpoint_loops) 
         strc += '};\n'  
         n_surfaces +=1

         strc += r'''Physical Surface("Bulk") = {''' + str(n_surfaces) + '};\n'
     
         #Boundary regions
         for name,sides in self.boundaries.items():
           if len(sides) > 0:
             strc += r'''Physical Line("''' + name + r'''") = {'''
             for s,side in enumerate(sides):
                 strc += str(side)
                 strc += '};\n' if s == len(sides) -1 else  ','

         #Periodic boundary
         for name,sides in self.periodic_sides.items():

            #Periodic Line
            strc += r'''Periodic Line{'''
            for l,side in enumerate(sides[0]):
              strc += str(side)
              strc += '} = {' if l == len(sides[0]) -1 else  ','

            for l,side in enumerate(sides[1]):
              strc += str(side)
              strc += '};\n' if l == len(sides[1]) -1 else  ','
            #-----------------

            #Physical Periodic Line
            strc += r'''Physical Line("''' + name +  r'''_a") = {'''
            for l,side in enumerate(sides[0]):
              strc += str(abs(side))
              strc += '};\n' if l == len(sides[0]) -1 else  ','

            strc += r'''Physical Line("''' + name + r'''_b") = {'''
            for l,side in enumerate(sides[1]):
              strc += str(abs(side))
              strc += '};\n' if l == len(sides[1]) -1 else  ','
         

         #Create mesh
         with open("mesh.geo", 'w+') as f:
           f.write(strc)
    
         with open(os.devnull, 'w') as devnull:
           output = subprocess.check_output("gmsh -format msh2 -2 mesh.geo -o mesh.msh".split(), stderr=devnull)





