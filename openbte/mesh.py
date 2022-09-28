import os
import subprocess
from openbte.objects import BoundaryConditions,List,Mesh
from openbte import Geometry
import numpy as np

def get_normal(nodes,lines,dim) :
          """Get the normal to a surface or line"""

          if dim == 2:
               n1 = nodes[np.array(lines[0],dtype=int)[0] - 1]
               n2 = nodes[np.array(lines[1],dtype=int)[0] - 1]
               v1 = list(n1-n2); v1.append(0)
               v = np.cross(v1,[0,0,1])[:2]
               v /= np.linalg.norm(v)
               return np.absolute(v)


          if dim == 3:
           for n,l in enumerate(lines):
               #The reason for this for is that sometimes nodes can be collinear
               n1 = nodes[np.array(lines[n],dtype=int)[0] - 1]
               n2 = nodes[np.array(lines[(n+1)%len(lines)],dtype=int)[0] - 1]
               n3 = nodes[np.array(lines[(n+2)%len(lines)],dtype=int)[0] - 1]
               v = np.cross(n1-n2,n1-n3)
               if np.linalg.norm(v) > 1e-3:
                   v /= np.linalg.norm(v)
                   return np.absolute(v)

def get_periodicity(nodes,lines,dim) :
          """Get the normal to a surface or line"""

          if dim == 2:
               n1 = nodes[np.array(lines[0],dtype=int)[0] - 1]
               n2 = nodes[np.array(lines[1],dtype=int)[0] - 1]
               v1 = list(n1-n2); v1.append(0)
               v = np.cross(v1,[0,0,1])[:2]
               v /= np.linalg.norm(v)
               return np.absolute(v)


          if dim == 3:
           for n,l in enumerate(lines):
               #The reason for this for is that sometimes nodes can be collinear
               n1 = nodes[np.array(lines[n],dtype=int)[0] - 1]
               n2 = nodes[np.array(lines[(n+1)%len(lines)],dtype=int)[0] - 1]
               n3 = nodes[np.array(lines[(n+2)%len(lines)],dtype=int)[0] - 1]
               v = np.cross(n1-n2,n1-n3)
               if np.linalg.norm(v) > 1e-3:
                   v /= np.linalg.norm(v)
                   return np.absolute(v)


def get_mesh()->Mesh:
    """Build mesh with gmsh"""

    #Dimension
    #Import mesh
    with open('mesh.msh', 'r') as f: lines = f.readlines()
    #Physical surfaces--
    lines = [l.split()  for l in lines]
    nb = int(lines[4][0])
    current_line = 5
    blabels = {int(lines[current_line+i][1]) : lines[current_line+i][2].replace('"',r'') for i in range(nb)}
    #-------------------

    #Nodes--
    current_line += nb+2
    n_nodes = int(lines[current_line][0])
    current_line += 1
    nodes = np.array([lines[current_line + n][1:4] for n in range(n_nodes)],float)

    #Guess the dimension (it assumes that 2D shapes lie on the xy plane
    if np.allclose(nodes[:,2],np.zeros(nodes.shape[0])):
       dim = 2
       nodes = nodes[:,:dim]
    else:   
       dim = 3  

    current_line += n_nodes+1

    #Size
    size = np.zeros(3)
    for i in range(dim):
      size[i] = np.max(nodes[:,i]) - np.min(nodes[:,i])

    #Elements
    current_line += 1
    n_elem_tot = int(lines[current_line][0])
    current_line += 1
    bulk_type = {2:[2,3],3:[4]}
    face_type = {2:[1],3:[2]}
    bulk_tags = [n for n in range(n_elem_tot) if int(lines[current_line + n][1]) in bulk_type[dim]] 
    face_tags = [n for n in range(n_elem_tot) if int(lines[current_line + n][1]) in face_type[dim]]
    elems = [list(np.array(lines[current_line + n][5:],dtype=int)-1) for n in bulk_tags]
    n_elems = len(elems)

    #Create maps between sides and elements    
    elem_side_map = { i:[] for i in range(len(elems))}
    sides = []
    tmp_indices = []
    for k,elem in enumerate(elems):
       tmp = list(elem)
       for i in range(len(elem)):
         tmp.append(tmp.pop(0))
         trial = sorted(tmp[:dim])
         sides.append(trial)
         tmp_indices.append(k)
    sides,inverse = np.unique(sides,axis=0,return_inverse = True)
    n_sides = len(sides)
    for k,s in enumerate(inverse): 
        elem_side_map[tmp_indices[k]].append(s)

    #Node->Sides
    node_side_map = { i:[] for i in range(len(nodes))}
    for s,side in enumerate(sides): 
        for t in side:
            node_side_map[t].append(s)

    #Side elem map
    tmp = { i:[] for i in range(len(sides))}
    for key, value in elem_side_map.items():
      for v in value:
          tmp[v].append(key)
    side_elem_map = np.ones((n_sides,2),dtype=int) 
    for key,values in tmp.items():
        side_elem_map[key] = np.array(values) #it broadcasts in case of boundary sides


    #Bulk Physical regions
    elem_physical_regions = {}
    for n in bulk_tags:
       tag = int(lines[current_line + n][3])
       elem_physical_regions.setdefault(blabels[tag],[]).append(n-len(face_tags))


    #Side Physical regions
    boundary_sides = [ sorted(np.array(lines[current_line + n][5:],dtype=int)-1) for n in face_tags] 
    side_physical_regions = {}
    periodic_physical_regions = {}
    sides_list = sides.tolist()
    for t in face_tags:
        tag = int(lines[current_line + t][3])
        side_physical_regions.setdefault(blabels[tag],[]).append(sides_list.index(boundary_sides[t]))

    #Side areas
    if dim == 2:
     side_areas = np.array([np.linalg.norm(nodes[s[1]] - nodes[s[0]]) for s in sides])
    else:   
     p =nodes[sides]
     side_areas = np.linalg.norm(np.cross(p[:,0]-p[:,1],p[:,0]-p[:,2]),keepdims=True,axis=1).T[0]/2
  
    
    #Elem volumes
    elem_volumes = np.zeros(n_elems)
    if dim == 3: #Assuming Tetraedron
     M = np.ones((4,4))
     for i,e in enumerate(elems):
      M[:,:3] = nodes[e,:]
      elem_volumes[i] = 1.0/6.0 * abs(np.linalg.det(M))

    if dim == 2: 
     M = np.ones((3,3))
     for i,e in enumerate(elems):
      if len(e) == 3:     
       M[:,:2] = nodes[e,0:2]
       elem_volumes[i] = abs(0.5*np.linalg.det(M))
      else:
       points = nodes[e]
       x = points[:,0]
       y = points[:,1]
       elem_volumes[i] = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

    #Side centroids
    side_centroids = np.array([np.mean(nodes[i],axis=0) for i in sides])

    #Elem centroids
    elem_centroids = np.array([np.mean(nodes[i],axis=0) for i in elems])

    #Face normals
    v1 =  nodes[sides[:,1]]-nodes[sides[:,0]]
    if dim == 3:
     v2 =  nodes[sides[:,2]]-nodes[sides[:,0]]
    else :
     v2 = np.array([0,0,1]).T
    v = np.cross(v1,v2)
    normal = (v.T/np.linalg.norm(v,axis=1))[:dim]
    
    c = side_centroids - elem_centroids[side_elem_map[:,0]] #with respect the first elements
    index = np.where(np.einsum('iu,ui->u',normal,c) < 0)[0]
    normal[:,index] = -normal[:,index]
    normal= normal.T


    normal_areas = np.einsum('si,s->si',normal,side_areas)

    #Compute dists
    dists = np.zeros((n_sides,dim)) 
    for s in range(n_sides):
        e1,e2 = side_elem_map[s]
        if e1 == e2:
         dists[s] =   normal[s]*np.dot(normal[s],side_centroids[s]  - elem_centroids[e1])
        else:
         dists[s] = elem_centroids[e2] - elem_centroids[e1]

    #Periodic sides-----------------------------------------------------------------------------------
    to_discard  = []; 
    periodic_sides_dict = {}

    if len(lines) > current_line+n+2:

     periodic_sides  = []; 

     n_periodic_sides = int(lines[current_line+n+3][0])


     a1 = []
     a2 = []
     #print(n_periodic_sides)
     #checkpoint = current_line+n+6
     checkpoint = current_line+n+3 + dim
     for ss  in range(n_periodic_sides):

         n_periodic_nodes = int(lines[checkpoint][0])
        
         #Get normal to the surface
         periodic_normal = get_normal(nodes,lines[checkpoint+1:checkpoint+1+n_periodic_nodes],dim)
         

         #Rationale: we take the nodes from gmsh and find the associated sides that are periodic
         #As a point may belong to different boundary sides (e.g. corner points), we also have to check collinearity
         for i in range(n_periodic_nodes):
         
           tags   = np.array(lines[checkpoint + 1 + i],dtype=int) - 1
           L      = nodes[tags[0]] - nodes[tags[1]]

           sides_1 = set(node_side_map[tags[0]])#-1 takes into account the fact that nodes' tags in gmsh start with 1
           sides_2 = set(node_side_map[tags[1]])

           #Sort collinearity out
           corr = []
           for s1 in sides_1:
            for s2 in sides_2:
                if np.allclose(side_centroids[s1],side_centroids[s2] + L):
                    corr.append([s1,s2])

           for s1,s2 in corr:

               for key,value in side_physical_regions.items():

                  if key[-2:] == '_b': #To keep

                     if (s1 in value) and not (s1 in periodic_sides) :
                         
                        periodic_sides_dict.setdefault(key[:-2],[]).append(s1)
                        periodic_sides.append(s1)
                        to_discard.append(s2)

                     if (s2 in value) and not (s2 in periodic_sides) :
                         
                        periodic_sides_dict.setdefault(key[:-2],[]).append(s2)
                        periodic_sides.append(s2)
                        to_discard.append(s1)


         checkpoint += n_periodic_nodes + dim #Just a coincidence that dim works here

     #Update physical side dictionaries
     for key,value in periodic_sides_dict.items():
        side_physical_regions.pop(key+'_a',None)
        side_physical_regions.pop(key+'_b',None)

     #Update other quantities
     for n,(s1,s2) in enumerate(zip(*(to_discard,periodic_sides))):
        #update map
        e2 = side_elem_map[s2][0]
        e1 = side_elem_map[s1][0]
        side_elem_map[s2][1] = e1
        elem_side_map[e1][elem_side_map[e1].index(s1)]=s2

        dists[s2] = elem_centroids[e1] + side_centroids[s2]-side_centroids[s1] - elem_centroids[e2]
   
    #Get boundary side indices
    boundary_sides_indices = []
    for key,value in side_physical_regions.items():
        boundary_sides_indices += value

    #Compute second-order correction for gradients---
    second_order_correction = np.zeros((len(boundary_sides_indices),dim))
    for k,s in enumerate(boundary_sides_indices):
        e1,_ = side_elem_map[s]
        second_order_correction[k] = dists[s] - (side_centroids[s]  - elem_centroids[e1])
    #------------------------------------------------


    #Identify all internal sides (e.g. no associated to boundary conditions)
    internal = np.arange(n_sides)[~np.isin(np.arange(n_sides),boundary_sides_indices + to_discard)]

    #Compute interpolation weights
    w              = elem_centroids[side_elem_map[:,0]] - nodes[sides[:,0]]
    tmp2           = np.einsum('ui,ui->u',normal_areas,w)
    tmp            = np.einsum('ui,ui->u',normal_areas,dists)
    interp_weights = 1+tmp2/tmp

    # compute_least_square_weigths
    diff_dist = np.zeros((n_elems,len(elems[0]),dim))
    for s in range(n_sides):
     if not s in to_discard:   
            e1,e2 = side_elem_map[s]
            ind1 = list(elem_side_map[e1]).index(s)

            if not e1 == e2:
             diff_dist[e1,ind1] =   dists[s]
             ind2 = list(elem_side_map[e2]).index(s)
             diff_dist[e2,ind2] =  -dists[s]
            else:
             diff_dist[e1,ind1] = dists[s]

    gradient_weights = np.linalg.pinv(diff_dist)

    #ind = np.unravel_index(np.argmax(gradient_weights, axis=None),gradient_weights.shape)

    elems = np.array(elems)

    return Mesh(n_elems,
                    n_nodes,
                    dim,
                    size,
                    nodes,
                    sides,
                    elems,
                    elem_side_map,
                    side_elem_map,
                    normal_areas,
                    gradient_weights,\
                    interp_weights,\
                    elem_centroids,\
                    side_centroids,\
                    elem_volumes,\
                    side_areas,\
                    dists,\
                    second_order_correction,\
                    internal,\
                    boundary_sides_indices,\
                    periodic_sides_dict,\
                    elem_physical_regions,\
                    side_physical_regions)


 


def get_geo(geometry: Geometry, boundary: dict = {}, periodicity: dict = {})->int:
   """Write mesh given polygons and boundary conditions"""

   step = 0.05
   strc = ''
   strc +='h='+str(step) + ';\n'

   polygons = geometry.polygons

   #Write points--
   n_point = 1
   points_map  = [] #to map points and lines
   surface_map = [0]*len(polygons)
   internal_region_map = {}
   for i,poly in enumerate(polygons):
    #External--------------- 
    points = list(poly.exterior.coords)[:-1]
    local_map = []
    for k,p in enumerate(points) :  
      local_map.append(n_point) 
      strc +='Point('+str(n_point) +') = {' + str(p[0]) +','+ str(p[1])+',0,h};\n'
      n_point +=1
    points_map.append(local_map)  
    #---------------------
   
    #Internals--------------- 
    for hole in list(poly.interiors):
         surface_map[i] +=1
         local_map = []
         points = list(hole.coords)[:-1]
         for k,p in enumerate(points) :  
           local_map.append(n_point) 
           strc +='Point('+str(n_point) +') = {' + str(p[0]) +','+ str(p[1])+',0,h};\n'
           n_point +=1
         points_map.append(local_map)  
         
   #Write lines
   n_line = 1
   for local_map in points_map:
       for n in range(len(local_map)): 
        strc +='Line('+str(n_line) +') = {' + str(local_map[n]) +','+ str(local_map[(n+1)%len(local_map)])+'};\n'
        n_line +=1

   #Write loops&surfaces
   n_loop = 1
   for local_map in points_map:
     strc += 'Line Loop(' + str(n_loop) + ') = {'
     for n in range(len(local_map)): 
         strc += str(local_map[n])
         strc += '};\n' if n == len(local_map) -1 else  ','
     n_loop +=1    

   #Build surfaces
   n_hole = 1
   n_surface = 1
   for holes in surface_map:
     strc += 'Plane Surface(' + str(n_surface) + ') = {' + str(n_hole)   #this is the external one
     n_hole +=1
     if holes == 0:
        strc += '};\n'
     else:   
        strc += ','
        for n in range(holes): 
         strc += str(n_hole)
         strc += '};\n' if n == holes -1 else  ','
         n_hole    +=1    
     n_surface +=1    

   #Interior regions
   for h,holes in enumerate(surface_map):
       for hole in range(holes):
        if not geometry.interior_regions[h] == '__empty__':
            strc += 'Plane Surface(' + str(n_surface) + ') = {' + str(n_hole-1) + '};\n'  #this is the external one
            strc += r'''Physical Surface("''' + geometry.interior_regions[h] + r'''") = {''' + str(n_surface) + '};\n'
            n_surface +=1    
         
   #Volume regions surface
   strc += r'''Physical Surface("Bulk") = {'''
   for l in range(len(polygons)):
     strc += str(l+1)
     strc += '};\n' if l == len(polygons)-1 else  ','
   
   #Boundary regions
   for name,sides in boundary.items():
       if len(sides) > 0:
        strc += r'''Physical Line("''' + name + r'''") = {'''
        for s,side in enumerate(sides):
            strc += str(side)
            strc += '};\n' if s == len(sides) -1 else  ','

   for name,sides in periodicity.items():

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
     


   return 1
  

