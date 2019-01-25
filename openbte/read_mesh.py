import numpy as np

nodes = {}; elem_sides_map = {}; side_nodes_map = {}
node_elems_map = {}; node_sides_map = {}; elem_nodes_map = {}; side_elems_map = {}

with open('mesh.msh') as f:
 [f.readline() for n in range(13)] 
 nodes = {n:np.array(f.readline().split()[1:4],'float') for n in range(int(f.readline()))}
 [f.readline() for n in range(2)] 
 for i in range(int(f.readline())):
  e = np.array(f.readline().split()[5:8],'int')
  [node_elems_map.setdefault(p,set()).add(i) for p in e] 
  [elem_nodes_map.setdefault(i,set()).add(p) for p in e] 
  elem = []   
  for n in range(3):   
   tmp = sorted(np.roll(e,n)[0:2])
   if not tmp in side_nodes_map.values():
    side_nodes_map.update({len(side_nodes_map):tmp})   
    for p in tmp: node_sides_map.setdefault(p,set()).add(len(side_nodes_map))
    ns = len(side_nodes_map) 
   else:
    ns = list(side_nodes_map.keys())[list(side_nodes_map.values()).index(tmp)]
   elem_sides_map.setdefault(i,set()).add(ns)
   side_elems_map.setdefault(ns,set()).add(i)




    




