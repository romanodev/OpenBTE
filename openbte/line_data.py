import openbte.utils as utils
import numpy as np
from mpi4py import MPI
import matplotlib.tri as mtri

comm = MPI.COMM_WORLD




def plot_line_data(line_data):
 

 if comm.rank == 0:
   from pubeasy import MakeFigure

   fig = MakeFigure()

   data = line_data['line_data']
   L    = line_data['L']

   for c,(key,values) in enumerate(data.items()):
    minl=1e9;maxl=-1e9
    for n,d in enumerate(values):
     minl = min([minl,min(L[n])])   
     maxl = max([maxl,max(L[n])])   
     fig.add_plot(L[n],d,name=key,color=fig.colors[c%3])

   fig.add_labels('Distance [nm]','data') 

   fig.finalize(grid=True,show=True,xlim=[minl,maxl])




def compute_line_data(solver,geometry,compute_line_data_options)->'line_data':

 data = None 
 if comm.rank == 0:
  #Parse options--
  dof      = compute_line_data_options.setdefault('dof','nodes')
  repeat   = compute_line_data_options.setdefault('repeat',[1,1,1])
  displ    = compute_line_data_options.setdefault('displ',[0,0,0])
  direction= compute_line_data_options.setdefault('direction','y')
  N        = compute_line_data_options.setdefault('N',100)
  delta = 1e-4
  if direction =='y':
     x  = compute_line_data_options.setdefault('x',0) 
     p1 = np.array([x,-geometry['size'][1]/2+delta]) 
     p2 = np.array([x,+geometry['size'][1]/2-delta])
  elif direction =='x':
     y  = compute_line_data_options.setdefault('y',0) 
     p1 = np.array([-geometry['size'][0]/2+delta,y]) 
     p2 = np.array([geometry['size'][0]/2-delta,y])
  #--------------------

  data_master =  utils.extract_variables(solver)
  data_master =  utils.expand_variables(data_master,geometry)

  if dof=='nodes':  
    utils.get_node_data(data_master,geometry)

  #repeat
  if np.prod(repeat) > 1:
      utils.duplicate_cells(geometry,data_master,repeat,displ)
  
  variables = compute_line_data_options['variables']

  line_data = {}
  for variable in variables:
    data = data_master[variable]['data']

    guess = []
    old_mat = 1
    p_old = [-1e3,-1e3]
    crossing = []
    pair = []

    L = []
    tmp_L = []
    output = []
    tmp_output = []
    already_passed = False
    dx = []
    for kk in range(N):

     p = p1 + (p2 - p1)*kk/(N-1)

     if len(dx) == 0:
      dx.append(0)
     else: 
      dx.append(dx[-1] + np.linalg.norm(p-p_old))
    
     elem,x,y,found = utils.find_elem(geometry,p,guess)
   
     if found:

      already_passed = False
      new_elem = elem

      nodes = geometry['elems'][elem]

      if len(nodes) == 3:
       d = mtri.LinearTriInterpolator(mtri.Triangulation(x, y,[[0,1,2]]),data[nodes])(p[0],p[1]).data
      else:
       d = interpolate.interp2d(x, y, data[nodes], kind='linear')(p[0],p[1])[0]

      tmp_output.append(float(d))

      tmp_L.append(dx[-1])

     else:
         if not already_passed :
          L.append(tmp_L)   
          output.append(tmp_output)
          tmp_output = []
          tmp_L = []
          already_passed = True

     guess = utils.compute_neighbors(geometry,elem)  

     old_elem = new_elem
     p_old = p.copy()

     L.append(tmp_L)   
     output.append(tmp_output)

    #store variable--
    name = variable + ' [' +data_master[variable]['units'] + ']'
    line_data[name] = output
    #----------------


  return {'line_data':line_data,'L':L}




