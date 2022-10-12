from shapely.geometry import Polygon,Point
from shapely.ops import unary_union
from multiprocessing import Array,Value,current_process
from typing import List,NamedTuple,Tuple
from multiprocessing import Lock,Barrier,current_process
import numpy as np
import time
from sqlitedict import SqliteDict
from openbte import utils
from scipy.interpolate import LinearNDInterpolator
import plotly.graph_objs as go

f64   = np.float64
i64   = np.int64

class SharedMemory:
    """A dict-like class handling shared memory"""

    def __init__(self,n_process):

     self.barrier = Barrier(n_process)
     self.sv = {}
     self.lock = Lock()

    def keys(self):

        return self.sv.keys()
    def __setitem__(self,name,value):

        self.lock.acquire()
        if type(value) == np.ndarray:
          #Array   
          X_shape = value.shape 
          if name in self.sv.keys():
           X = self.sv[name]['data']
          else:
           X = Array(np.ctypeslib.as_ctypes_type(value.dtype),int(np.prod(X_shape)))
          X_np = np.frombuffer(X.get_obj(),dtype=value.dtype).reshape(X_shape)
          np.copyto(X_np,value)
          mode = 'array'
          dtype = value.dtype
        else: 
         #Scalar  
         if name in self.sv.keys():
          X = self.sv[name]['data']
          X.value = value
         else:
          X = Value(np.ctypeslib.as_ctypes_type(type(value)),value)

         mode = 'scalar'
         X_shape = None
         dtype = None

        self.sv.update({name:{'data':X,'shape':X_shape,'mode':mode,'dtype':dtype}})
        self.lock.release()

    def __getitem__(self,name):

       variable = self.sv[name]
       if variable['mode'] == 'scalar':
          return variable['data'].value
       else:
          return np.frombuffer(variable['data'].get_obj(),dtype=variable['dtype']).reshape(variable['shape'])

    def sync(self):
         """Barrier"""

         self.barrier.wait()

    def reduce(self,variables):

         self.barrier.wait()

         self.lock.acquire()
         for name,value in variables:
                 self[name][:] = np.zeros_like(value)
         self.lock.release()
                
         self.barrier.wait()
         self.lock.acquire()
         for name,value in variables:
                 self[name][:] += value

         self.lock.release()
         self.barrier.wait()

    @classmethod
    def print(cls,strc):
         if current_process().name == 'process_0':
          print(strc)

    @classmethod
    def append(cls,filename,value):
         if current_process().name == 'process_0':
             with  open(filename, "a")  as f:
                f.write(str(value) + "\n")



class MaterialRTA(NamedTuple):

    heat_capacity         : np.array
    scattering_time       : np.array
    group_velocity        : np.array
    frequency             : np.array
    thermal_conductivity  : np.array


class EffectiveThermalConductivity(NamedTuple):
    """
    Effective thermal conductivity
    """

    contact            : str                    

    normalization      : f64 = 0

class BoundaryConditions(NamedTuple):
    """
    Represents an employee.

    :param diffuse: The employee's ID number
    """

    diffuse      : str = ''

    mixed        : dict = {}                    

    periodic     : dict = {}

    is_nanowire  : bool = False

    def get_bc_by_name(self,name : str):
      """
        Returns whether the employee is an executive.

        :param name: The employee's ID number
      """

      if name in self.mixed.keys():
          return 'Mixed'
          
      for key in self.periodic.keys():
          return 'Periodic'

      raise ValueError(
                "{} failed (no region associated to side).".format(name))


class Mesh(NamedTuple):

    n_elems          : int
    n_nodes          : int
    dim              : int
    size             : np.ndarray
    nodes            : np.ndarray
    sides            : np.ndarray
    elems            : np.ndarray
    elem_side_map    : dict
    side_elem_map    : np.ndarray
    normal_areas     : np.ndarray
    gradient_weights : np.ndarray
    interp_weights   : np.ndarray
    elem_centroids   : np.ndarray
    side_centroids   : np.ndarray
    elem_volumes     : np.ndarray
    side_areas       : np.ndarray
    dists            : np.ndarray
    second_order_correction : np.ndarray
    internal         : np.array
    boundary_sides   : np.array
    periodic_sides   : dict = {}
    elem_physical_regions: dict = {}
    side_physical_regions: dict = {}

    def save(self,filename : str = 'state'):
      """Save data"""

      utils.save(filename,self._asdict())  

    def find_elem(self,p,guess):
     """Find the element that belongs to a point"""

     #First try in guess--
     for ne in guess:
      polygon = Polygon(self.nodes[self.elems[ne]])
      if polygon.contains(Point(p[0],p[1])):
       return ne
     #------------------      
     for ne,elem in enumerate(self.elems):
       polygon = Polygon(self.nodes[elem])
       if polygon.contains(Point(p[0],p[1])):
        return ne

     return -1

    def compute_neighbors(self,elem):
     """Find all neighbors to an elem"""

     elems = []
     for side in self.elem_side_map[elem]:
        for elem_2 in self.side_elem_map[side]:
            if not elem_2 in elems:
               elems.append(elem_2)
               
     return elems


    def get_decomposed_directions(self,kappa):
     """Compute decomposed directions"""
     """TODO: add the possibility to consider a subset"""

     v_orth     = np.einsum('ki,ij,kj->k',self.normal_areas,kappa,self.normal_areas)/np.einsum('ki,ki->k',self.normal_areas,self.dists)
     v_non_orth = np.einsum('ij,kj->ki',kappa,self.normal_areas) - np.einsum('ki,k->ki',self.dists,v_orth)

     return v_orth,v_non_orth


    def gradient(self,bcs          : BoundaryConditions,\
                 variable          : np.ndarray,\
                 jump              : bool = True):
     """ Compute grad via the least-square method """


     A = np.zeros((self.n_elems,self.n_elems,len(self.elems[0])))

     #Interior points
     diff_data = np.zeros((self.n_elems,len(self.elems[0])))
     for s in self.internal:
            i,j = self.side_elem_map[s]

            ind1    = list(self.elem_side_map[i]).index(s)
            ind2    = list(self.elem_side_map[j]).index(s)

            delta   = variable[j] - variable[i]

            diff_data[i,ind1] =   delta
            diff_data[j,ind2] =  -delta

            #A[i,i,ind1] =  -1
            #A[i,j,ind1] =   1
            #A[j,i,ind2] =   1
            #A[j,j,ind2] =  -1

     #B = np.zeros((self.n_elems,len(self.elems[0])))   
     #Periodic boundaries
     if jump:
      for region,sides in self.periodic_sides.items():
         
         jump = bcs.periodic[region]

         for s in sides:
          i,j     = self.side_elem_map[s]
          ind1    = list(self.elem_side_map[i]).index(s)
          ind2    = list(self.elem_side_map[j]).index(s)
          diff_data[i,ind1]  +=  jump  
          diff_data[j,ind2]  -=  jump
          #B[i,ind1] =  jump
          #B[j,ind2] = -jump


     #t = np.einsum('kls,l->ks',A,variable)
     #print(np.allclose(t,diff_data))
     #quit()

     #C = np.einsum('kjs,ks->kj',self.gradient_weights,B)

     #T = np.einsum('kjs,kls->kjl',self.gradient_weights,A)

     #gradient = np.einsum('kjl,l->kj',T,variable) + C


     gradient =  np.array([np.einsum('js,s->j',self.gradient_weights[k],dt) for k,dt in enumerate(diff_data)])

     #print(np.allclose(gradient,gradient2))



     #Let's attempt to build a matrix--

     #print(gradient)
     #quit()

     #Correction---
     #elems = self.side_elem_map[self.boundary_sides][:,0]
     #for _ in range(10):
      #diff_data_boundary  =  np.einsum('ci,ci->c',gradient[elems],self.dists[self.boundary_sides]*np.linalg.norm(self.dists[self.boundary_sides])/2)
     # diff_data_boundary  =  -np.einsum('ci,ci->c',gradient[elems],self.second_order_correction)
     # old_gradient = gradient.copy()
     # for k,(elem,side) in enumerate(zip(*(elems,self.boundary_sides))):
     #    ind = list(self.elem_side_map[elem]).index(side)
     #    diff_data[elem,ind] = diff_data_boundary[k]
     # gradient =  np.array([np.einsum('js,s->j',self.gradient_weights[k],dt) for k,dt in enumerate(diff_data)])
     # print(np.linalg.norm(gradient-old_gradient))    
        

     #print(np.min(diff_data))
     #print(np.max(diff_data))
     #quit()
     #Dirichlet boundaries
     #for key,value in bcs.dirichlet.items():
     #   for s in mesh.side_physical_regions[key]: 
     #     i = mesh.side_elem_map[s][0]  #i is the element
     #     delta = value - variable[i]
     #     ind    = list(mesh.elem_side_map[i]).index(s)
     #     diff_data[i,ind] =  delta

     return gradient


    def vorticity(self,
                  bcs               : BoundaryConditions,\
                  J                 : np.ndarray):
        """Compute vorticity (for now is only in 2D)"""

        n_elems,_ = J.shape
        vorticity = np.zeros((n_elems,3))

        grad_x = self.gradient(bcs,J[:,0],jump=False)
        grad_y = self.gradient(bcs,J[:,1],jump=False)

        #Defines only for 2D
        vorticity[:,2] = grad_y[:,0]-grad_x[:,1]
        return vorticity*1e9 #W/m/m/m



    def duplicate_cells(self,variables :dict,
                        repeat    :List):
      """Creat Super Cell"""

      #No repeatition case
      if repeat == [1,1,1]:
        return self.nodes,self.elems

      n_nodes = self.nodes.shape[0]
      nodes   = self.nodes.copy()
      elems   = self.elems.copy()

      for x in range(repeat[0]):
       for y in range(repeat[1]):
        for z in range(repeat[2]):
           if not (x == 0 and y == 0 and z == 0):
              P     = np.zeros(self.dim); 
              P[0] = x*self.size[0]
              P[1] = y*self.size[1]
              if self.dim == 3:
               P[2] = y*self.size[2]

              elems = np.vstack((elems,self.elems + len(nodes)))
              nodes = np.vstack((nodes,self.nodes + P[np.newaxis,:]))

      
      for n,(key, value) in enumerate(variables.items()):  
          if value['data'].ndim == 1:
           value['data'] = np.tile(value['data'],np.prod(repeat))
          else:
           value['data'] = np.tile(value['data'],(np.prod(repeat),1))


      return nodes,elems

    
    def cell_to_node(self,cell_variables  :dict, nodes: np.ndarray, elems: np.ndarray):
        """From cell data to node data"""

        #Init node variables--
        n_nodes = len(nodes)
        node_variables = {}
        for key,value in cell_variables.items():
          if value['data'].ndim == 1:
            node_variables[key] = {'data':np.zeros(n_nodes),'units':value['units']}
          if value['data'].ndim == 2:
            node_variables[key] = {'data':np.zeros((n_nodes,value['data'].shape[1])),'units':value['units']}
        #---------------------
    

        conn = np.zeros(n_nodes)
        for k,elem in enumerate(elems):
         conn[elem] +=1
         for key,value in node_variables.items():
          value['data'][elem[0]] += cell_variables[key]['data'][k]
          value['data'][elem[1]] += cell_variables[key]['data'][k]
          value['data'][elem[2]] += cell_variables[key]['data'][k]

        #Normalization
        for key,value in node_variables.items():
         if value['data'].ndim == 1:
            value['data'] = np.divide(value['data'],conn)
         elif value['data'].ndim == 2:
            value['data'] = np.divide(value['data'],conn[:,np.newaxis])

        return node_variables

    

class SolverResults(NamedTuple):

    kappa_eff    :float = 0
    aux          :dict = {}
    variables    :dict = {}


class Material(NamedTuple):

    thermal_conductivity :np.ndarray
    sigma                :np.ndarray
    vmfp                 :np.ndarray
    t_coeff              :np.ndarray
    mfps                 :np.ndarray
    n_angles             :int
    n_mfp                :int
    h                    :float
    heat_source_coeff    :float

class MaterialFull(NamedTuple):

    W            :np.ndarray
    C            :np.ndarray
    v            :np.ndarray
    kappa        :np.ndarray
    alpha        :np.ndarray
    h            :f64 = 1e20



class OpenBTEResults(NamedTuple):

    material      :Material
    mesh          :Mesh
    solvers       :dict = []

    @classmethod
    def load(cls,filename = 'state'):
        """Load Results"""

        results =  utils.load_readonly(filename)

        return cls(**results)

    def save(self,filename : str = 'state'):
      """Save Results"""

      utils.save(filename,self._asdict())  



    def get_variables(self):
        """Get all the variables"""   

        variables = {}
        for key,value in self.solvers.items():
          variables.update(value.variables)

        return variables



    def plot_over_line(self,**kwargs):
        """Plot over line"""

        #Get options--
        N           = kwargs.setdefault('N',100)
        direction   = kwargs.setdefault('direction','x')
        cut         = kwargs.setdefault('cut',0)
        repeat      = kwargs.setdefault('repeat',[1,1,1])

        variables   = kwargs['variables']
        p1          = kwargs['p1']
        p2          = kwargs['p2']

        #Collect results variables--
        cell_variables = self.get_variables()

        nodes,elems    = self.mesh.duplicate_cells(cell_variables,repeat) #changed in place

        node_variables = self.mesh.cell_to_node(cell_variables,nodes,elems)

        #adjust points 
        delta = 1e-4
        p1 = np.array(p1)
        p2 = np.array(p2)
        p1 = p1 + delta*(p2-p1)
        p2 = p2 - delta*(p2-p1)

        #Init output
        x = p1[np.newaxis,:] + np.einsum('i,k->ki',(p2-p1),np.arange(N))/(N-1)

        output = {variable:[] for variable in variables}

        for v,variable in enumerate(variables):
            output[variable] = LinearNDInterpolator(nodes,node_variables[variable]['data'])(x[:,0],x[:,1])

        #Compute x    
        linear_x = np.zeros(N)
        for n in range(N-1):
            linear_x[n+1] = linear_x[n] + np.linalg.norm(x[n+1]-x[n])

        return linear_x,output

    def vtu(self,filename   :str = 'output',\
        dof        :str = 'cell',\
        repeat     :List = [1,1,1]):

       variables = self.get_variables()
       mesh      = self.mesh
   
       #Duplicate cells---
       nodes,elems = mesh.duplicate_cells(variables,repeat)
       n_nodes = len(nodes)
       n_elems = len(elems)
       #------------------


       #Recover variables from solvers
       strc   ='# vtk DataFile Version 2.0\n'
       strc  +='OpenBTE Data\n'
       strc  +='ASCII\n'
       strc  +='DATASET UNSTRUCTURED_GRID\n'
       strc  +='POINTS ' + str(n_nodes) +   ' double\n'

       #write points--
       for n in range(n_nodes):
         for i in range(mesh.dim): 
           strc +=str(nodes[n,i])+' '
         if mesh.dim == 2:
            strc +='0.0'+' '
         strc +='\n'

       #write elems--
       if mesh.dim == 2:
         if len(elems[0]) == 3: 
          m = 3 
          ct = '5'
         else: 
          m = 4 
          ct = '9'

       elif mesh.dim == 3:
        m = 4 
        ct = '10'
       n = m+1

       strc +='CELLS ' + str(n_elems) + ' ' + str(n*n_elems) + ' ' +  '\n'

       for k in range(n_elems):
        strc +=str(m) + ' '
        for i in range(n-1): 
          strc +=str(elems[k][i])+' '
        strc +='\n'
           
       strc +='CELL_TYPES ' + str(n_elems) + '\n'
       for i in range(n_elems): 
         strc +=ct + ' '
       strc +='\n'


       #write data
       if dof == 'cell':
         strc +='CELL_DATA ' + str(n_elems) + '\n'
       else: 
         strc +='POINT_DATA ' + str(n_nodes) + '\n'
 
       for n,(key, value) in enumerate(variables.items()):

        name = key + '[' + value['units'] + ']'
        if value['data'].ndim == 1: #scalar
             strc +='SCALARS ' + name + ' double\n'
             strc +='LOOKUP_TABLE default\n'
             for i in value['data']:
              strc +=str(i)+' '
             strc +='\n'

        elif value['data'].ndim == 2: #vector
         strc +='VECTORS ' + name + ' double\n'
         for i in value['data']:
           tmp = np.array2string(i,max_line_width=1e6)
           strc +=tmp[1:-1] 
           if mesh.dim == 2 and  value['data'].shape[1] == 2: strc += '  0.0'
           strc += '\n'

        elif value['data'].ndim == 3: #tensor
         store +='TENSORS ' + name + ' double\n'
         for i in value['data']:
          for j in i:
           strc = np.array2string(j,max_line_width=1e6)
           store +=strc[1:-1]+'\n'
          store +='\n'

       with open(filename + '.vtk','w') as f:
        f.write(strc)



    def show(self,repeat : List = [1,1,1],\
        write_html :bool = False,\
        include    :List = []):
        """Show results on a browser"""

        #Select only those that are actually requested
        tmp_variables      = self.get_variables()  
        if len(include) == 0: include  = list(tmp_variables.keys())
        variables = {}
        for variable in include:
            variables[variable] = tmp_variables[variable]
        #------------------------------------------------
 

        nodes,elems    = self.mesh.duplicate_cells(variables,repeat)

        node_variables = self.mesh.cell_to_node(variables,nodes,elems)

        #Expand vector variables
        variables = {}
        for key,value in node_variables.items():
  
         if value['data'].ndim == 1: #scalar
           variables[key] = {'data':value['data'],'units':value['units']}
         elif value['data'].ndim == 2 : #vector 
           variables[key + '(x)'] = {'data':value['data'][:,0],'units':value['units']}
           variables[key + '(y)'] = {'data':value['data'][:,1],'units':value['units']}
           if value['data'].shape[1] == 3:  
            variables[key + '(z)'] = {'data':value['data'][:,2],'units':value['units']}
           mag = np.array([np.linalg.norm(value) for value in value['data']])
           variables[key + '(mag.)'] = {'data':mag,'units':value['units']}
        #--------------

        buttons = [dict(label=name,method="update",args=[{"visible":(np.eye(len(variables)) == 1)[k]},])   for k,name in enumerate(variables.keys())]

        nn = 0 #First item to show
        fig = go.Figure()

        for kk,(name,value) in enumerate(variables.items()):

         z = np.zeros(len(nodes))  if np.shape(nodes)[1] == 2 else nodes[:,2]
         fig.add_trace(go.Mesh3d(x=nodes[:,0],
                                 y=nodes[:,1],
                                 z=z,
                                 colorscale='Viridis',
                                 colorbar_title = '[' + value['units'] + ']' if len(value['units']) > 0 else "",
                                 intensity = value['data'],
                                 colorbar=dict(len=0.5),
                                 intensitymode='vertex',i=elems[:,0],j=elems[:,1],k=elems[:,2],
                                 name=name,
                                 showscale = True,
                                 visible= True if kk == nn else False
                        ))

        fig.update_layout(
         font=dict(
         family="Times New Roman",
         size=12,
         color='black'
        ) 
        )

        #x2 = 900

        #fig.update_layout(width=x1,height=x2,autosize=True,margin=dict(t=20, b=10, l=00, r=20),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')  
        fig.update_layout(autosize=False,height=400,width=600,margin=dict(t=0, b=10, l=0, r=0),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')  

        updatemenus=[dict(direction='down',active=nn,buttons=list(buttons),showactive=True,x=0)]
        #xanchor = 'right',
        #yanchor = 'top')]


        #update axes---------------
        axis = dict( 
          backgroundcolor="rgb(255, 255,255)",
          gridcolor="rgb(255, 255, 255)",
          zerolinecolor="rgb(255, 255, 255)",
          visible=False,
          showbackground=True
        )

        fig.update_layout(showlegend=False)

        # Update 3D scene options
        fig.update_scenes(
        aspectratio=dict(x=1, \
                     y=self.mesh.size[1]/self.mesh.size[0], \
                     z= 1 if self.mesh.dim == 2 else self.mesh.size[2]/self.mesh.size[0]),
        aspectmode="manual",xaxis=dict(axis),yaxis=dict(axis),zaxis=dict(axis))

        #camera
        camera = dict(
         up=dict(x=0, y=1, z=0),
         center=dict(x=0, y=0, z=0),
         eye=dict(x=0, y=0, z=1.5)
         )

        config = {'displayModeBar': False}

        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
        fig.update_layout(scene_camera=camera)
        fig.update_layout(updatemenus=updatemenus)

        fig.show(config=config)

        if write_html:

         #fig.update_layout(
         #font=dict(
         #family="Times New Roman",
         #size=12,
         #color="black"
         #) 
         #)

         fig.write_html("plotly.html",config=config,include_plotlyjs='cdn',\
                                      include_mathjax='cdn',\
                                      full_html=False)


















