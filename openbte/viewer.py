import plotly.offline as py
import plotly
import plotly.graph_objs as go
import numpy as np
import sys
import openbte.utils as utils
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD

def get_surface_nodes(variables,geometry):

     sides = list(geometry['boundary_sides']) + \
                     list(geometry['periodic_sides']) + \
                     list(geometry['inactive_sides'])

     nodes = geometry['sides'][sides].flat


     triangles = np.arange(len(nodes)).reshape((len(sides),3))
     
     for key in variables.keys():
        solver['variables'][key]['data'] = solver['variables'][key]['data'][nodes]


     geometry['nodes'] = geometry['nodes'][nodes]
     geometry['elems'] = np.array(triangles)


def plot_results(solver,geometry,material,options_maps):

 if comm.rank == 0:  
   #Parse options--
   repeat   = options_maps.setdefault('repeat',[1,1,1])
   displ    = options_maps.setdefault('displ',[0,0,0])
   #---------------

   dim = int(geometry['meta'][2])

   data = utils.extract_variables(solver)

   data = utils.expand_variables(data,geometry) #this is needed to get the component-wise data for plotly
   
   utils.get_node_data(data,geometry)

   if np.prod(repeat) > 1:
      utils.duplicate_cells(geometry,data,repeat,displ)

   if dim == 3:
      get_surface_nodes(variables,geometry)


   nodes = geometry['nodes']
   elems = np.array(geometry['elems'])
   size  = [max(nodes[:,i]) - min(nodes[:,i])  for i in range(3)] 
   dim   = 2   if size[2] == 0 else 3


   nn = 0
   nv = len(data.keys())
   #----------------------

   #Create button
   buttons = [dict(label=name,method="update",args=[{"visible":(np.eye(nv) == 1)[k]},])   for k,name in enumerate(data.keys())]


   fig = go.Figure()

   for kk,(name,value) in enumerate(data.items()):

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
        family="Courier New, monospace",
        size=12,
        color="black"
    )
   )


   fig.update_layout(
    title={
        'text': "OpenBTE",
        'y':0.93,
        'x':0.88,
        'xanchor': 'center',
        'yanchor': 'top'})


   #if argv.setdefault('large',False):
   #    x1 = 700
   #    x2 = 500
   #else:    
   x1 = 400
   x2 = 450

   fig.update_layout(width=x1,height=x2,autosize=True,margin=dict(t=20, b=10, l=00, r=20),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')  

   updatemenus=[dict(direction='down',active=nn,x=0.4,y=1.15,buttons=list(buttons),showactive=True)]


   #update axes---------------
   axis = dict( 
          backgroundcolor="rgb(255, 255,255)",
          gridcolor="rgb(255, 255, 255)",
          zerolinecolor="rgb(255, 255, 255)",
          visible=False,
          showbackground=True
         )

   #axis = dict(ticktext=[],tickvals= [],showbackground=False)

   dirr = int(geometry['meta'][-1])
   bb = str(round(solver['kappa'][0],2))+' W/m/K' if 'kappa' in solver.keys() else '--'

   #Add thermal conductivity value------
   meta  = 'Bulk: ' + str(round(material['kappa'][dirr,dirr],2)) +' W/m/K<br>Fourier: '\
                    +       str(round(solver['kappa_fourier'][0],2)) + ' W/m/K<br>BTE:' \
                    +       bb

   fig.add_annotation(
            x=0.97,
            y=0.05,
            xref='paper',
            showarrow=False,
            yref='paper',
            text=meta,align='left')
   #-------------------------------------




   fig.update_layout(showlegend=False)

   # Update 3D scene options
   fig.update_scenes(
    aspectratio=dict(x=1, \
                     y=size[1]/size[0], \
                     z= 1 if dim == 2 else size[2]/size[0]),
    aspectmode="manual",
    xaxis=dict(axis),
    yaxis=dict(axis),
    zaxis=dict(axis))

   #camera
   camera = dict(
    up=dict(x=0, y=1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=0, z=2)
   )

   fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
   fig.update_layout(scene_camera=camera)
   fig.update_layout(updatemenus=updatemenus)

   #if argv.setdefault('write_html',False):
   # fig.write_html("plotly.html")
   #plotly.io.write_json(fig,'ff.json')

   # plotly.io.to_image(fig,format='png')
   #fig.write_image("test.png",scale=5)
   if 'google.colab' in sys.modules:
     fig.show(renderer='colab')
   else:
     fig.show(renderer='browser')


     #    output.update({'bte':solver['kappa_bte'][-1]}) 





