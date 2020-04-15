import plotly.offline as py
import plotly.graph_objs as go
import numpy as np


def plotly_trisurf(nodes, simplices, data,name,units,visible=False):

    z = np.zeros(len(nodes))  if np.shape(nodes)[1] == 2 else nodes[:,2]
    return go.Mesh3d(x=nodes[:,0],
                      y=nodes[:,1],
                      z=z,
                      colorscale='Viridis',
                      colorbar_title = '[' + units + ']',
                     intensity = data,
                     colorbar=dict(len=0.5),
                     intensitymode='vertex',
                     i=simplices[:,0],
                     j=simplices[:,1],
                     k=simplices[:,2],
                     name=name,
                     visible=visible
                    )

def add_data(nodes,elems,name,data,units,buttons,fig,nv):

      plotly_data=plotly_trisurf(nodes,elems,data,name,units,visible= True if len(buttons) == 0 else False)
      fig.add_trace(plotly_data)
      visible = nv*[False]; visible[len(buttons)] = True
      buttons.append(dict(label=name,
                     method="update",
                     args=[{"visible":visible},
                           ]))


def plot_results(data,nodes,elems):

   dim = 2   if (max(nodes[:,2]) - min(nodes[:,2])) == 0 else 3
   #Get total number of variables to be plotted 
   nv = 0
   for key,value in data.items():
     if value['data'].ndim == 1:
         nv += 1   
     elif value['data'].ndim == 2:    
         nv += 1   
         nv += dim 

   buttons = []
   fig = go.Figure()
   current = 0
   for key,value in data.items():

     if value['data'].ndim == 1: #scalar
      add_data(nodes,elems,value['name'],value['data'],value['units'],buttons,fig,nv)
     elif value['data'].ndim == 2 : #vector 
         add_data(nodes,elems,value['name'] + '(x)',value['data'][:,0],value['units'],buttons,fig,nv)
         add_data(nodes,elems,value['name'] + '(y)',value['data'][:,1],value['units'],buttons,fig,nv)
         if dim == 3: add_data(value['name']+ '(x)',value['data'][:,2],value['units'],buttons,fig,nv)
         add_data(nodes,elems,value['name'] + '(mag.)',[np.linalg.norm(value) for value in value['data']],value['units'],buttons,fig,nv)

   updatemenus=[dict(direction='down',active=0,x=0.1,y=0.95,buttons=list(buttons))]
   fig.update_layout(updatemenus=updatemenus)



   fig.update_layout(
    width=800,
    height=800,
    autosize=False,
    margin=dict(t=0.1, b=0, l=0, r=0)
   )  

   fig.update_layout(
    font=dict(
        family="Courier New, monospace",
        size=16,
        color="#7f7f7f"
    )
   )


   fig.update_layout(
    title={
        'text': "OpenBTE",
        'y':0.94,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})


   #update axes---------------
   axis = dict(
          showbackground=True,
          backgroundcolor="rgb(230, 230,230)",
          gridcolor="rgb(255, 255, 255)",
          zerolinecolor="rgb(255, 255, 255)",
         )

   #axis = dict(ticktext=[],tickvals= [])

   # Update 3D scene options
   fig.update_scenes(
    aspectratio=dict(x=1, y=1, z=0.5),
    aspectmode="manual",
    xaxis=dict(axis),
    yaxis=dict(axis),
    zaxis=dict(axis))



   #camera
   camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=0, z=3)
   )

   fig.update_layout(scene_camera=camera)

   py.iplot(fig, filename='OpenBTE')


