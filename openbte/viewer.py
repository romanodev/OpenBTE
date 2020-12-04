import plotly.offline as py
import plotly
import plotly.graph_objs as go
import numpy as np
import sys
from .utils import *
import time


def plotly_trisurf(nodes, simplices, data,name,units,visible=False):


    z = np.zeros(len(nodes))  if np.shape(nodes)[1] == 2 else nodes[:,2]
    return go.Mesh3d(x=nodes[:,0],
                      y=nodes[:,1],
                      z=z,
                      colorscale='Viridis',
                      colorbar_title = '[' + units + ']' if len(units) > 0 else "",
                     intensity = data,
                     colorbar=dict(len=0.5),
                     intensitymode='vertex',
                     i=simplices[:,0],
                     j=simplices[:,1],
                     k=simplices[:,2],
                     name=name,
                     showscale = True,
                     visible=visible
                    )

def add_data(nodes,elems,name,data,units,buttons,fig,nv):

      plotly_data=plotly_trisurf(nodes,elems,data,name,units,visible= True if (len(buttons) == nv-5 or nv == 1) else False)
      fig.add_trace(plotly_data)
      visible = nv*[False]; visible[len(buttons)] = True
      buttons.append(dict(label=name,
                     method="update",
                     args=[{"visible":visible},]))
      return  plotly_data


def plot_results(master,**argv):

   
   data  = master['variables']
   nodes = master['nodes']
   elems = master['elems']

  
   size = [max(nodes[:,i]) - min(nodes[:,i])  for i in range(3)] 
   dim = 2   if size[2] == 0 else 3

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
   #plotly_tot_tmp = {}
  
   for key,value in data.items():
     if value['data'].ndim == 1: #scalar
      plotly_data = add_data(nodes,elems,value['name'],value['data'],value['units'],buttons,fig,nv)
     elif value['data'].ndim == 2 : #vector 
         plotly_data = add_data(nodes,elems,value['name'] + '(x)',value['data'][:,0],value['units'],buttons,fig,nv)
         plotly_data = add_data(nodes,elems,value['name'] + '(y)',value['data'][:,1],value['units'],buttons,fig,nv)
         if dim == 3: 
             plotly_data = add_data(nodes,elems,value['name']+ '(z)',value['data'][:,2],value['units'],buttons,fig,nv)

         mag = [np.linalg.norm(value) for value in value['data']]
         plotly_data = add_data(nodes,elems,value['name'] + '(mag.)',mag,value['units'],buttons,fig,nv)

   
   #-----------------------

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


   if argv.setdefault('large',False):
       x1 = 700
       x2 = 500
   else:    
       x1 = 400
       x2 = 450

   fig.update_layout(width=x1,height=x2,autosize=True,margin=dict(t=20, b=10, l=00, r=20),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')  


   updatemenus=[dict(direction='down',active=nv-5 if nv > 1 else 0,x=0.4,y=1.15,buttons=list(buttons),showactive=True)]


   #update axes---------------
   axis = dict( 
          backgroundcolor="rgb(255, 255,255)",
          gridcolor="rgb(255, 255, 255)",
          zerolinecolor="rgb(255, 255, 255)",
          visible=False,
          showbackground=True
         )

   #axis = dict(ticktext=[],tickvals= [],showbackground=False)

   bb = str(round(argv['bte'],2))+' W/m/K' if 'bte' in argv.keys() else '--'
   fig.add_annotation(
            x=0.97,
            y=0.05,
            xref='paper',
            showarrow=False,
            yref='paper',
            text='Bulk: ' + str(round(argv['bulk'],2)) +' W/m/K<br>Fourier: '\
                    +       str(round(argv['fourier'],2)) + ' W/m/K<br>BTE:' \
                    +       bb,align='left')

   #fig.add_annotation(annotations)

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

   if argv.setdefault('write_html',False):
   
    fig.write_html("plotly.html")


   if argv.setdefault('write_data',False):

       variables = {}
       for d in fig.data:
        variables[d['name']]=d

       plotly_tot = {'sample_0':{'variables':variables,'size':size}}
      
       save_data('output_new',plotly_tot)

       #plotly.io.write_json(fig,'ff.json')

   # plotly.io.to_image(fig,format='png')
   #fig.write_image("test.png",scale=5)
   if argv.setdefault('show',True):
    if 'google.colab' in sys.modules:
     fig.show(renderer='colab')
    else:
     fig.show(renderer='browser')


   return fig

