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
                     #showscale = True if len(units) > 0 else False,
                     showscale = True,
                     visible=visible
                    )

def add_data(nodes,elems,name,data,units,buttons,fig,nv):

      plotly_data=plotly_trisurf(nodes,elems,data,name,units,visible= True if len(buttons) == nv-1 else False)
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
        'y':0.97,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

   fig.update_layout(
    width=800,
    height=800,
    autosize=True,
    margin=dict(t=50, b=50, l=140, r=0),
    template='plotly_dark'
   )  

   updatemenus=[dict(direction='down',active=nv-1,x=-0.05,y=1.05,buttons=list(buttons),showactive=True)]
   fig.update_layout(updatemenus=updatemenus)


   #update axes---------------
   axis = dict(
          backgroundcolor="rgb(230, 230,230)",
          gridcolor="rgb(255, 255, 255)",
          zerolinecolor="rgb(255, 255, 255)",
          showbackground=False
         )

   #axis = dict(ticktext=[],tickvals= [],showbackground=False)

   # Update 3D scene options
   fig.update_scenes(
    aspectratio=dict(x=1, y=1, z=1),
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
   #show_in_window(fig)

   #fig.show()

   py.iplot(fig, filename='OpenBTE')


#def show_in_window(fig):
#    import sys, os
#    import plotly.offline
#    from PyQt5.QtCore import QUrl
#    from PyQt5.QtWebEngineWidgets import QWebEngineView
#    from PyQt5.QtWidgets import QApplication
#
#    plotly.offline.plot(fig, filename='name.html', auto_open=False)

#    app = QApplication(sys.argv)
#    web = QWebEngineView()
#    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "name.html"))
#    web.load(QUrl.fromLocalFile(file_path))
#    web.show()
#    sys.exit(app.exec_())




