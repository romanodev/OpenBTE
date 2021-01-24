import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import plotly.io as pio
import dash
import dash_core_components as dcc
import dash_html_components as html
import webbrowser
from threading import Timer
import dash_bootstrap_components as dbc 
import deepdish
import flask
from dash.dependencies import Input, Output
import gzip
import pickle
import plotly.express as px
import pandas as pd
from google_drive_downloader import GoogleDriveDownloader as gdd
import json
import os


def load_data(fh):

 if os.path.isfile(fh + '.npz'):
      with gzip.open(fh + '.npz', 'rb') as f:

          return pickle.load(f)

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')

def get_layout(size = []):
 


 font=dict(
        family="Courier New, monospace",
        size=12,
        #color="#7f7f7f"
        color="gray"
       )
 


 if len(size) == 0:
   x = 1;y=1;z=1
   axis = dict(
          backgroundcolor="rgb(230, 230,230)",
          showline=False,
          zeroline=False,
          visible=False,
          showgrid=False,
          ticks='',
          showticklabels=False,
          gridcolor="rgb(255, 255, 255)",
          zerolinecolor="rgb(255, 255, 255)",
          showbackground=False
         )



 #camera
 else:  
  sl = True
  dim = 2   if size[2] == 0 else 3
  x = 1;y=size[1]/size[0]
  z= 1 if dim == 2 else size[2]/size[0]
  axis = dict(
          backgroundcolor="rgb(230, 230,230)",
          gridcolor="rgb(255, 255, 255)",
          zerolinecolor="rgb(255, 255, 255)",
          showbackground=False
         )

 # Update 3D scene options
 scene = dict(
    aspectratio=dict(x= x, \
                     y= y, \
                     z= z),
    xaxis=dict(axis),
    yaxis=dict(axis),
    zaxis=dict(axis))


 #camera
 camera = dict(
    up=dict(x=0, y=1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=0, z=1.5)
   )


 layout = dict(font=font,\
               #title=title,\
               autosize=True,\
               #width='50%',\
               height=500,\
               #showlegend=False,\
               scene=scene,\
               uirevision='static',\
               scene_camera = camera,\
               xaxis_showgrid=False,\
               yaxis_showgrid=False,\
               margin =dict(t=50, b=20, l=20, r=20),\
               paper_bgcolor='rgba(0,0,0,0)',\
               plot_bgcolor='rgba(0,0,0,0)')

 return layout


def get_dd_layout():

 style={}

 return style


def App(**argv):

 #load data--
 data_tot     = argv['bundle'] if 'bundle' in argv.keys() else load_data('bundle')



 server = flask.Flask(__name__) # define flask app.server


 d_sample = html.Div([
    dcc.Dropdown(
        id='samples',
        value=0,
        style = get_dd_layout(),
        clearable=False,
        searchable=False
    ),
 ])


 d_variable = html.Div([
    dcc.Dropdown(
        id='variables',
        value=0,
        style= get_dd_layout(),
        clearable=False,
        searchable=False
    ),
 ])


 fig_dcc = dcc.Graph(id='master',figure=go.Figure(layout=get_layout()),style={'left':'15%','width':'70%','hight':'70%','bottom':'15%','position':'absolute','background-color':'black'})

 external_stylesheets=[dbc.themes.CYBORG]

 app = dash.Dash(__name__, external_stylesheets=external_stylesheets,server=server)

 dd_sample = html.Div( children = [d_sample],style = {'left':'1%','height':'10%','width':'25%','bottom':'88%','position':'absolute'})
 dd_variable = html.Div( children = [d_variable],style = {'left':'28%','height':'10%','width':'25%','bottom':'88%','position':'absolute'})

 hidden = html.Div(id='intermediate-value', style={'display': 'none'})

 url = dcc.Location(id='url', refresh=True)


 img = html.Div(html.Img(src=app.get_asset_url('logo.png'), style={'width':'18%','left':'80%','bottom':'88%','position':'absolute'}))

 div1 = html.Div(style = {'left':'25%','height':'70%','width':'50%','bottom':'15%',"border":"2px white solid",'position':'absolute'},children=[fig_dcc,dd_sample,dd_variable,hidden,url]);

 app.layout = html.Div(children = [div1,img],style = {'height': '100vh','width': '100%',"border":"2px white solid"})

 #---------------------------------------------------------------------


 @app.callback(
    Output('master','figure'),
    Input('samples','value'), Input('variables','value'),Input('intermediate-value','children'),prevent_initial_call=True
 )

 def plot(sample,variable,data_tot):


       key_sample = list(data_tot.keys())[int(sample)]
       data = data_tot[key_sample]
       name = list(data['variables'].keys())[int(variable)]
       variable  = data['variables'][name]
       elems     = np.array(data['elems'])
       nodes     = np.array(data['nodes'])
       node_data      = variable['data']

       bb = str(round(data['bte'],2))+' W/m/K' if 'bte' in data.keys() else '--'
       text  = 'Bulk: ' + str(round(data['bulk'],2)) +' W/m/K<br>Fourier: '\
                    +       str(round(data['fourier'],2)) + ' W/m/K<br>BTE:' \
                    +       bb

       z = np.zeros(len(nodes))  if np.shape(nodes)[1] == 2 else nodes[:,2]

       plotly_data = go.Mesh3d(x=nodes[:,0],
                         y=nodes[:,1],
                         z=z,
                         colorscale='Viridis',
                         colorbar_title = '[' + variable['units'] + ']' if len(variable['units']) > 0 else "",
                         intensity = node_data,
                         colorbar=dict(len=0.5),
                         intensitymode='vertex',
                         i=elems[:,0],
                         j=elems[:,1],
                         k=elems[:,2],
                         name=name,
                         showscale = True,
                         visible=True)

       fig = go.Figure(data=plotly_data,layout=get_layout(data_tot['size']))

       fig.add_annotation(
            x=1,
            y=0.05,
            xref='paper',
            showarrow=False,
            yref='paper',
            text=text,align='left')

       return  fig



 @app.callback(
    [Output('intermediate-value', 'children'),Output('samples','options'),Output('variables','options')],
    [Input('url','search')],prevent_initial_call=True
 )
 def update_output_div(pathname):
  
    input_value = pathname[1:]

    #load data--
    #data = load_data('bundle')

    variables = data_tot[list(data_tot.keys())[0]]['variables'].keys()
     
    #variables = [key  for key,value in  data[list(data.keys())[0]]['solver']['variables'].items()]

    #Get size-----------------
    size = 3*[0]

    for sample,v in data_tot.items():

      nodes = v['nodes']
      
      size  = [max([size[i],(max(nodes[:,i]) - min(nodes[:,i]))])  for i in range(3)]


    options_samples=[{'label': s, 'value': str(n)} for n,s in enumerate(data_tot)]
    options_variables=[{'label': s, 'value': str(n)} for n,s in enumerate(variables)]
 

    data_tot['size'] = size

    return data_tot,options_samples,options_variables


#---------------------------------------------------------------------

 app.run_server(host='localhost',port=8000)


