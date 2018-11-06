from __future__ import print_function
get_ipython().magic(u'matplotlib nbagg')
from openbte.material import *
from openbte.plot import *
from ipywidgets import interact, interactive, fixed, interact_manual,HBox,VBox
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Button, GridBox, Layout, ButtonStyle
from IPython.display import Latex
import numpy as np
#from openbte.solver_new import *
#from openbte.geometry import *


t = widgets.Text(value = 'Thermal Conductivity [W/m/K]: ')

sel = widgets.Select(
    options=['Geometry',' '],
    value='Geometry',
    description='Variable',
    disabled=False
)

w = widgets.FloatSlider(
    value=0.2,
    min=0.01,
    max=0.3,
    step=0.01,
    description='Porosity:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.3f')
w.style.handle_color = 'lightblue'


ki = widgets.BoundedFloatText(
    value=100.0,
    min=0,
    max=1000,
    step=0.1,
    description='TC Inclusion [W/K/m]',
    disabled=True,
    layout = {'width': '300px'},\
    style = {'description_width': '150px'})

mi = widgets.BoundedFloatText(
    value=0.1,
    min=1e-3,
    max=1e3,
    step=0.001,
    description='Kn Inclusion',
    disabled=True,
    layout = {'width': '300px'},\
    style = {'description_width': '150px'})




def create_geometry(porosity,shape,angle,lattice,dx,dy,inclusion,direction,xp,yp):

    if lattice == 'squared':
     type = 'porous/square_lattice'
     step = 0.1
    if lattice == 'staggered':
     type = 'porous/staggered_lattice'
     step = 0.2
    if lattice == 'honeycomb':
      type = 'porous/hexagonal_lattice'
      step = 0.1 *np.sqrt(sqrt(3.0))

    geo = Geometry(type = type,shape=shape,
                                  lx = 1.0, ly = 1.0,\
                                  step=step/1.0,\
                                  angle=angle,\
                                  dx = dx,\
                                  dy = dy,\
                                  porosity=porosity,
                                  direction=direction,
                                  inclusion = inclusion,
                                  Periodic=[xp,yp,False],
                                  mesh=True,plot=False);


    sel.options=['Geometry',' ']
    sel.value = ' '
    sel.value = 'Geometry'
    t.value = 'Thermal Conductivity [W/m/K]: '

    
    mi.disabled = not inclusion
    ki.disabled = not inclusion
    
        


def create_material(kappa_matrix,kappa_inclusion,mfp_matrix,mfp_inclusion):
    Material(region='Matrix',kappa = kappa_matrix,filename='material_a',mfps=[mfp_matrix*1e-9]);
    Material(region='Inclusion',kappa = kappa_inclusion,filename='material_b',mfps=[mfp_inclusion*1e-9]);
    w.value = w.value


def plot_data(variable):

 if not variable in ['Geometry',' '] :
  if variable == 'Flux (Magnitude) [Fourier]':
      var = 'fourier_flux'
      direction = 'magnitude'
  elif variable == 'Flux (X) [Fourier]':
      var = 'fourier_flux'
      direction = 'x'
  elif variable == 'Flux (Y) [Fourier]':
      var = 'fourier_flux'
      direction = 'y'
  elif variable == 'Temperature [Fourier]':
      var = 'fourier_temperature'
      direction = None

  if variable == 'Flux (Magnitude) [BTE]':
    var = 'bte_flux'
    direction = 'magnitude'
  elif variable == 'Flux [BTE]':
    var = 'bte_flux'
    direction = 'x'
  elif variable == 'Flux (Y) [BTE]':
    var = 'bte_flux'
    direction = 'y'
  elif variable == 'Temperature [BTE]':
    var = 'bte_temperature'
    direction = None



  Plot(variable='map/' + var,direction=direction,show=True,write=False,streamlines=False,repeat_x=1,repeat_y =1,plot_interfaces=True)
  #Plot(variable='line/'+ var,direction=direction,show=True,write=False,streamlines=False,repeat_x=1,repeat_y =1,plot_interfaces=True)


 if variable == 'Geometry':
  Geometry(type='load',filename = 'geometry',plot=True)

def run_fourier(b):
 sol = Solver(max_bte_iter = 0,verbose=False);

 kappa = sol.state['kappa_fourier']
 #if b.value==0:
 # display(t)
 t.value = 'Thermal Conductivity [W/m/K]: '.ljust(20) +  '{:8.2f}'.format(kappa)
 b.value +=1
 sel.options=['Geometry',display(Latex(r'''Flux (X) [Fourier]''')), 'Flux (Y) [Fourier]', 'Flux (Magnitude) [Fourier]','Temperature [Fourier]']

 sel.value = 'Flux (Magnitude) [Fourier]'

def run_bte(b):
  sol = Solver(max_bte_iter = 100,verbose=True);

  kappa = sol.state['kappa_bte']
  #if b.value==0:
  # display(t)
  t.value = 'Thermal Conductivity [W/m/K]: '.ljust(20) +  '{:8.2f}'.format(kappa)
  b.value +=1
  sel.options=['Geometry','Flux [X] [BTE]', 'Flux (Y) [BTE]', 'Flux (Magnitude) [BTE]','Temperature [BTE]']

  sel.value = 'Flux (Magnitude) [BTE]'



w = widgets.FloatSlider(
    value=0.2,
    min=0.01,
    max=0.3,
    step=0.01,
    description='Porosity:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.3f')
w.style.handle_color = 'lightblue'


d = widgets.RadioButtons(
    options=['circle', 'square', 'triangle'],
    value='square',
    description='shape:',
    disabled=False
)

l = widgets.RadioButtons(
    options=['squared', 'staggered', 'honeycomb'],
    value='squared',
    description='Pore lattice:',
    disabled=False
)

a = widgets.FloatSlider(
    value=0.0,
    min=0.0,
    max=360,
    step=10,
    description='Angle:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.0f')
a.style.handle_color = 'lightblue'

dx = widgets.FloatSlider(
    value=0.0,
    min=-0.5,
    max=0.5,
    step=0.1,
    description='dx',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f')
dx.style.handle_color = 'red'

dy = widgets.FloatSlider(
    value=0.0,
    min=-0.5,
    max=0.5,
    step=0.1,
    description='dy',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f')
dy.style.handle_color = 'red'


inc = widgets.Checkbox(
    value=True,
    description='Inclusion',
    disabled=False
)


xp = widgets.Checkbox(
    value=True,
    description='Periodic along x',
    disabled=False
)

yp = widgets.Checkbox(
    value=True,
    description='Periodic along y',
    disabled=False
)


gra = widgets.RadioButtons(
    options=['x', 'y'],
    value='x',
    description='Direction of the applied temperature',
    disabled=False
)

km = widgets.BoundedFloatText(
    value=150.0,
    min=1,
    max=1000,
    step=0.1,
    description='TC Matrix [W/K/m]',
    disabled=False,
    layout = {'width': '300px'},\
    style = {'description_width': '150px'})



mm = widgets.BoundedFloatText(
    value=0.1,
    min=1e-3,
    max=1e3,
    step=0.001,
    description='Kn Matrix',
    disabled=False,
    layout = {'width': '300px'},\
    style = {'description_width': '150px'})



#plot geometry-------
test = interact(create_geometry, porosity=w, shape=d, angle=a,lattice=l,dx=dx,dy=dy,inclusion=inc,direction=gra,xp=xp,yp=yp);
v3 = interactive(create_material, kappa_matrix=km,kappa_inclusion=ki,mfp_matrix = mm,mfp_inclusion=mi);



#display(test.children[:])
#v1 = VBox(test.children[0:2])
#v2 = VBox(test.children[2:4])
#display(HBox([v1,v2,v3]))
display(v3)
#--------------------


v = interactive(plot_data,variable=sel);
display(v)


#plot output-------
button_fourier = widgets.Button(description="Fourier");
button_fourier.value = 0
button_fourier.on_click(run_fourier)
button_bte= widgets.Button(description="BTE");
button_bte.on_click(run_bte)
button_bte.value = 0
display(HBox([button_fourier,button_bte,t]))
#-------------------





#
