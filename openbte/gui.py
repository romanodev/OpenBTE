from __future__ import print_function
get_ipython().magic(u'matplotlib inline')
from openbte.geometry import *
from openbte.solver import *
from openbte.material import *
from openbte.plot import *
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display




def create_geometry(porosity,shape,angle,lattice):

    if lattice == 'squared':
     type = 'porous/square_lattice'
     step = 0.1
    if lattice == 'staggered':
     type = 'porous/staggered_lattice'
     step = 0.2
    if lattice == 'honeycomb':
      type = 'porous/hexagonal_lattice'
      step = 0.1 *np.sqrt(sqrt(3.0))



    Geometry(type = type,shape=shape,
                                  lx = 1.0, ly = 1.0,\
                                  step=step,\
                                  angle=angle,\
                                  porosity=porosity,
                                  inclusion = True,
                                  mesh=True,plot=True);


def create_material(kappa_matrix,kappa_inclusion):
    Material(region='Matrix',kappa = kappa_matrix,filename='material_a');
    Material(region='Inclusion',kappa = kappa_inclusion,filename='material_b');

def run():
 Solver(max_bte_iter = 0,verbose=False);
 Plot(variable='map/fourier_flux',direction='magnitude',show=True,write=False,streamlines=False,repeat_x=1,repeat_y =1,plot_interfaces=True)



r=widgets.ToggleButton(
    value=False,
    description='Click me',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Description',
    icon='check'
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


d = widgets.RadioButtons(
    options=['circle', 'square', 'triangle'],
    value='circle',
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


km = widgets.BoundedFloatText(
    value=150.0,
    min=1,
    max=1000,
    step=0.1,
    description='Matrix [W/K/m]',
    disabled=False,
    layout = {'width': '300px'},\
    style = {'description_width': '150px'})

ki = widgets.BoundedFloatText(
    value=50.0,
    min=0,
    max=1000,
    step=0.1,
    description='Inclusion [W/K/m]',
    disabled=False,
    layout = {'width': '300px'},\
    style = {'description_width': '150px'})



interact(create_geometry, porosity=w, shape=d, angle=a,lattice=l);
interact(create_material, kappa_matrix=km,kappa_inclusion=ki);
button = widgets.Button(description="Run")
