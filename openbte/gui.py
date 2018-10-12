from __future__ import print_function
get_ipython().magic(u'matplotlib nbagg')
from openbte.geometry import *
from openbte.solver import *
from openbte.material import *
from openbte.plot import *
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display


t = widgets.Text()
#display(t)

sel = widgets.Select(
    options=['Geometry','Flux (X)', 'Flux (Y)', 'Flux (Magnitude)'],
    value='Geometry',
    # rows=10,
    description='Variable',
    disabled=False
)


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

    geo = Geometry(type = type,shape=shape,
                                  lx = 1.0, ly = 1.0,\
                                  step=step,\
                                  angle=angle,\
                                  porosity=porosity,
                                  inclusion = True,
                                  mesh=True,plot=False);
    #geo.plot_polygons()


def create_material(kappa_matrix,kappa_inclusion):
    Material(region='Matrix',kappa = kappa_matrix,filename='material_a');
    Material(region='Inclusion',kappa = kappa_inclusion,filename='material_b');

def plot_data(variable):

 print(variable)

 if not variable=='Geometry':
  if variable == 'Flux (Magnitude)':
      var = 'map/fourier_flux'
      direction = 'magnitude'
  elif variable == 'Flux (X)':
      var = 'map/fourier_flux'
      direction = 'x'
  elif variable == 'Flux (Y)':
      var = 'map/fourier_flux'
      direction = 'y'

  #gcf().gca().clear()
  Plot(variable=var,direction=direction,show=True,write=False,streamlines=False,repeat_x=1,repeat_y =1,plot_interfaces=True)
  print('g')
  show()
 else:
  #if not fignum_exists(1):
  Geometry(type='load',filename = 'geometry',plot=True)
  #geo.plot_polygons()
   #create_geometry(w.value,d.value,a.value,l.value)


def run(b):
 sol = Solver(max_bte_iter = 0,verbose=False);

 kappa = sol.state['kappa_fourier']
 if b.value==0:
  display(t)
 t.value = 'Thermal Conductivity: '.ljust(20) +  '{:8.2f}'.format(kappa)+ ' W/m/K'
 b.value +=1
 #plot_data('Flux (Magnitude)')
 sel.value = 'Flux (Magnitude)'
 #display(sel)



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
interact(plot_data,variable=sel);

button = widgets.Button(description="Run");
button.value = 0

#out = widgets.Output()

#def on_button_clicked(b):
    #button.description = 'Run'
    #with out:
    # kappa =  run()
    # print('Thermal Conductivity: '.ljust(20) +  '{:8.2f}'.format(3.333)+ ' W/m/K')


v = button.on_click(run)
#widgets.VBox([button, out])
display(button);

#display(t)
#def f(kappa):#
    #print('Thermal Conductivity: '.ljust(20) +  '{:8.2f}'.format(3.333)+ ' W/m/K')

#out = widgets.interactive_output(f, {'kappa': 0.0})

#widgets.HBox([widgets.VBox([a, b, c]), out])
