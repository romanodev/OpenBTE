

def apply_film_in_plane(argv) :
     """Create Options for in-plane transport """

     argv['boundary'] = ['Periodic','Diffuse']
     argv['lx']       = argv['L']
     argv['ly']       = argv['L']
     argv['model']    = 'bulk'

def apply_film_cross_plane(argv) :
     """Create Options for cross-plane transport """

     argv['boundary'] = ['Isothermal','Periodic']
     argv['lx'] = argv['L']
     argv['ly'] = argv['L']
     argv['model']    = 'bulk'


def hexagonal_lattice(options_hexagonal_lattice)->'options_geometry':

  import numpy   as np

  lx = options_hexagonal_lattice['S']
  diameter  = options_hexagonal_lattice['D']
  r = diameter/2

  ly = lx*np.sqrt(3)
  A = np.pi*r*r
  phi = 2*A/lx/ly

  return {'lx':lx,'ly':ly,'porosity':phi,'step':options_hexagonal_lattice['step'],'shape':'circle','base':[[0.0,0.0],[-0.5,0.5]],'model':'lattice'}


def honeycomb_lattice(options_honeycomb_lattice)->'options_geometry':

  import numpy   as np

  lx = options_honeycomb_lattice['S']*np.sqrt(3)
  diameter  = options_honeycomb_lattice['D']
  r = diameter/2

  ly = lx*3/np.sqrt(3)
  A = np.pi*r*r
  phi = 4*A/lx/ly
  
  options_honeycomb_lattice.update({'lx':lx,'ly':ly,'porosity':phi,'shape':'circle','base':[[0.0,1.0/3.0],[0.0,-1.0/3.0],[-0.5,-1.0/6.0],[-0.5,1.0/6.0]],'model':'lattice'})

  return  options_honeycomb_lattice

def pore_grid(options_pore_grid)->'options_geometry':

    import numpy as np

    #Import grid---
    grid = np.array(options_pore_grid['grid'])
    N = int(np.sqrt(len(grid)))
    grid = grid.reshape((N,N)).nonzero()
    base = np.zeros((len(grid[0]),2))

    for n,(x,y) in enumerate(zip(grid[0],grid[1])):
      base[n] = [(y+1)/N-1/2/N-0.5,1-((x+1)/N-1/2/N-0.5)]

    phi = len(base)/N/N*options_pore_grid['porosity']

    l = options_pore_grid['L']
    
    if len(base) > 0:
     options_pore_grid.update({'model':'lattice','lx':l,'ly':l,'porosity':phi,'step':options_pore_grid['step']*l,'shape':options_pore_grid.setdefault('shape','square'),'base':base})
    else: #Bulk
     options_pore_grid.update({'model':'bulk','lx':l,'ly':l,'step':options_pore_grid['step']*l})

    return options_pore_grid


def square_lattice(options_square_lattice)->'options_geometry':

  import numpy   as np

  lx        = options_square_lattice['S']
  ly        = lx
  
  #if 'diameter' in options_square_lattice.keys():
  #   shape    = 'circle'
  porosity =  (options_square_lattice['D']/2)**2*np.pi/lx**2

  options_square_lattice['model'] = 'lattice'
  #else:    
  #   porosity  = options_square_lattice['porosity']
  #   shape     = options_square_lattice.setdefault('shape','circle')
  

  return {'lx':lx,'ly':ly,'porosity':porosity,'step':options_square_lattice['step'],'shape':'circle','base':[[0.0,0.0]],'model':'lattice'}

  #return options_square_lattice



