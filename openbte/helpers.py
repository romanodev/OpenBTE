

def hexagonal_lattice(options_hexagonal_lattice)->'options_geometry':

  import numpy   as np

  lx = options_hexagonal_lattice['periodicity']
  if 'neck' in  options_hexagonal_lattice.keys():
   neck  = options_hexagonal_lattice['neck']
   r = (lx-neck)/2
  else: 
   diameter  = options_hexagonal_lattice['diameter']
   r = diameter/2


  ly = lx*np.sqrt(3)
  A = np.pi*r*r
  phi = 2*A/lx/ly

  return {'lx':lx,'ly':ly,'porosity':phi,'step':options_hexagonal_lattice['step'],\
          'lz':options_hexagonal_lattice['lz'],'shape':'circle','base':[[0.0,0.0],[-0.5,0.5]],'direction':options_hexagonal_lattice.setdefault('direction','x')}


def honeycomb_lattice(options_honeycomb_lattice)->'options_geometry':

  import numpy   as np

  lx = options_honeycomb_lattice['periodicity']
  diameter  = options_honeycomb_lattice['diameter']

  r = diameter/2
  ly = lx*3/np.sqrt(3)
  A = np.pi*r*r
  phi = 4*A/lx/ly

  return {'lx':lx,'ly':ly,'porosity':phi,'step':options_honeycomb_lattice['step'],\
          'lz':options_honeycomb_lattice['lz'],'shape':'circle','base':[[0.0,1.0/3.0],[0.0,-1.0/3.0],[-0.5,-1.0/6.0],[-0.5,1.0/6.0]],'direction':options_honeycomb_lattice.setdefault('direction','x')}


def square_lattice(options_square_lattice)->'options_geometry':

  import numpy   as np

  lx        = options_square_lattice['periodicity']
  ly        = lx
  if 'diameter' in options_square_lattice.keys():
     shape    = 'circle'
     porosity =  (options_square_lattice['diameter']/2)**2*np.pi
  else:    
     porosity  = options_square_lattice['porosity']
     shape     = options_square_lattice.setdefault('shape','circle')
  

  return {'lx':lx,'ly':ly,'porosity':porosity,'step':options_square_lattice['step'],\
          'lz':options_square_lattice.setdefault('lz',0),'shape':shape,'base':[[0.0,0.0]],'direction':options_square_lattice.setdefault('direction','x')}



