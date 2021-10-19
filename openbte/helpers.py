

def hexagonal_lattice(options_hexagonal_lattice)->'options_geometry':

  import numpy   as np

  lx = options_hexagonal_lattice['pitch']
  neck  = options_hexagonal_lattice['neck']
  ly = lx*np.sqrt(3)
  r = (lx-neck)/2
  A = np.pi*r*r
  phi = 2*A/lx/ly

  return {'lx':lx,'ly':ly,'porosity':phi,'step':options_hexagonal_lattice['step'],\
          'lz':options_hexagonal_lattice['lz'],'shape':'circle','base':[[0.0,0.0],[-0.5,0.5]],'direction':options_hexagonal_lattice.setdefault('direction','x')}



