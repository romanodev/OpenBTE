

import os
import multiprocessing
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from openbte.geometry        import Geometry
from openbte.geometry        import circle       
from openbte.geometry        import rectangle    
from openbte.geometry        import triangle
from openbte.bte             import BTE_RTA
from openbte.fourier         import Fourier
from openbte.utils           import load_rta 
from openbte.mesh            import get_mesh
from openbte.material        import Gray2D,Gray3D,Gray3DEqui,RTA2DSym,RTA2DMode,RTA3D


#Write the version#
import pkg_resources  
v = pkg_resources.require("OpenBTE")[0].version   

print('OpenBTE')
print('Version: ',v)
