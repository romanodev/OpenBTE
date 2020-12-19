import numpy as np
import subprocess,os


class UserStructure(object):

    def __init__(self,**argv):

      self.argv = argv
      

    def generate_structure(self):
  
       if 'structure' in self.argv.keys():  
         return self.argv['structure'](**self.argv) 
       else  :
         return self.argv['lx'],self.argv['ly'],self.argv.setdefault('lz',0)

    def generate_pattern(self,mesh):

        if 'pattern' in self.argv.keys():
            self.argv['mesh'] = mesh
            return self.argv['pattern'](**self.argv)
        else:
            return np.zeros(len(mesh.elems))

    def solve(self,**argv):
  
       self.argv.update(argv)  

       return self.argv['solve'](**self.argv)

    def plot(self,mat,solver,mesh):

       self.argv.update({'mat':mat,'mesh':mesh,'solver':solver})

       if isinstance(self.argv['plot'],list):
           for f in  self.argv['plot']:
             f(**self.argv)
       else:
            self.argv['plot'](**self.argv)

    def material(self):
   
       return self.argv['material'](**self.argv)
