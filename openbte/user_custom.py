import numpy as np
import subprocess,os


class UserStructure(object):

    def __init__(self,generator,**argv):

      self.generator = generator   
      self.argv = argv
      if 'pattern' in argv.keys():
         self.pattern = True 

    def generate_structure(self):
   
       return self.generator(**self.argv)   


    def generate_pattern(self,mesh):
        if self.pattern: 
            self.argv['mesh'] = mesh
            return self.argv['pattern'](**self.argv)
        else:
            return np.zeros(len(self.elems))

    def solve(self,**argv):
   
       argv.update(self.argv)  
       return self.argv['solve'](**argv)

    def plot(self,mat,solver,mesh):
   
       return self.argv['plot'](mat,solver,mesh)
