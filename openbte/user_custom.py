import numpy as np
import subprocess,os


class UserStructure(object):

    def __init__(self,**argv):

      self.argv = argv



    def generate_structure(self):
  
       if 'structure' in argv.keys():  
        
         return self.generator(**self.argv)   


    def generate_pattern(self,mesh):
        if self.pattern: 
            self.argv['mesh'] = mesh
            return self.argv['pattern'](**self.argv)
        else:
            return np.zeros(len(self.elems))

    def solve(self,**argv):
  
       self.argv.update(argv)  

       return self.argv['solve'](**self.argv)

    def plot(self,mat,solver,mesh):
   
       return self.argv['plot'](mat,solver,mesh)
