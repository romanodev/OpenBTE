import numpy as np
import subprocess,os


class UserStructure(object):

    def __init__(self,generator,**argv):

      self.generator = generator   
      self.argv = argv

    def generate_structure(self):
   
       return self.generator()   


