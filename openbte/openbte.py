from openbte import Material,Solver,Geometry,Plot
import numpy as np
import yaml
import re,os

def main():

 if len(os.sys.argv) == 1:
   name = 'input.yaml'  
   with open(name) as f:
    data = yaml.safe_load(f)
 else:   
  a = os.sys.argv[1]
  if ':' in a:
   data = yaml.safe_load(a)

  else: 
   name = os.sys.argv[1]   
   with open(name) as f:
    data = yaml.safe_load(f)

 if 'Material' in data.keys():
  Material(**data['Material'])

 if 'Geometry' in data.keys():
  Geometry(**data['Geometry'])

 if 'Solver' in data.keys():
  Solver(**data['Solver'])

 if 'Plot' in data.keys():
  Plot(**data['Plot'])

 

