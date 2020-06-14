from openbte import Material,Solver,Geometry,Plot
import numpy as np
import yaml
import re,os

def main():
 

 if len(os.sys.argv) == 2:
      name = os.sys.argv[1]   
 else:  
      name = 'input.yaml'  

 #from https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number

 loader = yaml.SafeLoader
 loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


 with open(name) as f:
    data = yaml.load(f, Loader=loader)

 if 'Material' in data.keys():
  Material(**data['Material'])

 if 'Geometry' in data.keys():
  Geometry(**data['Geometry'])

 if 'Solver' in data.keys():
  Solver(**data['Solver'])

 if 'Plot' in data.keys():
  Plot(**data['Plot'])

 

