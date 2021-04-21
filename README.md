
<img src="docs/source/_static/openbte_logo.png" width="300">

[![license](https://img.shields.io/github/license/romanodev/openbte)](https://github.com/romanodev/OpenBTE/blob/master/LICENSE)
[![documentation](https://readthedocs.org/projects/pip/badge/?version=latest)](https://openbte.readthedocs.io/en/latest/)
[![downloads](https://img.shields.io/pypi/dm/openbte)](https://pypi.org/project/openbte/)
[![docker](https://img.shields.io/docker/pulls/romanodev/openbte)](https://hub.docker.com/r/romanodev/openbte)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18u1ieij2Wn6WEZFN2TmMteYHAJADMdSk?usp=sharing)
[![Python](https://img.shields.io/pypi/pyversions/openbte)](https://www.python.org/)

OpenBTE solves the Linearized Boltzmann Transport equation, with current focus on steady-state phonon transport.

```python

from openbte import Material,Geometry,Solver,Plot

Material(filename='rta_Si_300') #Retrieves data from the database

Geometry(lx = 50,ly = 50, lz=10,step=4,porosity=0.2,shape='circle') #Creates the structure

Solver() #Solves the BTE

Plot(model='vtu',repeat=[5,1,1]) #Creates a file which can be open with Paraview

```

![Thermal Flux](bte.png "Thermal Flux")












