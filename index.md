---
layout: default
---


```python
from openbte import Material,Geometry,Solver,Plot

Material(filename='Si_300') #From the database

Geometry(model='porous/square_lattice',l = 10, porosity=0.05,base=[[0.6,0.4],\
                                              			   [0.4,0.6],\
                                            			   [0.4,0.4],\
                                             			   [0.6,0.6]])
Solver()
Plot(variable=['flux','temperature','flux_fourier'])
```

![]({{ site.url }}/OpenBTE/assets/flux.png)
![]({{ site.url }}/OpenBTE/assets/temp.png)
![]({{ site.url }}/OpenBTE/assets/ff.png)

To become a beta tester for new functionalities, please fill this [form](https://forms.gle/cJBE4Jjqkrh8djJX8).
