---
layout: default
title: Manual
permalink: /manual/
description: Here we show software architecture
---


<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    extensions: [
      "MathMenu.js",
      "MathZoom.js",
      "AssistiveMML.js",
      "a11y/accessibility-menu.js"
    ],
    jax: ["input/TeX", "output/CommonHTML"],
    TeX: {
      extensions: [
        "AMSmath.js",
        "AMSsymbols.js",
        "noErrors.js",
        "noUndefined.js",
      ]
    }
  });
</script>

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<p align="center">
<img  align="center" width="700" src="https://docs.google.com/drawings/d/e/2PACX-1vRqrihU3IHGVNRaNN7sc2r5CMphXVz6iT8jesHsX0blyj7GPh5KyiUiOFw8WMH9bHHNZYMzBTIgLPNo/pub?w=800&h=500">
</p>


# Material

The `Material` module creates the file `material.h5`, where bulk data is stored. It can be used in three ways:

__Database mode__. The module simply downloads a file from the online database. Example:

```python
Material(model='database',filename='C_0')
```
In this case `Material` simply retrieves the data from the [database](../database). The value of `filename` has to match one of the `Filename` fields of the database.

__Unlisted mode__. Similarly to the Database mode, the Unlisted mode retrives the material file from a `file_id`. This mode is useful when running OpenBTE via Colab with private input data. Example:
 
 ```python
Material(model='unlisted',filename='some_file_id')
```
    
__Local mode__. The file `material.h5` is created from scratch. Depending on the model being used, the input files have designed names. Example: 
    
```python
Material(model='rta',n_phi=92)
```
In this case, we are using the `rta` model and the designed input file is `rta.h5`. <br>
The current list of material models, options and designed filenames is [here](../material). <br>
The format of input files is desribed [here](../format) <br>
The list of interface to external solver is [here](../interface)

# Geometry

The module `Geometry` aids the creation of the structure and save it in `geometry.h5`. In OpenBTE, any structure is identified by a rectangular unit-cell with size `lx` by `ly`, a `porosity` (i.e. the volume fraction), a `base` - i.e. the position of the pores in the unit-cell and a `shape`. Below is a complete list of options.

|  Option  | Description    | Format |
|:-----------|:---------------|:-------------|
|   `lx`     | Size of the unit-cell along x | float [nm] |
|   `ly`     | Size of the unit-cell along y | float [nm] |
|   `porosity` | Volume fraction |   float |
|   `shape`   | shape of the pore | see below  |
|   `base`    | The positions of the pores | $$[[x_0,y_0],[x_1,y_1] ...$$ |
|   `direction`  | direction of the applied gradient| `x` (default) or `y`  |
|   `step`  | characteristic mesh size| float [nm]  |
 
Additional info/tips:

 - Commonly, a shape may cross one or even two boundaries of the unit-cell. The user needs to to include in the base the image pore, since OpenBTE will automatically added it.
 
 - The base has the range $$x\in [-0.5,0.5] $$ and $$y\in [-0.5,0.5] $$
 
 - When two pores overlap, including the cases where the overlap is due to periodicity, a bigger pore - the union of the two overlapping pores, is created.
 
 - The area of the pores is defined by the chosen porosity, the shape and number of pores in the base. For example, when the number of pores in the base doubles - with all other parameters being equal, the area of each pore halves.
 
 - The `step` keyword defined the characteristic size of the mesh. The number of elements will be roughly $$ lx*ly/\mathrm{step}^2 $$.  Typical calcylations have 400-10 K elements. 
 
 - Periodic boundary conditions are applied to the the portion of the boundary orthogonal to the applied gradient. 
 
 - Diffuse scattering boundary conditions are applied to the the portion of the boundary parallel to the applied gradient.
 
 - Diffuse scattering boundary conditions are applied along the walls of the pores.
 
 
 An example of `Geometry` instance is 
 
 ```python
 Geometry(porosity=0.3,shape'circle',lx=100,ly=100,step=5)
 ```
 
# Solver
# Plot
# Examples
https://nbviewer.jupyter.org/github/romanodev/OpenBTE/blob/master/openbte/Tutorial.ipynb  
The examples are provided via a [notebook](https://nbviewer.jupyter.org/github/romanodev/OpenBTE/blob/master/openbte/Tutorial.ipynb ) or [interactive simulations](https://colab.research.google.com/drive/1eAfX3PgyO7TyGWPee8HRx5ZbQ7tZfLDr) (Google Colab).

# Notes

### <a name="1"></a> 1. <pre> `2DSym` is a system that is infinite in the third dimension. Below are the cases are shown.

<p align="left">
<img  align="left" width="700" src="https://docs.google.com/drawings/d/e/2PACX-1vRUa6nwKHA_kCBaofjivbwPbmgweDab5xXCKdEesLTZF622a020f0xm7rlufdCufwhquPBLLTTFzrEO/pub?w=885&h=138">
</p>

</pre>

 



