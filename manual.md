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

The current list of material models, options and designed filenames is below

## Material Models

|  `model`   | Description    |  File  |   Options     |  
|:-----------|:---------------|:-------|:--------------|
|   `full`     | Full scattering operator | `full.h5` | None |
|   `rta`     | Relaxation time approximation <br> (RTA) | `rta.h5` | `n_phi`: # of polar angles (`48`) <br> `n_theta`: # of azimuthal angles (`24`)  <br> `n_mfp`: # of mean free paths (`50`)|
|   `rta2D`     | RTA for materials <br> with infinite thickness and <br> two-dimensional (2D) materials| `rta.h5` | `n_phi`: # of polar angles (`48`)  <br> `n_theta`: # of azimuthal angles (`24`)  <br> `n_mfp`: # of mean free paths (`50`)|
|   `mfp`     | mean-free-path BTE (MFP-BTE)| `mfp.h5` | `n_phi`: # of polar angles (`48`)  <br> `n_theta`: # of azimuthal angles (`24`) <br> `n_mfp`: # of mean free paths (`50`)|
|   `mfp2DSim`     | MFP-BTE for materials with <br> infinite thickness | `mfp.h5` | `n_phi`: # of polar angles (`48`) <br> `n_mfp`: # of mean free paths (`50`)|
|   `mfp2D`     | MFP-BTE for 2D materials| `mfp.h5` | `n_phi`: # of polar angles (`48`)   <br> `n_theta`: # of azimuthal angles (`24`)<br> `n_mfp`: # of mean free paths (`50`)|
|   `gray`     | single MFP| None | `n_phi`: # of polar angles (`48`)   <br> `n_theta`: # of azimuthal angles (`24`) <br> `mfp`: mean free path (m) <br> `kappa`: bulk thermal conductivity (Wm$$^{-1}$$k$$^{-1}$$)|
|   `gray2Dsim`     | single MFP in materials with <br> infinite thickness| None | `n_phi`: # of polar angles (`48`) <br> `mfp`: mean free path (m) <br> `kappa`: bulk thermal conductivity (Wm$$^{-1}$$k$$^{-1}$$)|
|   `gray2D`     | single MFP in 2D materials| None | `n_phi`: # of polar angles (`48`) <br> `mfp`: mean free path (m) <br> `kappa`: bulk thermal conductivity (Wm$$^{-1}$$k$$^{-1}$$)|
|   `fourier`     | Fourier's law| None | `kappa`: bulk thermal conductivity (Wm$$^{-1}$$k$$^{-1}$$)|

## Input files format

Below we report the format for the files `hdf5` files `rta.h5`, `mfp.h5` and `full.h5`. All values are intended to be `numpy` arrays. A good Pythonic option for creating `hdf5` files is the `deepdish` [package](https://deepdish.readthedocs.io/en/latest/io.html)

Here is an example:

```python
import deepdish as dd

data = {'field1':np.ones(2),'field2':np.ones(3)}
dd.io.save('file.h5',data)
```


### `mfp.h5`

| Field     | Size     | Units | Description | 
|:-------------|:------------------|:---------|:---------|
| `mfp`    | $$N$$      | m  |   MFP ($$N$$: number of MFPs)|
| `K`    | $$N$$      | Wm$$^{-1}$$K$$^{-1}$$  | discrete MFP distribution |   
| `kappa`    | $$3 \times 3$$      | Wm$$^{-1}$$K$$^{-1}$$  | bulk thermal conductivity tensor |    

### `full.h5`

| Field     | Size     | Units |  Description |
|:-------------|:------------------|:---------|:---------|
| `f`    | $$N$$      | s$$^{-1}$$  | frequency ($$N = N_bN_q$$, where $$N_b:$$ number of branches, $$N_q$$: number of wave vectors) |
| `alpha`    | $$1$$      | m$$^{-3}$$ | $$VN_q$$ ($$V$$: volume of the unit cell) |
| `A`     | $$N \times N $$ | s$$^{-1}$$  | scattering operator as Eq. 7 in [this](https://arxiv.org/pdf/2002.08940.pdf) |
| `v` | $$ N \times 3 $$| ms$$^{-1}$$ | group velocity |
| `kappa`    | $$3 \times 3$$      | Wm$$^{-1}$$K$$^{-1}$$  | bulk thermal conductivity tensor | 


### `rta.h5`

| Field     | Size     | Units |  Description |
|:-------------|:------------------|:---------|:---------|
| `f`    | $$N$$      | s$$^{-1}$$  | frequency ($$N = N_bN_q$$, where $$N_b:$$ number of branches, $$N_q$$: number of wave vectors) |
| `alpha`    | $$1$$      | m$$^{-3}$$ | $$VN_q$$ ($$V$$: volume of the unit cell) |
| `tauinv`     | $$NN $$ | s$$^{-1}$$  | scattering rates as Eq. 7 in [this](https://arxiv.org/pdf/2002.08940.pdf) |
| `v` | $$ N \times 3 $$| ms$$^{-1}$$ | group velocity |
| `kappa`    | $$3 \times 3$$      | Wm$$^{-1}$$K$$^{-1}$$  | bulk thermal conductivity tensor |  


## External Solvers

The list of interface to external solves is below

### AlmaBTE

This interface converts the output of [AlmaBTE](http://www.almabte.eu/)'c command `phononinfo` to OpenBTE format and write the file `rta.f5`. Let's assume you have the file `Si_4_4_4_300K.phononinfo` in your current directory. Then simply type

```python
AlmaBTE2OpenBTE Si_4_4_4_300K.phononinfo
```
The created file, `rta.h5`, can be used by a `Material`

# Geometry

The module `Geometry` currently features two models, `Lattice` and `Custom`.

##Lattice
The structure is identified by a rectangular unit-cell with size `lx` by `ly`, a `porosity` (i.e. the volume fraction), a `base` - i.e. the position of the pores in the unit-cell and a `shape`. Below is a complete list of options.

|  Option  | Description    | Format |
|:-----------|:---------------|:-------------|
|   `lx`     | Size of the unit-cell along x | float [nm] |
|   `ly`     | Size of the unit-cell along y | float [nm] |
|   `porosity` | Volume fraction |   float |
|   `shape`   | shape of the pore | see below  |
|   `base`    | The positions of the pores | $$[[x_0,y_0],[x_1,y_1] ...$$ |

 
Additional info/tips:

 - Commonly, a shape may cross one or even two boundaries of the unit-cell. The user needs to to include in the base the image pore, since OpenBTE will automatically added it.
 
 - The base has the range $$x\in [-0.5,0.5] $$ and $$y\in [-0.5,0.5] $$
 
 - When two pores overlap, including the cases where the overlap is due to periodicity, a bigger pore - the union of the two overlapping pores, is created.
 
 - The area of the pores is defined by the chosen porosity, the shape and number of pores in the base. For example, when the number of pores in the base doubles - with all other parameters being equal, the area of each pore halves.
 
 - The `step` keyword defined the characteristic size of the mesh. The number of elements will be roughly $$ lx*ly/\mathrm{step}^2 $$.  Typical calcylations have 400-10 K elements. 
 
 - Periodic boundary conditions are applied to the the portion of the boundary orthogonal to the applied gradient. 
 
 - Diffuse scattering boundary conditions are applied to the the portion of the boundary parallel to the applied gradient.
 
 - Diffuse scattering boundary conditions are applied along the walls of the pores.
 
###Shape

A shape can be either a predefined one - including `square`, `triangle` and `circle` - or user defined. In the latter case, the option `shape=custom` must be used and two additional keywords are to be used, i.e. `shape_function` and `shape_options`, where the options can be used for additional parameters to be used to shape function. Note that the shape coordinates are normalized to $$(-0.5,0.5)$$ both in $$x$$ and $$y$$ coordinates. Lastly, the shape function must at least take the option `area` in input, which is internally calculates.


 
##Commong options

|  Option  | Description    | Format |
|:-----------|:---------------|:-------------|
|   `direction`  | direction of the applied gradient| `x` (default) or `y`  |
|   `step`  | characteristic mesh size| float [nm]  |
 
 An example of `Geometry` instance is 
 
 
 ```
 def shape(options):
   area = options['area']
   T = options['T']
   f = np.sqrt(2)

   poly_clip = []
   a = area/T/2

   poly_clip.append([0,0])
   poly_clip.append([a/f,a/f])
   poly_clip.append([a/f-T*f,a/f])
   poly_clip.append([-T*f,0])
   poly_clip.append([a/f-T*f,-a/f])
   poly_clip.append([a/f,-a/f])

   return poly_clip
   
geo = Geometry(porosity=0.05,lx=100,ly=100,step=5,shape='custom',base=[[0,0]],lz=0,save=False,shape_function=shape,shape_options={'T':0.05})
 
 ```
 ##Custom
 
 With the custom model, the structured is defined a series of polygons defining the regions of the material to be carved out. Below is an example (also see figure below, panel c)
 
 ```
k = 0.1
h = 0.1
d = 0.07
poly1 = [[-k/2,0],[-k/2,-h],[k/2,0]]
poly2 = [[-0.6,0],[-0.6,-0.8],[0.6,-0.8],[0.6,-0],[k/2+d,0],[-k/2-d,-k-2*d],[-k/2-d,0]]

geo = Geometry(model='custom',lx=100,ly=100,step=5,polygons = [poly1,poly2])

```

Note that periodicity is endured when defining the polygons.
 
 
<p align="left">
<img  align="left" width="700" src="https://docs.google.com/drawings/d/e/2PACX-1vTZ57K5UB6qc_n56lKufOOYpEy8S8K_12fzD-oGRbnO5Rouc-aQhSbU3ci4euuUOl72EvEiszekMZos/pub?w=943&h=898">
</p>
 


# Solver

Solver reads the files `geometry.h5` and `material.h5` and, after solving the BTE, creates the dile `solver.h5`. Here are the list of options

|  Option  | Description    | Format |
|:-----------|:---------------|:-------------|
|   `max_bte_iter`  | max number of BTE iterations| int (50) |
|   `max_bte_error`  | error on the BTE effective thermal conductivity| flat (1e-3)  |
|   `max_fourier_iter`  | max number of Forier iterations| int (50) |
|   `max_fourier_error`  | error on the Fourier effective thermal conductivity| flat (1e-3)  |
|   `only_fourier`  |whether only Fourier is needed| Bool (False)  |
|   `geometry`  | the geometry object (to avod storing)| None  |
|   `material`  | the material object (to avod storing)| None  |

# Plot

`Plot` reads in input the files `material.h5`, `geometry.h5` and `solver.h5`. Currently, the possible `models` are `geometry` and `maps`. 


# Examples
https://nbviewer.jupyter.org/github/romanodev/OpenBTE/blob/master/openbte/Tutorial.ipynb  
The examples are provided via a [notebook](https://nbviewer.jupyter.org/github/romanodev/OpenBTE/blob/master/openbte/Tutorial.ipynb ) or [interactive simulations](https://colab.research.google.com/drive/1eAfX3PgyO7TyGWPee8HRx5ZbQ7tZfLDr) (Google Colab).

# Notes

### <a name="1"></a> 1. <pre> `2DSym` is a system that is infinite in the third dimension. Below are the cases are shown.

<p align="left">
<img  align="left" width="700" src="https://docs.google.com/drawings/d/e/2PACX-1vRUa6nwKHA_kCBaofjivbwPbmgweDab5xXCKdEesLTZF622a020f0xm7rlufdCufwhquPBLLTTFzrEO/pub?w=885&h=138">
</p>

</pre>

 



