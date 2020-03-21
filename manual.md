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

- __Database mode__. The module simply downloads a file from the online database. Example:

```python
Material(model='database',filename='C_0')
```
In this case `Material` simply retrieves the data from the [database](../database). The value of `filename` has to match one of the `Filename` fields of the database.

 -__Unlisted mode__. Similarly to the Database mode, the Unlisted mode retrives the material file from a `file_id`. This mode is useful when running OpenBTE via Colab with private input data. Example:
 
 ```python
Material(model='unlisted',filename='some_file_id')
```
    
 -__Local mode__. The file `material.h5` is created from scratch. Depending on the model being used, the input files have designed names. Example: 
    
```python
Material(model='rta',n_phi=92)
```

Currently, there are the follwing material models:

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


The format of input files is desribed in the [database](../database) section. 



# Geometry
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

 



