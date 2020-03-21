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

The first step is to generate `material.h5`, a file in `hdf5` format that containes all the necessary information related to the bulk material. There are three material models: `MFP`, `RTA`, `FULL`; the module `Material` converts data from bulk first-principles calculations to `material.h5`. For example, to create the `MFP` model, we have

```python
Material(model='rta',n_phi=92)
```

Currently, there are the following material models and options:

|  `model`   | Description    |  File  |   Options     |  
|:-----------|:---------------|:-------|:--------------|
|   `full`     | Full scattering operator | `full.h5` | None |
|   `rta`     | Relaxation time approximation (RTA) | `rta.h5` | `n_phi`: # of polar angles (`48`) <br> `n_theta`: # of azimuthal angles (`24`)  <br> `n_mfp`: # of mean free paths (`50`)|
|   `rta2D`     | RTA for materials with infinite thickness and two-dimensional (2D) materials| `rta.h5` | `n_phi`: # of polar angles (`48`)  <br> `n_theta`: # of azimuthal angles (`24`)  <br> `n_mfp`: # of mean free paths (`50`)|
|   `mfp`     | mean-free-path BTE| `mfp.h5` | `n_phi`: # of polar angles (`48`)  <br> `n_theta`: # of azimuthal angles (`24`) <br> `n_mfp`: # of mean free paths (`50`)|
|   `mfp2DSim`     | mean-free-path BTE| `mfp.h5` | `n_phi`: # of polar angles (`48`) <br> `n_mfp`: # of mean free paths (`50`)|
|   `mfp2D`     | mean-free-path BTE| `mfp.h5` | `n_phi`: # of polar angles (`48`)   <br> `n_theta`: # of azimuthal angles (`24`)<br> `n_mfp`: # of mean free paths (`50`)|
|   `gray`     | single MFP| None | `n_phi`: # of polar angles (`48`)   <br> `n_theta`: # of azimuthal angles (`24`) <br> `mfp`: mean free path (m) <br> `kappa`: bulk thermal conductivity (Wm$$^{-1}$$k$$^{-1}$$)|
|   `gray2Dsim`     | single MFP in materials with infinite thickness| None | `n_phi`: # of polar angles (`48`) <br> `mfp`: mean free path (m) <br> `kappa`: bulk thermal conductivity (Wm$$^{-1}$$k$$^{-1}$$)|
|   `gray2D`     | single MFP in 2D materials| None | `n_phi`: # of polar angles (`48`) <br> `mfp`: mean free path (m) <br> `kappa`: bulk thermal conductivity (Wm$$^{-1}$$k$$^{-1}$$)|
|   `fourier`     | Fourier's law| None | `kappa`: bulk thermal conductivity (Wm$$^{-1}$$k$$^{-1}$$)|





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

 



