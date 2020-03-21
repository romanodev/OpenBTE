---
layout: default
title: Material
permalink: /material/
description: Material Model
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
