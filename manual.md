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
Material(model='mfp',**options)
```

In the current directory we must have a file named `mfp.h5`, with the follwing fields:

| Name     | Size     | Units |  
|:-------------|:------------------|
| `mfp`    | $$N$$      | m  | 
| `K`    | $$N$$      | Wm$$^{-1}$$K$$^{-1}$$  |    
    
where $$N$$ is the number of MFPs. The `mfp` model accept the following options:

| Field     | Values (default)     | Description |  
|:-------------|:------------------|
| `submodel`    |  `2DSym`,`3D`, `2D` (`2DSym`)  |  Dimensionality of the system(NOTE) |
| `n_phi`    | Integer (48)      | Polar angle discretization| 
| `n_theta`  | Integer (24)      | Azimuthal angle discretization| 
| `n_mfp`    | Integer(50)       | MFPs discretiztion. It must be less than $$N$$.|    


   `2DSym`           `3D`             `2D`
<p align="center">
<img  align="center" width="700" src="https://docs.google.com/drawings/d/e/2PACX-1vRUa6nwKHA_kCBaofjivbwPbmgweDab5xXCKdEesLTZF622a020f0xm7rlufdCufwhquPBLLTTFzrEO/pub?w=885&h=138">
</p>

# Geometry
# Solver
# Plot
# Examples
https://nbviewer.jupyter.org/github/romanodev/OpenBTE/blob/master/openbte/Tutorial.ipynb  
The examples are provided via a [notebook](https://nbviewer.jupyter.org/github/romanodev/OpenBTE/blob/master/openbte/Tutorial.ipynb ) or [interactive simulations](https://colab.research.google.com/drive/1eAfX3PgyO7TyGWPee8HRx5ZbQ7tZfLDr) (Google Colab).
 



