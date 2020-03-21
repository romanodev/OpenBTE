---
layout: default
title: Format
permalink: /format/
description: Input file format
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

Below we report the format for the files `hdf5` files `rta.h5`, `mfp.h5` and `full.h5`. All values are intended to be `numpy` arrays. A good Pythonic option for creating `hdf5` files is [`deepdish`](https://deepdish.readthedocs.io/en/latest/io.html).

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
