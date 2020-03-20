---
layout: default
title: Database
permalink: /database/
description: User-generated input data
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

## Here we keep a list of self-hosted data from users. 

| Filename     | T[K]     | Type |  Author(s)          | contact(s) |  Paper(s) | 
|:-------------|:------------------|:------|:-------|
| [Si_0](#si0)     | 300      | MFP  | Giuseppe Romano  | romanog@mit.edu |  [P1](https://www.sciencedirect.com/science/article/pii/S0010465517302059) |


## Background

If you have published material data and wish to share if with the community, you may want to add an entry to the database above. The database above has the following features:

 - The object being stored are the `material.h5` files.
 - Data is self-hosted, i.e. you will be providing a link to your file. Currently, we support Google drive's file ids but more platforms are being added.
 - The quality of the data is not endorsed. We only check that all the needed fields of the hdf5 files are correctly populated. 
 - Once launched, OpenBTE will show the user the recommended citations based on the used input data.

## Adding your own data

The first step in adding your own data is, actually, having published data. If you have high quality data but haven't published it yet, you may want to consider submitting it to [zenodo](https://zenodo.org/) first. Once you are ready, please follow these steps:

 - Depending on the material model, create one of these files: `full.h5`, `mfp.h5`, `rta.h5`. The format of these files is described below.
 
 - Create the `material.h5` file. For this step, please refer to the [manual](../manual.html). We recomment using the option  `check_kappa = True`. This flag compares the bulk thermal conductivity computed by OpenBTE with the one provided by the user (tolerance $$1e{-3}$$). See the [format](#format) section for more details.

 - Fill this [form](https://forms.gle/Kjhky3wjrrghXBb48). 

## <a name="format"></a> Format

The file `material.h5` is created by using the `Material` module in connection with an input file, a `model`, and additional options. Please refer to the [manual](manual.html) for more details. Depending on the chosen model, you might need a file with designed name `full.h5`, `rta.h5` or `mfp.h5`. A good option for creating such a files is [`deepdish`](https://deepdish.readthedocs.io/en/latest/io.html). The format of each file are explained below. Note that the values are intended to be `numpy` arrays.

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
| `alpha`    | $$1$$      | m$$^{-3}$$ | $$V^{-1}N_q^{-1}$$ ($$V$$: volume of the unit cell) |
| `A`     | $$N \times N $$ | s$$^{-1}$$  | scattering operator as Eq. 7 in [this](https://arxiv.org/pdf/2002.08940.pdf) |
| `v` | $$ N \times 3 $$| ms$$^{-1}$$ | group velocity |
| `kappa`    | $$3 \times 3$$      | Wm$$^{-1}$$K$$^{-1}$$  | bulk thermal conductivity tensor | 


### `rta.h5`

| Field     | Size     | Units |  Description |
|:-------------|:------------------|:---------|:---------|
| `f`    | $$N$$      | s$$^{-1}$$  | frequency ($$N = N_bN_q$$, where $$N_b:$$ number of branches, $$N_q$$: number of wave vectors) |
| `alpha`    | $$1$$      | m$$^{-3}$$ | $$V^{-1}N_q^{-1}$$ ($$V$$: volume of the unit cell) |
| `tauinv`     | $$NN $$ | s$$^{-1}$$  | scattering rates as Eq. 7 in [this](https://arxiv.org/pdf/2002.08940.pdf) |
| `v` | $$ N \times 3 $$| ms$$^{-1}$$ | group velocity |
| `kappa`    | $$3 \times 3$$      | Wm$$^{-1}$$K$$^{-1}$$  | bulk thermal conductivity tensor |  

## Notes

Here we include the notes from the above database.

### <a name="si0"></a> Si_0 

Generated with [AlmaBTE](http://www.almabte.eu/), using a 32-32-32 grid, RTA, and force constants from [AlmaBTE's database](http://www.almabte.eu/index.php/database/).


[back](./)
