---
layout: default
title: Database
permalink: /database/
description: User-generated input data
---

## Here we keep a list of self-hosted data from users. 

| Filename     | T[K]     | Type |  Author(s)          | contact(s) |  Paper(s) | 
|:-------------|:------------------|:------|:-------|
| [Si_0](#si0)     | 300      | MFP  | Giuseppe Romano  | romanog@mit.edu |  [P1](https://www.sciencedirect.com/science/article/pii/S0010465517302059) |


## Adding your data

If you have published material data and wish to share if with the community, you may want to add an entry to the database above. The database above has the following features:

 - The object being stored are the `material.h5` files.
 - Data is self-hosted, i.e. you will be providing a link to your file. Currently, we support Google drive's file ids but more platforms are being added.
 - The quality of the data is not endorsed. We only check that all the needed fields of the hdf5 files are correctly populated. 
 - Once launched, OpenBTE will show the user the recommended citations based on the used input data.

The first step in adding your own data is, actually, having published data. If you have high quality data but haven't published it yet, you may want to consider submitting it to [zenodo](https://zenodo.org/) first. Once you are ready, please fill this [form](https://forms.gle/Kjhky3wjrrghXBb48). 

## Format

The file `material.h5` is created by using the `Material` module in connection with an input file, a `model`, and additional options. Please refer to the [manual](manual.html) for more details. Depending on the chosen model, you might need a file with designed name `full.h5`, `rta.h5` or `mfp.h5`. A good option for creating such a files is [`deepdish`](https://deepdish.readthedocs.io/en/latest/io.html). The format of each file are explained below. Note that the values are intended to be `numpy` arrays.

## `mfp.h5`


| Name     | Size     | Units |  
|:-------------|:------------------|
| `mfp`    | $$N$$      | m  | 
| `K`    | $$N$$      | Wm$$^{-1}$$K$$^{-1}$$  |    
    
where $$N$$ is the number of MFPs.     


## Notes

Here we include the notes from the above database.

### <a name="si0"></a> Si_0 

Generated with [AlmaBTE](http://www.almabte.eu/), using a 32-32-32 grid, RTA, and force constants from [AlmaBTE's database](http://www.almabte.eu/index.php/database/).


[back](./)
