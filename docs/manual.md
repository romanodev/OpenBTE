---
layout: default
title: Manual
description: Here we show software architecture
---

<p align="center">
<img  align="center" width="700" src="https://docs.google.com/drawings/d/e/2PACX-1vRqrihU3IHGVNRaNN7sc2r5CMphXVz6iT8jesHsX0blyj7GPh5KyiUiOFw8WMH9bHHNZYMzBTIgLPNo/pub?w=800&h=500">
</p>

# Material

The first step is to generate `material.h5`, a file in `hdf5` format that containes all the necessary information related to the bulk material. There are three material models: `MFP`, `RTA`, `FULL`; the module `Material` converts data from bulk first-principles calculations to `material.h5`. For example, to create the `MFP` model, we have

```python
Material(model='MFP')
```

In the current directory, we must have a file named `mfp.h5`, with the follwing fields:

# Geometry
# Solver
# Plot


