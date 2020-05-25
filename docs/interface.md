---
layout: default
title: Interface
permalink: /interface/
description: Interfaces to external solvers
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

# AlmaBTE

This interface converts the output of [AlmaBTE](http://www.almabte.eu/)'c command `phononinfo` to OpenBTE format and write the file `rta.f5`. Let's assume you have the file `Si_4_4_4_300K.phononinfo` in your current directory. Then simply type

```python
AlmaBTE2OpenBTE Si_4_4_4_300K.phononinfo
```
The created file, `rta.h5`, can be used by a `Material` [statement](../manual).



