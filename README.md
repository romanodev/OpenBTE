
<img src="docs/_static/openbte_logo.png" width="300">

[![license](https://img.shields.io/github/license/romanodev/openbte)](https://github.com/romanodev/OpenBTE/blob/master/LICENSE)
[![documentation](https://readthedocs.org/projects/pip/badge/?version=latest)](https://openbte.readthedocs.io/en/latest/)
[![downloads](https://img.shields.io/pypi/dm/openbte)](https://pypi.org/project/openbte/)
[![Open in Colab (Inverse Design)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DluLzUBbXKNz6K14R1gErOi17u9W_CLu?usp=sharing)
[![Python](https://img.shields.io/pypi/pyversions/openbte)](https://www.python.org/)

OpenBTE computes mode-resolved phonon transport using the Boltzmann transport equations. Main features include:

- Interfaced with first-principles solvers (e.g. AlmaBTE)
- Diffuse, isothermal, and periodic boundary conditions.
- Efficient discretization based on the vectorial phonon mean-free-paths (MFP)
- The full anisotropy of the phonon dispersion is retained, as opposed to commonly used isotropization.
- GPU-Accelerated differentiable phonon transport simulations based on JAX (added in 2022)
- Inverse design capabilities

Developed by Giuseppe Romano (romanog@mit.edu).

Coming soon: Steady-state BTE using the full scattering operator [Link](https://arxiv.org/abs/2002.08940). You can provide your own scattering operator (only upper diagonal) and perform space-dependent simulations.

**References**:

G. Romano, OpenBTE: a Solver for ab-initio Phonon Transport in Multidimensional Structures, arXiv:2106.02764, (2021) [Link](https://arxiv.org/abs/2106.02764)

G. Romano and S. G. Johnson, Inverse design in nanoscale heat transport via interpolating interfacial phonon transmission, Structural and Multidisciplinary Optimization, (2022)  [Link](https://arxiv.org/abs/2202.05251) 

G. Romano, Efficient calculations of the mode-resolved ab-initio thermal conductivity in nanostructures, arXiv:2105.08181 (2021) [Link](https://arxiv.org/abs/2105.08181)  

G. Romano, A Di Carlo, and J.C. Grossman, Mesoscale modeling of phononic thermal conductivity of porous Si: interplay between porosity, morphology and surface roughness. Journal of Computational Electronics 11 (1), 8-13 52 (2012) [Link](https://link.springer.com/article/10.1007/s10825-012-0390-2)







