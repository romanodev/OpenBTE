Vectorial MFP Interpolation
============================

Next step is to perform vectorial MFP interpolation, as documented `here <https://arxiv.org/abs/2105.08181>`_. For systems with translational symmetry along the z axis, you can use

.. code-block:: python

  from openbte import RTA2DSym

  rta_data = RTA2DSym(rta_data,**kwargs)

where options include ``n_mfp`` and ``n_phi``, i.e. the number of MFP (default = 50) and polar angle bins (default = 48). If you are simulating a 3D material (not yet fully supported), you can use 

.. code-block:: python

  from openbte import RTA3D

  mat = RTA3D(rta_data,**kwargs)

where options now include ``n_theta`` as well, i.e. the azimuthal angular bins. 



