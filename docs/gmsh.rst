Gmsh
==========

OpenBTE uses `Gmsh <https://gmsh.info/>`_  as a backend for geometry building. Once the file ``mesh.msh`` is created, it can be imported with

.. code-block:: python

 from openbte import get_mesh

 mesh = get_mesh()

The physical regions will then be referred to by boundary conditions. OpenBTE handles 2D and 3D geometries, although only 2D systems are currently supported. The name of physical regions associating to two periodic boundaries must end with ``_a`` and ``_b``. 

