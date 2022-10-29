Geometry Interface
===================

For simple geometries, it is possible to write Gmsh code automatically using the ``Geometry`` module. As an example, let's start with the creation of a porous material, consisting of a unit-cell and a square pore located in the center. The fist step is to build the ``Geometry`` object


.. code-block:: python

   from openbte import Geometry

   G = Geometry(0.1)

where ``0.1`` is the characteristic mesh size in nm. Then, the outer frame of the geometry is defined with the ``add_shape`` method

.. code-block:: python

  from openbte import rectangle

  L        = 10 #nm

  G.add_shape(rectangle(area = L*L,aspect_ratio = 1))

which in this case creates a square frame with side of 10 nm, whis is the unit-cell of the domain. The pore in the center added via

.. code-block:: python

  from openbte import rectangle

  porosity = 0.2

  area = porosity*L*L

  G.add_hole(rectangle(area = area,x=0,y=0))

If no name of the hole is given, then it will be considered as a void region and not included in the meshing. If a name if given, then it is included in the simulation domain and referred to it during the BTE solution, e.g. for heat sources. Finally, we have to define the boundary regions. This task entails selecting the boundary, through ``selector`` and associate a name to it. The following selelectors are available: ``outer``, ``inner``, ``all``, ``top``, ``bottom``, ``left`` and ``right``. In this case, we assign the name ``Boundary`` to all internal region, i.e. the wall of the pore.

.. code-block:: python

   G.set_boundary_region(selector = 'inner',region = 'Boundary')


Periodic boundary conditions can be assigned with the ``set_periodicity`` method

.. code-block:: python

  G.set_periodicity(direction = 'x',region      = 'Periodic_x')

  G.set_periodicity(direction = 'y',region      = 'Periodic_y')

The region names ``Boundary``, ``Periodic_x`` and ``Periodic_y`` will be referred to in the boundary conditions. Note that **all** the boundaries of the simulation domain need to be associated to a region name. Lastly, the file ``mesh.msh`` is created with

.. code-block:: python

   G.save()

To inspect your geometry, you can call ``gmsh`` from your command line

.. code-block:: bash

   gmsh mesh.geo

and check the physical regions in the ``visibility`` section of the ``tool`` drop-down menu.









