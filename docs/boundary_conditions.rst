Boundary Conditions
=============================

The object ``BoundaryConditions`` connects the physical boundaries defined in ``Geometry`` with the actual physics. There are three boundary conditions: ``diffuse``, ``mixed`` and ``periodic``. In case of periodic boundary conditions, we can also specify a temperature jump (albeit is not strictly a temperature jump but a heat source/sink pair) applied along the associated direction. In our case, we apply a temperature jump of 1 K along x and associate the region ``Boundary`` to diffuse boundary conditions

.. code-block:: Python

   from openbte.objects import BoundaryConditions

   boundary_conditions = BoundaryConditions(periodic={'Periodic_x': 1,'Periodic_y':0},diffuse='Boundary')

The mixed boundary conditions include a thermostatting boundaries and a boundary conductance [in Wm :math:`^-{2}` K :math:`^{-1}`]


.. code-block:: Python

   from openbte.objects import BoundaryConditions

   boundary_conditions = BoundaryConditions(mixed={'Isothermal': {'value':300,'boundary_conductance':1e4}})

where we assumed that a region named ``Isothermal`` was previously defined.
