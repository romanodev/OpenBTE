Example 1: Porous Material
==========================

An OpenBTE simulation is specified by a combination of a material, geometry and solver. We begin with creating a material. To this end, we load previously computed first-principled calculations on Si at room temperature 

.. code-block:: python

   from openbte import load_rta
   
   rta_data = load_rta('Si_rta')

Next step is to perform MFP interpolation

.. code-block:: python

   from openbte import RTA2DSym

   mat = RTA2DSym(rta_data)

The ``RTA2DSym`` model is for simulation domains which have translational symmetry along the z axis. To create a geometry, we instantiate an object of the class ``Geometry``

.. code-block:: python

   from openbte import Geometry

   G = Geometry(0.1)


where :math:`0.1` is the carhacteristic size (in nm) of the mesh. In this example, we create a porous material with porosity :math:`0.2` and rectangular aligned pores. Given the periodicity of the system, we simulate only a unit-cell to which we apply periodic boundary conditions. To define the unit-cell, we use the ``add_shape`` method

.. code-block:: python

   from openbte import rectangle
   
   L        = 10 #nm

   G.add_shape(rectangle(area = L*L))

To add the hole in the middle, we use the ``add_hole`` method   

.. code-block:: python
   
   porosity = 0.2 

   area = porosity*L*L

   G.add_hole(rectangle(area = area,x=0,y=0))

To apply boundary conditions, we need to assign a name to sides and refer to them in the solver section. Sides are selected with ``selector``. In this case, we assign all internal sides the name ``Boundary``

.. code-block:: python

   G.set_boundary_region(selector = 'inner',region = 'Boundary')

To apply periodic boundary conditions along both axes, we use the ``set_periodicity`` method

.. code-block:: python

  G.set_periodicity(direction = 'x',region      = 'Periodic_x')

  G.set_periodicity(direction = 'y',region      = 'Periodic_y')

At this point, we are ready to save the mesh on disk

.. code-block:: python

   G.write_geo()

If everything went smoothly, you should see ``mesh.geo`` in your current directory. You can open them with GMSH_ to check that the geometry has been created correctly. To create a meshed geometrym we use the function ``get_mesh()``

.. code-block:: python

   from openbte import get_mesh

   mesh = get_mesh()


Before setting up the solvers, we need to specify boundary conditions and perturbation. In this case, we apply a difference of temperature of :math:`\Delta T_{\mathrm{ext}} = 1` K along x

.. code-block:: python

 from openbte.objects import BoundaryConditions

 boundary_conditions = BoundaryConditions(periodic={'Periodic_x': 1,'Periodic_y':0},diffuse='Boundary')

Note that we also specifies diffuse boundary conditions along the region ``Boundary``. In this example, we are interested in the effective thermal conductivity along x

.. code-block:: python

 from openbte.objects import EffectiveThermalConductivity

 effective_kappa = EffectiveThermalConductivity(normalization=-1,contact='Periodic_x')

where ``normalization`` (:math:`\alpha`) is used in the calculation of the effective thermal conductivity :math:`\kappa_{\mathrm{eff}} = \alpha\int_{-L/2}^{L/2}\mathbf{J}(L/2,y)\cdot \mathbf{\hat{n}}dy`. For rectangular domain, :math:`\alpha =-L_x/L_y/\Delta T_{\mathrm{ext}}`.

To run BTE calculations, we first solve standard heat conduction

.. code-block:: python

 from openbte import Fourier

 fourier     = Fourier(mesh,mat.thermal_conductivity,boundary_conditions,\
                        effective_thermal_conductivity=effective_kappa)

Finally, using ``fourier`` as first guess, we solve the BTE

.. code-block:: python

 from openbte import BTE_RTA

 bte     = BTE_RTA(mesh,mat,boundary_conditions,fourier=fourier,\
           effective_thermal_conductivity=effective_kappa)

Before plotting the results, we group together Fourier and BTE results 

.. code-block:: python

   from openbte.objects import OpenBTEResults

   results =  OpenBTEResults(mesh=mesh,material = mat,solvers=[fourier,bte])

Lastly, the temperature and heat flux maps can be obtained with

.. code-block:: python

   results.show()

.. raw:: html

    <iframe src="_static/plotly.html" height="475px" width="100%"  display= inline-block  ></iframe>


`GMSH <https://gmsh.info/>`_






