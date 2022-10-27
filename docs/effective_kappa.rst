Effective Thermal Conductivity
===============================

To compute the effective thermal conductivity, we average the flux over a given contact. For example, assuming a rectangular domain with size :math:`L_x \times L_y`, the effective thermal conductivity for a perturbation applied along x (:math:`\kappa_{xx}`) is given by 

.. math::

  \kappa_{xx} = \alpha\int_{-L_y/2}^{L_y/2}\mathbf{J}(L/2,y)\cdot \mathbf{\hat{n}}dy
  
where :math:`\alpha =-L_x/L_y/\Delta T_{\mathrm{ext}}`. To this end, we define the boundary region name and the normalization factor :math:`alpha`

.. code-block:: python

 from openbte.objects import EffectiveThermalConductivity

 effective_kappa = EffectiveThermalConductivity(normalization=-1,contact='Periodic_x')

