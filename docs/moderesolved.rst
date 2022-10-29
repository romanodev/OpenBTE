Mode-Resolved Bulk Data
==========================

The material underlying the structure is described by the following mode-resolved quantities:

+-----------------------------------------------+---------------------------------+----------------------------------+
| Property [symbol]                             |            Shape                |      Units                       |
+===============================================+=================================+==================================+
| ``scattering_time`` [:math:`\tau]`            |             :math:`N`           |        s                         |
+-----------------------------------------------+---------------------------------+----------------------------------+
| ``heat_capacity`` [:math:`C]`                 |             :math:`N`           |  Jm :math:`^{-3}` K :math:`^{-1}`|
+-----------------------------------------------+---------------------------------+----------------------------------+
| ``group_velocity`` [:math:`\mathbf{v}`]       |             :math:`N\times 3`   |   ms :math:`^{-1}`               |
+-----------------------------------------------+---------------------------------+----------------------------------+


where :math:`N=N_q\times N_b`, with :math:`N_q` and :math:`N_b` being the number of wave-vectors and polarization, respectively. The bulk thermal conductivity tensor is given by 

.. math::

   \kappa^{\alpha\beta}=\sum_\mu C_\mu \tau_\mu v_\mu^\alpha v_\mu^\beta.

The (specific) heat capacity is defined by

.. math::

   C_\mu = \frac{k_B}{N_q\mathcal{V}} \left[\frac{\eta_\mu}{\sinh(\eta_\mu)}\right]^2

with :math:`\eta_\mu = \hbar \omega_\mu/(2 k_b T)`. Note that the presence of :math:`N_q` in the heat capacity is not standard but introduced here for convenience.

The material data is stored as a ``sqlite3`` database (with `.db` extension), which can be conveniently handled by `sqlitedict <https://pypi.org/project/sqlitedict/>`_. With your material file, say ``foo.db``, is in the current directory, you can load it with

.. code-block:: python

  from openbte import load_rta

  rta_data = load_rta('foo',source='local')

The above code is nothing than a wrapper to the ``sqlite3``'s loader. Note that there is also a minimal database of precomputed materials. Currently, it includes silicon at room temperature, computed with `AlmaBTE <https://almabte.bitbucket.io/>`_.

.. code-block:: python

  from openbte import load_rta

  rta_data = load_rta('Si_rta',source='database')

To save your own material data, you can also use a wrapper

.. code-block:: python

  import openbte.utils

  #C = ...
  #v   = ...
  #tau = ...

  utils.save('rta',{'scattering_time':tau,'heat_capacity':C,'group_velocity':v}

You can double-check the consistency of your data by comparing the resulting thermal conductivity with the expected one

.. code-block:: python

  import numpy as np
  #C = ...
  #v   = ...
  #tau = ...
  #kappa = ... #Expected thermal conductivity tensor

  print(np.allclose(np.einsum('u,ui,uj,u->ij',C,v,v,tau)-kappa))

  



