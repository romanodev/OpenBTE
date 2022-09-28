Material
===================================

A material model takes bulk data as input and create the file ``material.npz``. There are several material models, as expanded upon below.

Gray model approximation
-----------------------------------

The simplest model assumes single-MFP BTE. To create ``material.npz`` the mean-free-path (in nm) and the bulk thermal conductivity must be specified. Here is an example:

.. code-block:: python

   Material(model='gray',mfp=100,kappa=130)

Note that the gray model assumes a two-dimensional material.

Relaxation time approximation
-----------------------------------

The RTA model implements the anisotropic-mean-free-path-BTE (aMFP-BTE_). It needs a file called ``rta.npz`` in your current directory, with the following properties

.. table:: 
   :widths: auto
   :align: center

   +--------------------------+-------------+--------------------------------------------------------------------------+--------------------------+
   | **Item**                 | **Shape**   |       **Symbol [Units]**                                                 |    **Name**              |
   +--------------------------+-------------+--------------------------------------------------------------------------+--------------------------+
   | ``scattering_time``      |  N          |   :math:`\tau` [:math:`s`]                                               | Scattering time          |
   +--------------------------+-------------+--------------------------------------------------------------------------+--------------------------+
   | ``heat_capacity``        |  N          |   :math:`C` [:math:`\mathrm{W}\mathrm{s}\textrm{K}^{-1}\textrm{m}^{-3}`] | Specific heat capacity   |
   +--------------------------+-------------+--------------------------------------------------------------------------+--------------------------+
   | ``group_velocity``       |  N x 3      |   :math:`\mathbf{v}` [:math:`\mathrm{m}\textrm{s}^{-1}`]                 | Group velocity           |
   +--------------------------+-------------+--------------------------------------------------------------------------+--------------------------+
   | ``frequency``            |  N          |   :math:`f` [:math:`Hz`]                                                 | Frequency                |
   +--------------------------+-------------+--------------------------------------------------------------------------+--------------------------+
   | ``thermal_conductivity`` |  3 x 3      |   :math:`\kappa` [:math:`\mathrm{W}\textrm{K}^{-1}\textrm{m}^{-1}`]      | Thermal conductivity     |
   +--------------------------+-------------+--------------------------------------------------------------------------+--------------------------+


Each item must be a ``numpy`` array with the prescribed shape. The thermal conductivity tensor is given by :math:`\kappa^{\alpha\beta} = \sum_{\mu} C_\mu  v_\mu^{\alpha} v_\mu^{\beta} \tau_\mu`. To save a dictionary in the ``npz`` format, we recommend to use following script

.. code-block:: python

   from openbte.utils import save,check_rta_data

   data = {'scattering_time':tau,'heat_capacity':C,'group_velocity':v,'thermal_conductivity':kappa}

   check_rta_data(data)

   save(data,'rta')

With ``rta.npz`` in your current directory, ``material.npz`` can be generated with

.. code-block:: python

   Material(model='rta2DSym',**options)

.. table:: 
   :widths: auto
   :align: center

   +------------------------+-------------------------+-------------------+
   | **Item**               | **Note**                |    **Default**    |                                               
   +------------------------+-------------------------+-------------------+
   | ``n_mfp``              |  number of mfps         |        50         |
   +------------------------+-------------------------+-------------------+
   | ``n_phi``              |  number of polar angles |        48         |
   +------------------------+-------------------------+-------------------+


Interface with AlmaBTE
###############################################

AlmaBTE_ is a popular package that compute the thermal conductivity of bulk materials, thin films and superlattices. OpenBTE is interfaced with AlmaBTE for RTA calculations via the script ``almabte2openbte.py``. 

Assuming you have ``AlmaBTE`` in your current ``PATH``, this an example for ``Si``.

- Download Silicon force constants from AlmaBTE's database_.

  .. code-block:: bash

   wget https://almabte.bitbucket.io/database/Si.tar.xz   
   tar -xf Si.tar.xz && rm -rf Si.tar.xz  

- Compute bulk scattering time with AlmaBTE.

  .. code-block:: bash

   echo "<singlecrystal> 
   <compound name='Si'/>
   <gridDensity A='8' B='8' C='8'/>
   </singlecrystal>" > inputfile.xml
   
   VCAbuilder inputfile.xml
   phononinfo Si/Si_8_8_8.h5 300.0
    
- A file named ``Si_8_8_8_300K.phononinfo`` is in your current directory. Note that you can specify the temperature. Here we choose 300 K. `The file ``rta.npz`` can then be created with 

  .. code-block:: bash

     AlmaBTE2OpenBTE Si_8_8_8_300K.phononinfo

- Using OpenBTE command line interface, the ``material`` may be created with

  .. code-block:: bash

     OpenBTE $'Material:\n model: rta2DSym'

Interface with Phono3Py
###############################################

Phono3py_ calculates the bulk thermal conductivity using both the RTA and full scattering operator. Currently, only the former is supported. Once Phono3py is solved, the ``rta.npz`` is created by


.. code-block:: bash

   phono3pytoOpenBTE unitcell_name nx ny nz 

where ``unitcell_name`` is the file of your unit cell and ``nx ny nz`` is the reciprical space discretization.

Here is an example assuming you have a working installation of Phono3py:

.. code-block:: bash

   git clone https://github.com/phonopy/phono3py.git

   cd phono3py/examples/Si-PBEsol

   phono3py --dim="2 2 2" --sym-fc -c POSCAR-unitcell

   phono3py --dim="2 2 2" --pa="0 1/2 1/2 1/2 0 1/2 1/2 1/2 0" -c POSCAR-unitcell --mesh="8 8 8"  --fc3 --fc2 --ts=100

   Phono3py2OpenBTE POSCAR-unitcell 8 8 8 

Note that ``rta.npz`` is also created in the case you want to use a RTA model.   


.. _Deepdish: https://deepdish.readthedocs.io/
.. _Phono3py: https://phonopy.github.io/phono3py/
.. _AlmaBTE: https://almabte.bitbucket.io/
.. _database: https://almabte.bitbucket.io/database/
.. _aMFP-BTE: https://arxiv.org/abs/2105.08181
.. _Deepdish: https://deepdish.readthedocs.io/
.. _`Wu et al.`: https://www.sciencedirect.com/science/article/pii/S0009261416310193?via%3Dihub
.. _Phono3py: https://phonopy.github.io/phono3py/





