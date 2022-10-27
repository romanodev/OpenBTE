AlmaBTE Interface
============================

`AlmaBTE <https://almabte.bitbucket.io/>`_ is a package that computes the thermal conductivity of bulk materials, thin films and superlattices. OpenBTE is interfaced with AlmaBTE for RTA calculations via the script ``almabte2openbte.py``.

Assuming you have ``AlmaBTE`` in your current ``PATH``, this an example for ``Si``.

- Download Silicon force constants from AlmaBTE's `database <https://almabte.bitbucket.io/database/>`_

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

- A file named ``Si_8_8_8_300K.phononinfo`` is in your current directory. Note that you can specify the temperature. Here we choose 300 K. The file ``rta.db`` can then be created with

  .. code-block:: bash

     AlmaBTE2OpenBTE Si_8_8_8_300K.phononinfo

Finally, you can load the data with

.. code-block:: python

 from openbte import load_rta

 rta_data = load_rta('rta',source='local')

