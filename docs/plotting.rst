Plotting thermal and heat flux maps
===================================

Flux and temperatures maps can be visualized with

.. code-block:: python

   from openbte.objects import OpenBTEResults

   results = OpenBTEResults.load()

   results.show()

For advanced visualization, e.g. slicing etc..., you can save results in the ``.vtu`` format

.. code-block:: python

   results.vtu()

This will craete a file, ``output.vtu``, compatible with the popular software `Paraview <https://www.paraview.org/>`_ .   


