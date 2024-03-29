Line data
==========

Line data can be plotted with

.. code-block:: python

   N = 100

   path = np.stack((np.zeros(N),np.linspace(-0.5,0.5,N))).T

   x,data = results.plot_over_line(variables=['Temperature_Fourier','Temperature_BTE','Flux_fourier','Flux_BTE'],x=path)

The variable ``x`` is the distance on the path and ``data`` is a dictionary containing the data interpolated on the path.  For example, the temperature computed with BTE can be accessed with ``data['Temperature_BTE']``.
