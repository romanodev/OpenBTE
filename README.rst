Description
===========

Tool for calculate tile bounding box (in geodesic coordinates) by specified XYZ (x - column, y - row, z - zoom) of tile.


Installation
============

::

  pip install sphericalmercator


Usage
=====

::

  from sphericalmercator import SphericalMercator

  merc = SphericalMercator(levels=15, size=256)
  print merc.xyz_to_envelope(x=2474, y=1280, zoom=12)
  # outputs (37.44140625, 55.727110085045986, 37.529296875, 55.7765730186677)
