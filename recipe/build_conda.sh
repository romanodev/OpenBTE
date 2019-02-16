#!/bin/bash

conda config --set anaconda_upload yes
conda build --numpy=1.14 -c guyer/label/hidden -c conda-forge --python 3.6 .
conda build purge
