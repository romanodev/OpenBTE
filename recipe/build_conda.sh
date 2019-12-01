#!/bin/bash

conda config --set anaconda_upload no
#cd ../
#sudo python setup.py sdist
#cd recipe
CONDA_INSTRUMENTATION_ENABLED=1 conda build --python=3.5 --numpy=1.14 -c conda-forge meta.yaml
conda build purge
