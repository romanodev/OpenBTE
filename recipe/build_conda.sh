#!/bin/bash

conda build --numpy=1.14 -c conda-forge -c  guyer/label/hidden -c conda-forge/label/cf201901 --python 2.7 openbte
