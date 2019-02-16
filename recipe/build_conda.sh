#!/bin/bash

conda build --numpy=1.14 -c guyer/label/hidden -c conda-forge --python 3.6 openbte
