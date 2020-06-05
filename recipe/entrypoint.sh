#!/bin/bash --login
set -e
conda activate openbte
exec "$@"
