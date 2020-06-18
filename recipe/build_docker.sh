#!/bin/bash


cp ../dist/openbte-1.16.tar.g .
docker build -t romanodev/openbte .
#docker build --no-cache -t romanodev/openbte .
docker push romanodev/openbte
