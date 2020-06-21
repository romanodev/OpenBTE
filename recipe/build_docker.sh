#!/bin/bash


docker build -t romanodev/openbte .
#docker build --no-cache -t romanodev/openbte .
docker push romanodev/openbte
