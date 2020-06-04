FROM continuumio/anaconda3
MAINTAINER Giuseppe Romano <romanog@mit.edu>

RUN apt-get update

RUN apt-get install -y libglu1-mesa libxcursor1 libxft-dev libxinerama-dev

RUN pip install --upgrade gmsh

RUN apt-get install -y vim  build-essential libopenmpi-dev swig libsuitesparse-dev
RUN useradd -ms /bin/bash openbte
RUN pip install --upgrade --no-cache openbte

EXPOSE 8050

USER openbte
WORKDIR /home/openbte

