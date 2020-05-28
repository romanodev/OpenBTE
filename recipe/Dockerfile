FROM continuumio/anaconda3
MAINTAINER Giuseppe Romano <romanog@mit.edu>

RUN apt-get update
RUN apt-get install -y  build-essential libopenmpi-dev libgmsh-dev swig libsuitesparse-dev
RUN useradd -ms /bin/bash openbte
RUN pip install --upgrade --no-cache openbte
USER openbte
WORKDIR /home/openbte

