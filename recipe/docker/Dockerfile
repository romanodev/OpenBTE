FROM continuumio/anaconda3
MAINTAINER Giuseppe Romano <romanog@mit.edu>

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y vim
RUN apt-get install -y libopenmpi-dev
RUN apt-get install -y mpich
RUN apt-get install -y libgl-dev libxcursor-dev libxft-dev libxinerama-dev libglu1-mesa

RUN useradd -ms /bin/bash openbte


RUN wget http://geuz.org/gmsh/bin/Linux/gmsh-3.0.0-Linux64.tgz
RUN tar -xzf gmsh-3.0.0-Linux64.tgz
RUN cp gmsh-3.0.0-Linux/bin/gmsh /usr/bin/
RUN rm -rf gmsh-3.0.0-Linux
RUN rm gmsh-3.0.0-Linux64.tgz

RUN pip install  openbte==0.9.33

USER openbte
WORKDIR /home/openbte

