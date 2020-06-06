FROM continuumio/anaconda3
MAINTAINER Giuseppe Romano <romanog@mit.edu>

RUN apt-get update

#RUN apt-get install -y libglu1-mesa libxcursor1 libxft-dev libxinerama-dev

RUN apt-get install -y vim


RUN conda update -n base -c defaults conda

RUN apt-get install -y build-essential libopenmpi-dev libgmsh-dev swig libsuitesparse-dev

RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz

RUN tar -xzf openmpi-4.0.3.tar.gz

RUN cd openmpi-4.0.3 && ./configure --prefix=/usr/local/openmpi && make all && make install && rm -f openmpi-4.0.3.tar.gz 

RUN wget http://geuz.org/gmsh/bin/Linux/gmsh-4.5.6-Linux64.tgz
RUN tar -xzf gmsh-4.5.6-Linux64.tgz
RUN cp gmsh-4.5.6-Linux64/bin/gmsh /usr/bin/
RUN rm -rf gmsh-4.5.6-Linux64
RUN rm gmsh-4.5.6-Linux64.tgz

COPY openbte-1.14.tar.gz .
RUN pip install openbte-1.14.tar.gz

RUN rm openbte-1.14.tar.gz
#RUN pip install openbte==1.14

RUN useradd -ms /bin/bash openbte

ENV OMPI_MCA_btl_vader_single_copy_mechanism none 

USER openbte

WORKDIR /home/openbte

#RUN conda create -n openbte -c gromano -c conda-forge -c ostrokach-forge openbte==1.13 

#RUN echo "source activate openbte" > ~/.bashrc
#ENV PATH /opt/conda/envs/openbte/bin:$PATH


EXPOSE 8050







