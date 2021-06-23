FROM continuumio/anaconda3
MAINTAINER Giuseppe Romano <romanog@mit.edu>

RUN apt-get update

RUN apt-get install -y vim

RUN conda update -n base -c defaults conda

RUN apt-get install -y build-essential libopenmpi-dev libgmsh-dev 

RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz

RUN tar -xzf openmpi-4.0.3.tar.gz

RUN cd openmpi-4.0.3 && ./configure --prefix=/usr/local/openmpi && make all && make install && rm -f openmpi-4.0.3.tar.gz 

COPY requirements.txt .
RUN pip install -r requirements.txt 

COPY openbte-2.34.tar.gz .
RUN pip install --no-cache openbte-2.34.tar.gz
RUN rm openbte-2.34.tar.gz

#RUN useradd -ms /bin/bash openbte

ENV OMPI_MCA_btl_vader_single_copy_mechanism none 

ENV OMPI_MCA_btl_base_warn_component_unused 0
#RUN mkdir /workspace

#USER openbte

#WORKDIR /home/openbte


EXPOSE 8050







