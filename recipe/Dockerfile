FROM continuumio/anaconda3
MAINTAINER Giuseppe Romano <romanog@mit.edu>

RUN apt-get update

RUN apt-get install -y libglu1-mesa libxcursor1 libxft-dev libxinerama-dev

RUN apt-get install -y vim

RUN conda update -n base -c defaults conda

RUN conda create -n openbte -c gromano -c conda-forge -c ostrokach-forge openbte==1.13 

RUN echo "source activate openbte" > ~/.bashrc
ENV PATH /opt/conda/envs/openbte/bin:$PATH


#SHELL [ "/bin/bash", "--login", "-c" ]

#CMD [ "sh", "-c", "conda activate openbte" ]



#COPY entrypoint.sh .

# make non-activate conda commands available
#ENV PATH=/opt/conda/bin:$PATH
# make conda activate command available from /bin/bash --login shells
#RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.profile

#RUN conda init bash

#CMD conda init bash &&  conda activate openbte

#ENTRYPOINT set -e && conda activate openbte 

#ENTRYPOINT python input.py

EXPOSE 8050







