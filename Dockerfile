FROM continuumio/anaconda3
MAINTAINER Giuseppe Romano <romanog@mit.edu>


RUN apt-get update 
RUN apt-get install -y libxft-dev libxinerama-dev libxcursor-dev sudo vim wget




RUN sudo wget http://geuz.org/gmsh/bin/Linux/gmsh-3.0.0-Linux64.tgz && \
    sudo tar -xzf gmsh-3.0.0-Linux64.tgz && \
    sudo cp gmsh-3.0.0-Linux/bin/gmsh /usr/bin/ && \
    sudo rm -rf gmsh-3.0.0-Linux && \
    sudo rm gmsh-3.0.0-Linux64.tgz

RUN apt-get install -y libopenmpi-dev mpich
RUN pip install --user shapely pyvtk unittest2 termcolor future alabaster\
    pyclipper deepdish networkx mpi4py

VOLUME /app/OpenBTE

WORKDIR /app/OpenBTE


RUN apt-get install -y libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev


RUN conda update -y qt pyqt -c conda-forge
RUN apt-get install -y qt5-default
RUN apt-get install -y iceweasel

RUN echo 'export PATH=/usr/local/bin:$PATH' >>~/.bashrc
RUN echo "alias julab=\"jupyter lab --browser=\"iceweasel\" --allow-root --ip=127.0.0.1 & \" " >>~/.bashrc
RUN echo "alias pp=\"mpiexec -np 32 --allow-run-as-root python input.py \" " >>~/.bashrc
RUN echo "alias p=\"python input.py \" " >>~/.bashrc
RUN echo "python /app/OpenBTE/setup.py develop " >>~/.bashrc

RUN conda upgrade -y spyder

#RUN apt-get install -y system-config-printer cups 
#RUN apt-get install -y hplip
#RUN sudo /etc/init.d/cups restart

WORKDIR /app/OpenBTE
CMD /bin/bash



