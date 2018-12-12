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
RUN apt-get install -y system-config-printer cups 
RUN apt-get install -y hplip
RUN conda upgrade -y spyder
RUN apt-get install -y git

RUN echo 'export PATH=/usr/local/bin:$PATH' >>~/.bashrc
RUN echo "alias julab=\"jupyter lab --browser=\"iceweasel\" --allow-root --ip=127.0.0.1 & \" " >>~/.bashrc
RUN echo "alias pp=\"mpiexec -np 32 --allow-run-as-root python input.py \" " >>~/.bashrc
RUN echo "alias p=\"python input.py \" " >>~/.bashrc
RUN echo "git clone https://github.com/romanodev/OpenBTE.git /app/OpenBTE" >>~/.bashrc
RUN echo "python setup.py develop">>~/.bashrc



RUN echo "alias push=\" git add --all && git commit -m "auto" && git push          \" " >>~/.bashrc


RUN echo "export LS_OPTIONS='--color=auto' " >> ~/.bashrc
RUN echo "alias ls=\" ls \$LS_OPTIONS \" " >>~/.bashrc
RUN echo "alias ll=\"ls \$LS_OPTIONS -l\" " >>~/.bashrc
RUN echo "alias l=\"ls \$LS_OPTIONS -lA \" " >>~/.bashrc
RUN echo "/etc/init.d/cups restart " >> ~/.bashrc
RUN echo "/usr/sbin/lpadmin -p mitprint -E -v lpd://romanog@mitprint.mit.edu/bw -P /etc/cups/ppd/mitprint.ppd " >> ~/.bashrc

RUN echo "/usr/sbin/lpadmin -p office -P /etc/cups/ppd/office.ppd -E -v socket://18.38.0.23:9100" >> ~/.bashrc
RUN echo "/usr/sbin/lpadmin -d office" >> ~/.bashrc


RUN wget "https://drive.google.com/uc?export=download&id=0Bx6uy3hgjYCEVTdCVEwybXVBOG1KQmFQTWVmeEZWQmctVnJv" -O /etc/cups/ppd/office.ppd
RUN wget "https://drive.google.com/uc?export=download&id=0Bx6uy3hgjYCEWUR5VXJnc1JDQkhQZm9wLXVYTGJYOGhCRnBn" -O /etc/cups/ppd/mitprint.ppd

WORKDIR /app/OpenBTE

CMD /bin/bash




