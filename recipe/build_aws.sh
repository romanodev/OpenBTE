#/usr/bin/bash

#Amazon Machine Image: Ubuntu Server 18.04 LTS (HVM), SSD Volume Type


sudo apt-get update
sudo apt-get install -y vim  build-essential libopenmpi-dev swig libsuitesparse-dev
sudo apt install python3-pip
sudo pip3 install openbte
sudo sh -c '  echo "btl_base_warn_component_unused=0" >>  /etc/openmpi/openmpi-mca-params.conf '
sudo apt-get install -y libglu1-mesa libxcursor1 libxft-dev libxinerama-dev


wget http://geuz.org/gmsh/bin/Linux/gmsh-3.0.0-Linux64.tgz
tar -xzf gmsh-3.0.0-Linux64.tgz
sudo cp gmsh-3.0.0-Linux/bin/gmsh /usr/bin/
rm -rf gmsh-3.0.0-Linux
rm gmsh-3.0.0-Linux64.tgz
