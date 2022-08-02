#!/bin/bash

export TZ=Etc/UTC
ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

apt-get update
apt-get -y -qq install software-properties-common
add-apt-repository -y ppa:fenics-packages/fenics
apt-get update
apt-get -y -q install fenics
apt-get -y -q install \
        libglu1 \
        libxcursor-dev \
        libxft2 \
        libxinerama1 \
        libfltk1.3-dev \
        libfreetype6-dev  \
        libgl1-mesa-dev \
        libocct-foundation-dev \
        libocct-data-exchange-dev && \
        apt-get clean

apt-get -y -q install python3-pip
python3 -m pip install gmsh
python3 -m pip install https://github.com/multiphenics/multiphenics/archive/master.tar.gz

apt-get -y install git
git clone https://github.com/cidetec-energy-storage/cideMOD.git