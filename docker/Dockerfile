# Copyright (C) 2016-2021 by the multiphenics authors
#
# This file is part of PXD.
#
# PXD is free software: you can redistribute it and/or modify
# it under the terms of the Affero GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PXD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Affero GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PXD. If not, see <http://www.gnu.org/licenses/>.
#

FROM multiphenics/multiphenics
MAINTAINER Raul Ciria <rciria@cidetec.es>

USER fenics
COPY --chown=fenics . /tmp/PXD

USER root
WORKDIR /tmp/PXD
RUN pip install -qU pip && \
    pip install -q . && \
    pip cache purge

USER root
RUN apt-get -qq update && \
    apt-get -qq --no-install-recommends install libglu1-mesa-dev libxcursor-dev libxinerama-dev && \
    apt-get -qq autoremove -y && \
    apt-get clean


USER fenics
WORKDIR $FENICS_HOME
RUN mkdir PXD && \
    ln -s $FENICS_PREFIX/lib/python3.6/dist-packages/PXD*egg/PXD PXD/src && \
    mv /tmp/PXD/tests PXD/ && \
    mv /tmp/PXD/examples PXD && \
    mv /tmp/PXD/data PXD

USER root
RUN rm -rf /tmp/PXD && \
    echo export PYTHONPATH=$PYTHONPATH:$(find /usr/local/lib -name "gmsh-*-sdk")/lib >>/root/.bashrc && \
    echo export PYTHONPATH=$PYTHONPATH:$(find /usr/local/lib -name "gmsh-*-sdk")/lib >>/home/fenics/.bashrc
