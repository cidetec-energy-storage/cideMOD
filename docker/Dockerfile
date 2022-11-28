#
# Copyright (c) 2022 CIDETEC Energy Storage.
#
# This file is part of cideMOD.
#
# cideMOD is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

FROM multiphenics/multiphenics
LABEL maintainer = "CIDETEC <cidemod@cidetec.es>"

USER fenics
COPY --chown=fenics . /tmp/cideMOD

USER root
WORKDIR /tmp/cideMOD
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
RUN mkdir cideMOD && \
    ln -s $FENICS_PREFIX/lib/python3.6/dist-packages/cideMOD*egg/cideMOD cideMOD/src && \
    mv /tmp/cideMOD/tests cideMOD/ && \
    mv /tmp/cideMOD/examples cideMOD && \
    mv /tmp/cideMOD/data cideMOD && \
    mv /tmp/cideMOD/docs cideMOD

USER root
RUN rm -rf /tmp/cideMOD && \
    echo export PYTHONPATH=$PYTHONPATH:$(find /usr/local/lib -name "gmsh-*-sdk")/lib >>/root/.bashrc && \
    echo export PYTHONPATH=$PYTHONPATH:$(find /usr/local/lib -name "gmsh-*-sdk")/lib >>/home/fenics/.bashrc
