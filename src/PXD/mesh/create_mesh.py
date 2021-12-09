#!/usr/bin/env python3
#
# Copyright (c) 2021 CIDETEC Energy Storage.
#
# This file is part of PXD.
#
# PXD is free software: you can redistribute it and/or modify
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
# along with this program. If not, see <http://www.gnu.org/licenses/>.#
import argparse
from pathlib import Path

from mpi4py import MPI

from PXD.helpers.config_parser import CellParser
from PXD.mesh.gmsh_adapter import GmshMesher

assert MPI.COMM_WORLD.size == 1, "Mesh cannot be created in parallel"

def create_mesh(file_path, mode, x,y,z):
    params = Path(file_path)
    options = {
        'mode': mode,
        'N_x': x,
        'N_y': y,
        'N_z': z,
    }

    cell = CellParser(str(params.absolute()),str(params.parent.absolute()), log=False)
    mesher = GmshMesher(options, cell)
    created = mesher.prepare_mesh()
    if created:
        print('New Mesh created succesfully.', flush=True)
    else:
        print('Using cached mesh.', flush=True)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Generate battery geometry and mesh. Should allways be run in serial')
    parser.add_argument('params_file', type=str, nargs=1, help='Path to cell params file')
    parser.add_argument('mode', type=str, nargs=1, choices=('P3D', 'P4D'), help='Type of mesh')
    parser.add_argument('N_x', type=int, nargs=1, default=30, help='number of discretization elements in x direction')
    parser.add_argument('N_y', type=int, nargs=1, default=30, help='number of discretization elements in x direction')
    parser.add_argument('N_z', type=int, nargs=1, default=30, help='number of discretization elements in x direction')
    args = parser.parse_args()
    create_mesh(
        file_path=args.params_file[0],
        mode=args.mode[0],
        x=args.N_x[0],
        y=args.N_y[0],
        z=args.N_z[0]
    )
