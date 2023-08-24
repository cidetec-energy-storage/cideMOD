#
# Copyright (c) 2023 CIDETEC Energy Storage.
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

import typing
import dolfinx as dfx
from mpi4py import MPI
from dolfinx.common import Timer
from ufl import Measure, FacetNormal

import json
import os
import shutil
from pathlib import Path
from collections import namedtuple

import appdirs
import meshio
import numpy as np

from cideMOD.mesh.base_mesher import BaseMesher
from cideMOD.mesh.gmsh_generator import GmshGenerator

dir_path = Path(appdirs.user_data_dir('cideMOD', False))
os.makedirs(os.path.join(dir_path, 'meshes', 'templates'), exist_ok=True)
os.makedirs(os.path.join(dir_path, 'meshes', 'current'), exist_ok=True)


class GmshConverter(BaseMesher):
    def _mesh_template(self, mtype, filename):
        return os.path.join(dir_path, 'meshes', 'templates', mtype, filename)

    def _mesh_store(self, filename):
        return os.path.join(dir_path, 'meshes', 'current', filename)

    def prepare_parameters(self, mtype=None):
        parameters = {
            'model': self.model,
            'structure': self.cell.structure,
            'x_nodes': self.options.N_x
        }
        if mtype:
            parameters['template'] = mtype
        if int(self.model[1]) - 1 > 1:
            parameters['cell_height'] = self.cell.separator.height.get_reference_value()
        if int(self.model[1]) - 1 > 2:
            parameters['cell_width'] = self.cell.separator.width.get_reference_value()
        if 'a' in self.cell.structure:
            parameters['anode_length'] = self.cell.anode.thickness.get_reference_value()
        if 'c' in self.cell.structure:
            parameters['cathode_length'] = self.cell.cathode.thickness.get_reference_value()
        if 's' in self.cell.structure:
            parameters['separator_length'] = self.cell.separator.thickness.get_reference_value()
        if 'pcc' in self.cell.structure:
            parameters['positiveCC_length'] = self.cell.positiveCC.thickness.get_reference_value()
        if 'ncc' in self.cell.structure:
            parameters['negativeCC_length'] = self.cell.negativeCC.thickness.get_reference_value()
        if self.options.model in ('P3D', 'P4D'):
            parameters['y_nodes'] = self.options.N_y
        if self.options.model in ('P4D',):
            parameters['z_nodes'] = self.options.N_z

        return parameters

    def mesh_updated(self, current_params):
        """Checks if the mesh is up to date"""
        # Check if mesh exists
        mesh_files = [
            self._mesh_store('log'),
            self._mesh_store('mesh.xdmf'),
            self._mesh_store('mesh.h5'),
            self._mesh_store('mesh.msh')
        ]
        for mf in mesh_files:
            if not os.path.exists(mf):
                return False
        # Check if mesh is the same
        log_file = self._mesh_store('log')
        if os.path.exists(log_file):
            with open(log_file, 'r') as fin:
                last_params = json.load(fin)
            return last_params == current_params
        else:
            return True

    def clean_mesh_files(self):
        dirs = os.listdir(self._mesh_store(''))
        for f in dirs:
            if os.path.isdir(self._mesh_store(f)):
                shutil.rmtree(self._mesh_store(f))
            elif not f.endswith('.json'):
                os.remove(self._mesh_store(f))

    def get_field_data(self):
        msh = meshio.read(self._mesh_store('mesh.msh'))
        field_data = {key: int(item[0]) for key, item in msh.field_data.items()}
        return field_data

    def build_mesh(self, scale=1, tab_geometry=None, dimless_model=False):
        if not self.mesh_updated(self.prepare_parameters()):
            self.prepare_mesh(scale, dimless_model)
        self._comm.barrier()
        t = Timer('Load Mesh')
        t.start()
        self.mesh, cell_tags, facet_tags = self.read_gmsh()
        self.subdomains = cell_tags
        self.boundaries = facet_tags
        self.interfaces = facet_tags

        # Get field data
        self.field_data = self.get_field_data()
        self.field_data['anode'] = 21
        self.field_data['cathode'] = 22
        # Get mesh dimensions
        self.dimension = self.mesh.topology.dim

        # Load restrictions
        facet_dim = self.interfaces.dim

        def _locate_entities(subdomains, meshtags):
            return  meshtags.find(self.field_data.get(subdomains))

        self.anode = (self.dimension, _locate_entities('anode', self.subdomains))
        self.separator = (self.dimension, _locate_entities('separator', self.subdomains))
        self.cathode = (self.dimension, _locate_entities('cathode', self.subdomains))
        self.positiveCC = (self.dimension, _locate_entities('positiveCC', self.subdomains))
        self.negativeCC = (self.dimension, _locate_entities('negativeCC', self.subdomains))

        self.field_restrictions = {
            'anode': self.anode,
            'separator': self.separator,
            'cathode': self.cathode,
            'positiveCC': self.positiveCC,
            'negativeCC': self.negativeCC
        }
        self.electrodes = (
            self.dimension,
            np.unique(np.concatenate([self.anode[1], self.cathode[1]]))
        )
        self.electrolyte = (
            self.dimension,
            np.unique(np.concatenate([self.anode[1], self.separator[1], self.cathode[1]]))
        )
        self.current_collectors = (
            self.dimension,
            np.unique(np.concatenate([self.positiveCC[1], self.negativeCC[1]]))
        )
        self.solid_conductor = (
            self.dimension,
            np.unique(np.concatenate([self.electrodes[1], self.current_collectors[1]]))
        )
        self.anode_CC_facets = (facet_dim, _locate_entities('anode-CC', self.interfaces))
        self.cathode_CC_facets = (facet_dim, _locate_entities('cathode-CC', self.interfaces))
        self.electrode_CC_facets = (
            facet_dim,
            np.unique(np.concatenate([self.anode_CC_facets[1], self.cathode_CC_facets[1]]))
        )
        self.positive_tab = (facet_dim, _locate_entities('positive_plug', self.boundaries))
        self.negative_tab = (facet_dim, _locate_entities('negative_plug', self.boundaries))
        self.tabs = (
            facet_dim, np.unique(np.concatenate([self.positive_tab[1], self.negative_tab[1]]))
        )
        if self.model in ('P3D', 'P4D'):
            self.Y_m_surface = (self.boundaries.dim, _locate_entities('Y_m', self.boundaries))

        # Generate measures
        a_s_c_order = all([self.structure[i + 1] == 's'
                           for i, el in enumerate(self.structure) if el == 'a'])

        def int_dir(default_dir="+"):
            assert default_dir in ("+", "-")
            reversed_dir = "-" if default_dir == "+" else "+"
            return default_dir if a_s_c_order else reversed_dir
        meta = {"quadrature_degree": 2}
        self.dx = Measure('dx', domain=self.mesh, subdomain_data=self.subdomains, metadata=meta)
        self.dx_a = self.dx(subdomain_id=self.field_data.get('anode', 999))
        self.dx_s = self.dx(subdomain_id=self.field_data.get('separator', 999))
        self.dx_c = self.dx(subdomain_id=self.field_data.get('cathode', 999))
        self.dx_pcc = self.dx(self.field_data.get('positiveCC', 999))
        self.dx_ncc = self.dx(self.field_data.get('negativeCC', 999))
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundaries, metadata=meta)
        self.ds_a = self.ds(self.field_data['negative_plug'])
        self.ds_c = self.ds(self.field_data['positive_plug'])
        if self.model in ('P3D', 'P4D'):
            self.ds_Ym = self.ds(self.field_data['Y_m'])
        self.dS = Measure('dS', domain=self.mesh, subdomain_data=self.interfaces, metadata=meta)
        self.dS_as = self.dS(
            self.field_data['anode-separator'], metadata={**meta, "direction": int_dir("+")})
        self.dS_sa = self.dS(
            self.field_data['anode-separator'], metadata={**meta, "direction": int_dir("-")})
        self.dS_sc = self.dS(
            self.field_data['cathode-separator'], metadata={**meta, "direction": int_dir("+")})
        self.dS_cs = self.dS(
            self.field_data['cathode-separator'], metadata={**meta, "direction": int_dir("-")})
        self.dS_a_cc = self.dS(
            self.field_data.get('anode-CC', 999), metadata={**meta, "direction": int_dir("-")})
        self.dS_cc_a = self.dS(
            self.field_data.get('anode-CC', 999), metadata={**meta, "direction": int_dir("+")})
        self.dS_c_cc = self.dS(
            self.field_data.get('cathode-CC', 999), metadata={**meta, "direction": int_dir("+")})
        self.dS_cc_c = self.dS(
            self.field_data.get('cathode-CC', 999), metadata={**meta, "direction": int_dir("-")})

        self._compute_volumes()

        # Facet normal directions
        self.normal = FacetNormal(self.mesh)
        self.f_to_c = self.mesh.topology.connectivity(self.dimension - 1, self.dimension)
        t.stop()

    def read_gmsh(self) -> typing.Tuple[dfx.mesh.Mesh,
                                        dfx.cpp.mesh.MeshTags_int32,
                                        dfx.cpp.mesh.MeshTags_int32]:
        with dfx.io.XDMFFile(self._comm, self._mesh_store("mesh.xdmf"), 'r') as file:
            mesh = file.read_mesh()
            # Compute connectivity
            mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
            cell_tags = file.read_meshtags(mesh, 'Cell tags')
            facet_tags = file.read_meshtags(mesh, 'Facet tags')

        field_data = self.get_field_data()
        cell_tags, field_data = self.check_subdomains(cell_tags, field_data)
        return mesh, cell_tags, facet_tags

    def get_measures(self):
        if self._measures is not None:
            return self._measures

        # Measure labels
        dx_measures = ['x', 'x_a', 'x_s', 'x_c', 'x_pcc', 'x_ncc']
        ds_measures = ['s', 's_a', 's_c']
        dS_measures = [
            'S_a_s', 'S_s_a', 'S_c_s', 'S_s_c',
            'S_a_ncc', 'S_ncc_a', 'S_c_pcc', 'S_pcc_c'
        ]
        if self.model in ('P3D', 'P4D'):
            ds_measures.extend(['s_Ym'])
        measure_labels = dx_measures + ds_measures + dS_measures

        # Measures
        measures = []
        for label in measure_labels:
            integral_type = f'd{label[0]}'
            if integral_type != 'dS':
                measure = getattr(self, f'd{label}')
            elif 'cc' in label:
                ldomain, rdomain = ['cc' if domain in ('pcc', 'ncc') else domain
                                    for domain in label.split('_')[1:]]
                measure = getattr(self, f'{integral_type}_{ldomain}_{rdomain}')
            else:
                ldomain, rdomain = label.split('_')[1:]
                measure = getattr(self, f'{integral_type}_{ldomain}{rdomain}')
            measures.append(measure)

        d = namedtuple('Measures', measure_labels)
        self._measures = d._make(measures)
        return self._measures

class GmshMesher(GmshConverter):
    def prepare_mesh(self, scale=1, dimless_model=False):
        parameters = self.prepare_parameters()
        is_updated = self.mesh_updated(parameters)
        if not is_updated:
            self.clean_mesh_files()
            self.generate_gmsh_mesh(scale, dimless_model)
            with open(self._mesh_store('log'), 'w') as fout:
                json.dump(parameters, fout)
        return not is_updated

    def generate_gmsh_mesh(self, scale, dimless_model):
        if self._comm.rank == 0:
            # NOTE: Generates mesh in rank 0
            filename = self._mesh_store('mesh.msh')
            N_x = self.options.N_x
            N_y = self.options.N_y
            N_z = self.options.N_z
            gm = GmshGenerator(comm=MPI.COMM_SELF, verbose=self.verbose)
            L, H, W = self.get_dims(scale)
            H = [h for h in H if h]
            W = [w for w in W if w]
            if self.model == 'P2D':
                L = L if dimless_model else [1 for _ in L]
                gm.create_1D_mesh(filename=filename, structure=self.structure, L=L, nL=N_x)
            elif self.model == 'P3D':
                L, H = (L, min(H)) if dimless_model else ([1 for _ in L], 1)
                gm.create_2D_mesh(filename=filename, structure=self.structure,
                                H=H, nH=N_y, L=L, nL=N_x)
            elif self.model == 'P4D':
                L, H, W = (L, min(H), min(W)) if dimless_model else ([1 for _ in L], 1, 1)
                gm.create_3D_mesh_with_tabs(filename=filename, structure=self.structure,
                                        H=H, Z=W, nH=N_y, nZ=N_z, L=L, nL=N_x)

        # NOTE: Every rank should be waiting for the mesh generation
        if self._comm.size > 1:
            self._comm.barrier()