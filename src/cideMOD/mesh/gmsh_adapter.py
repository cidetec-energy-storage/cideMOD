#
# Copyright (c) 2021 CIDETEC Energy Storage.
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
import multiphenicsx.mesh as mpx
from mpi4py import MPI
from dolfinx.common import Timer
from ufl import Measure

import json
import os
import shutil
from pathlib import Path

import appdirs
import meshio
import numpy as np

from cideMOD.mesh.base_mesher import BaseMesher
from cideMOD.mesh.gmsh_generator import GmshGenerator
# from cideMOD.mesh.restrictions_functions import (
#     _boundary_restriction,
#     _generate_interface_mesh_function,
#     _interface_restriction,
#     _subdomain_restriction,
# )

dir_path = Path(appdirs.user_data_dir('cideMOD',False))
os.makedirs(os.path.join(dir_path,'meshes','templates'), exist_ok=True)
os.makedirs(os.path.join(dir_path,'meshes','current'), exist_ok=True)

class GmshConverter(BaseMesher):
    def _mesh_template(self, mtype, filename):
        return os.path.join(dir_path,'meshes','templates',mtype,filename)

    def _mesh_store(self, filename):
        return os.path.join(dir_path,'meshes','current',filename)

    def prepare_parameters(self, mtype = None):
        parameters = {
            'mode': self.mode,
            'structure': self.cell.structure 
        }
        if mtype:
            parameters['template'] = mtype
        if int(self.mode[1])-1>1:
            parameters['cell_height'] = self.cell.separator.height
        if int(self.mode[1])-1>2:
            parameters['cell_widht'] = self.cell.separator.width
        if 'a' in self.cell.structure:
            parameters['anode_lenght'] = self.cell.negative_electrode.thickness
        if 'c' in self.cell.structure:
            parameters['cathode_lenght'] = self.cell.positive_electrode.thickness
        if 's' in self.cell.structure:
            parameters['separator_lenght'] = self.cell.separator.thickness
        if 'pcc' in self.cell.structure:
            parameters['positiveCC_lenght'] = self.cell.positive_curent_colector.thickness
        if 'ncc' in self.cell.structure:
            parameters['negativeCC_lenght'] = self.cell.negative_curent_colector.thickness
            parameters['x_nodes'] = self.options.N_x
        if self.options.mode in ('P3D', 'P4D'):
            parameters['y_nodes'] = self.options.N_y
        if self.options.mode in ('P4D',):
            parameters['z_nodes'] = self.options.N_z
        
        return parameters
        
    def mesh_updated(self, current_params):
        # Check if mesh exists
        mesh_files = [
            self._mesh_store('log'),
            self._mesh_store('mesh.xml'),
            self._mesh_store('mesh_facet_region.xml'),
            self._mesh_store('mesh_physical_region.xml'),
            self._mesh_store('mesh_interface_region.xml')
        ]
        for mf in mesh_files:
            if not os.path.exists(mf):
                return False
        # Check if mesh is the same
        log_file = self._mesh_store('log')
        if os.path.exists(log_file):
            with open(log_file,'r') as fin:
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

    def build_mesh(self, comm, scale = 1, tab_geometry=None):
        if not self.mesh_updated(self.prepare_parameters()):
            self.prepare_mesh(scale)
        comm.barrier()
        t = Timer('Load Mesh'); t.start()
        self.mesh, cell_tags, facet_tags = self.read_gmsh(comm)
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
        self.anode = (self.dimension, self.subdomains.indices[self.subdomains.values==self.field_data['anode']])
        self.separator = (self.dimension, self.subdomains.indices[self.subdomains.values==self.field_data['separator']])
        self.cathode = (self.dimension, self.subdomains.indices[self.subdomains.values==self.field_data['cathode']])
        self.positiveCC = (self.dimension, self.subdomains.indices[self.subdomains.values==self.field_data.get('positiveCC')])
        self.negativeCC = (self.dimension, self.subdomains.indices[self.subdomains.values==self.field_data.get('negativeCC')])
        self.field_restrictions = {
            'anode':self.anode, 'separator':self.separator, 'cathode':self.cathode, 'positiveCC':self.positiveCC, 'negativeCC':self.negativeCC
        }
        self.electrodes = (self.dimension, np.unique(np.concatenate([self.anode[1], self.cathode[1]])))
        self.electrolyte = (self.dimension, np.unique(np.concatenate([self.anode[1], self.separator[1] ,self.cathode[1]])))
        # self.solid_conductor = self._subdomain_restriction(['anode','cathode', 'positiveCC', 'negativeCC']) #This is not being used right now
        self.current_colectors = (self.dimension, np.unique(np.concatenate([self.positiveCC[1], self.negativeCC[1] ])))
        anode_cc_interface = self.boundaries.indices[self.boundaries.values==self.field_data.get('anode-CC')]
        cathode_cc_interface = self.boundaries.indices[self.boundaries.values==self.field_data.get('cathode-CC')]
        self.electrode_cc_interfaces = (self.boundaries.dim, np.unique(np.concatenate([ anode_cc_interface, cathode_cc_interface ])))
        self.positive_tab = (self.boundaries.dim, self.boundaries.indices[self.boundaries.values==self.field_data['positivePlug']])

        # Generate measures
        a_s_c_order = all([self.structure[i+1]=='s' for i, el in enumerate(self.structure) if el == 'a'])
        def int_dir(default_dir="+"):
            assert default_dir in ("+","-")
            reversed_dir = "-" if default_dir == "+" else "-"
            return default_dir if a_s_c_order else reversed_dir
        meta = {"quadrature_degree":2}
        self.dx = Measure('dx', domain=self.mesh, subdomain_data=self.subdomains, metadata={"quadrature_degree":2})
        self.dx_a = self.dx(subdomain_id=self.field_data.get('anode',999))
        self.dx_s = self.dx(subdomain_id=self.field_data.get('separator',999))
        self.dx_c = self.dx(subdomain_id=self.field_data.get('cathode',999))
        self.dx_pcc = self.dx(self.field_data.get('positiveCC',999))
        self.dx_ncc = self.dx(self.field_data.get('negativeCC',999))
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundaries, metadata={"quadrature_degree":2})
        self.ds_a = self.ds(self.field_data['negativePlug'])
        self.ds_c = self.ds(self.field_data['positivePlug'])
        self.dS = Measure('dS', domain=self.mesh, subdomain_data=self.boundaries, metadata=meta)
        self.dS_as = self.dS(self.field_data['anode-separator'], metadata={**meta, "direction": int_dir("+")})
        self.dS_sa = self.dS(self.field_data['anode-separator'], metadata={**meta, "direction": int_dir("-")})
        self.dS_sc = self.dS(self.field_data['cathode-separator'], metadata={**meta, "direction": int_dir("+")})
        self.dS_cs = self.dS(self.field_data['cathode-separator'], metadata={**meta, "direction": int_dir("-")})
        self.dS_a_cc = self.dS(self.field_data.get('anode-CC',999), metadata={**meta, "direction": int_dir("-")})
        self.dS_cc_a = self.dS(self.field_data.get('anode-CC',999), metadata={**meta, "direction": int_dir("+")})
        self.dS_c_cc = self.dS(self.field_data.get('cathode-CC',999), metadata={**meta, "direction": int_dir("+")})
        self.dS_cc_c = self.dS(self.field_data.get('cathode-CC',999), metadata={**meta, "direction": int_dir("-")})

        self.calc_area_ratios(scale)
        t.stop()

    def read_gmsh(self, comm)-> typing.Tuple[dfx.mesh.Mesh, dfx.cpp.mesh.MeshTags_int32, dfx.cpp.mesh.MeshTags_int32]:
        with dfx.io.XDMFFile(comm,self._mesh_store("mesh.xdmf"),'r') as file:
            mesh = file.read_mesh()
            mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
            subdomains = file.read_meshtags(mesh, 'subdomains')
            boundaries = file.read_meshtags(mesh, 'boundaries')
        
        field_data = self.get_field_data()
        subdomains, field_data = self.check_subdomains(subdomains, field_data)
        return mesh, subdomains, boundaries

class TemplateMesher(GmshConverter):
    def prepare_mesh(self, mtype:str='standard', **kwargs):
        template_path = self._mesh_template(mtype,'{}.geo'.format(self.mode.lower()))
        parameters = self.prepare_parameters(mtype)
        assert os.path.exists(template_path), "Cannot find template mesh in '{}'".format(template_path)
        if not self.mesh_updated(parameters):
            gm = GmshGenerator()
            gm.generate_mesh_from_template(template_path, self._mesh_store('mesh.msh'), dim=int(self.mode[1])-1, parameters=parameters)
            with open(self._mesh_store('log'), 'w') as fout:
                json.dump(parameters, fout)


class GmshMesher(GmshConverter):
    def prepare_mesh(self, scale=1):
        parameters = self.prepare_parameters()
        is_updated = self.mesh_updated(parameters)
        if not is_updated:
            self.clean_mesh_files()
            self.generate_gmsh_mesh(scale)
            with open(self._mesh_store('log'), 'w') as fout:
                json.dump(parameters, fout)
        return not is_updated

    def generate_gmsh_mesh(self, scale=1):
        filename = self._mesh_store('mesh.msh')
        N_x = self.options.N_x
        N_y = self.options.N_y
        N_z = self.options.N_z
        gm = GmshGenerator()
        L, H, W = self.get_dims(scale)
        H = [h for h in H if h]
        W = [w for w in W if w]
        if self.mode == 'P2D':
            gm.create_1D_mesh(filename=filename,structure=self.structure, L = L)
        elif self.mode == 'P3D':
            H=min(H)
            gm.create_2D_mesh(filename=filename,structure=self.structure, H=H,nH = N_y, L=L)
        elif self.mode == 'P4D':
            H=min(H)
            W=min(W)
            gm.create_3D_mesh_with_tabs(filename=filename,structure=self.structure, H=H, Z=W, nH = N_y, nZ=N_z, L=L)