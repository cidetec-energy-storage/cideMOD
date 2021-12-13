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
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
from dolfin import *
from multiphenics import *

import json
import os
import shutil
from pathlib import Path

import appdirs
import meshio
from dolfin_utils.meshconvert.meshconvert import convert2xml

from PXD.mesh.base_mesher import BaseMesher
from PXD.mesh.gmsh_generator import GmshGenerator
from PXD.mesh.restrictions_functions import (
    _boundary_restriction,
    _generate_interface_mesh_function,
    _interface_restriction,
    _subdomain_restriction,
)

dir_path = Path(appdirs.user_data_dir('PXD',False))
os.makedirs(os.path.join(dir_path,'meshes','templates'), exist_ok=True)
os.makedirs(os.path.join(dir_path,'meshes','current'), exist_ok=True)
comm = MPI.comm_world

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
        field_data['interfaces'] = {
            'anode-separator': 1,
            'cathode-separator': 2,
            'anode-CC': 3,
            'cathode-CC': 4
        }
        return field_data

    def build_mesh(self, scale = 1, tab_geometry=None):
        if not self.mesh_updated(self.prepare_parameters()): 
            if MPI.size(comm)==1:
                self.prepare_mesh()
                # self.cell.write_param_file(self._mesh_store("params.json"))
                # command = "python3 {path}/create_mesh.py {params} {mode} {Nx} {Ny} {Nz}".format(
                #     path = dir_path,
                #     params = self._mesh_store("params.json"),
                #     mode = self.options.mode,
                #     Nx = self.options.N_x,
                #     Ny = self.options.N_y,
                #     Nz = self.options.N_z
                # )
                # os.system(command)
            else:
                raise Exception("Mesh not ready, run create_mesh script first")
        t = Timer('Load Mesh')
        
        self.mesh = Mesh(comm, self._mesh_store("mesh.xml"))
        self.subdomains = MeshFunction("size_t", self.mesh, self._mesh_store(f"mesh_physical_region.xml"))
        self.boundaries = MeshFunction("size_t", self.mesh, self._mesh_store(f"mesh_facet_region.xml"))
        self.interfaces = MeshFunction("size_t", self.mesh, self._mesh_store(f"mesh_interface_region.xml"))

        # Get field data
        self.field_data = self.get_field_data()
        self.field_data['anode'] = 11
        self.field_data['cathode'] = 12
        # Scale mesh and get dimensions
        self.mesh.scale(1/scale)
        self.dimension = self.mesh.topology().dim()
        # Load restrictions
        ext = 'xdmf' # "xdmf" or "xml"
        self.anode = MeshRestriction(self.mesh, self._mesh_store(f'restrictions/anode.rtc.{ext}'))
        self.separator = MeshRestriction(self.mesh, self._mesh_store(f'restrictions/separator.rtc.{ext}'))
        self.cathode = MeshRestriction(self.mesh, self._mesh_store(f'restrictions/cathode.rtc.{ext}'))
        self.positiveCC = MeshRestriction(self.mesh, self._mesh_store(f'restrictions/positiveCC.rtc.{ext}'))
        self.negativeCC = MeshRestriction(self.mesh, self._mesh_store(f'restrictions/negativeCC.rtc.{ext}'))
        self.field_restrictions = {
            'anode':self.anode, 'separator':self.separator, 'cathode':self.cathode, 'positiveCC':self.positiveCC, 'negativeCC':self.negativeCC
        }
        self.electrodes = MeshRestriction(self.mesh, self._mesh_store(f'restrictions/electrodes.rtc.{ext}'))
        self.electrolyte = MeshRestriction(self.mesh, self._mesh_store(f'restrictions/electrolyte.rtc.{ext}'))
        # self.solid_conductor = self._subdomain_restriction(['anode','cathode', 'positiveCC', 'negativeCC']) #This is not being used right now
        self.current_colectors = MeshRestriction(self.mesh, self._mesh_store(f'restrictions/current_colectors.rtc.{ext}'))
        self.electrode_cc_interfaces = MeshRestriction(self.mesh, self._mesh_store(f'restrictions/electrode_cc_interfaces.rtc.{ext}'))
        self.positive_tab = MeshRestriction(self.mesh, self._mesh_store(f'restrictions/positive_tab.rtc.{ext}'))

        # Generate measures
        a_s_c_order = all([self.structure[i+1]=='s' for i, el in enumerate(self.structure) if el is 'a'])
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
        self.dS = Measure('dS', domain=self.mesh, subdomain_data=self.interfaces, metadata=meta)
        self.dS_as = self.dS(self.field_data['interfaces']['anode-separator'], metadata={**meta, "direction": int_dir("+")})
        self.dS_sa = self.dS(self.field_data['interfaces']['anode-separator'], metadata={**meta, "direction": int_dir("-")})
        self.dS_sc = self.dS(self.field_data['interfaces']['cathode-separator'], metadata={**meta, "direction": int_dir("+")})
        self.dS_cs = self.dS(self.field_data['interfaces']['cathode-separator'], metadata={**meta, "direction": int_dir("-")})
        self.dS_a_cc = self.dS(self.field_data['interfaces']['anode-CC'], metadata={**meta, "direction": int_dir("-")})
        self.dS_cc_a = self.dS(self.field_data['interfaces']['anode-CC'], metadata={**meta, "direction": int_dir("+")})
        self.dS_c_cc = self.dS(self.field_data['interfaces']['cathode-CC'], metadata={**meta, "direction": int_dir("+")})
        self.dS_cc_c = self.dS(self.field_data['interfaces']['cathode-CC'], metadata={**meta, "direction": int_dir("-")})

        self.calc_area_ratios(scale)
        t.stop()

    def gmsh_convert(self):
        os.rename(self._mesh_store('mesh.msh2'), self._mesh_store('mesh.msh'))
        convert2xml(self._mesh_store('mesh.msh'), self._mesh_store('mesh.xml'))

        # Read old-style xml files
        mesh = Mesh(self._mesh_store("mesh.xml"))
        subdomains = MeshFunction("size_t", mesh, self._mesh_store("mesh_physical_region.xml"))
        boundaries = MeshFunction("size_t", mesh, self._mesh_store("mesh_facet_region.xml"))
        field_data = self.get_field_data()
        subdomains, field_data = self.check_subdomains(subdomains, field_data)
        interfaces = _generate_interface_mesh_function(mesh, subdomains, field_data)
        
        t0 = Timer('Gen restrictions')
        # Generate restrictions
        anode = _subdomain_restriction('anode', mesh, subdomains, field_data) 
        separator = _subdomain_restriction('separator', mesh, subdomains, field_data) # Only used for processing
        cathode = _subdomain_restriction('cathode', mesh, subdomains, field_data)
        positiveCC = _subdomain_restriction('positiveCC', mesh, subdomains, field_data)
        negativeCC = _subdomain_restriction('negativeCC', mesh, subdomains, field_data)
        electrodes = _subdomain_restriction([anode, cathode], mesh, subdomains, field_data)
        electrolyte = _subdomain_restriction([anode, separator, cathode], mesh, subdomains, field_data)
        current_colectors = _subdomain_restriction([positiveCC, negativeCC], mesh, subdomains, field_data)
        # solid_conductor = _subdomain_restriction([electrodes, current_colectors]) #This is not being used right now
        electrode_cc_interfaces = _interface_restriction([ ['anode','negativeCC'], ['cathode', 'positiveCC'] ], mesh, subdomains, field_data)
        positive_tab = _boundary_restriction(['positivePlug'], mesh, boundaries, field_data)
        t0.stop()

        t1 = Timer('Write XDMF mesh files')
        # Mesh and subdomains are written with xml
        # XDMFFile(self._mesh_store("mesh.xdmf")).write(mesh)
        # XDMFFile(self._mesh_store("mesh_physical_region.xdmf")).write(subdomains)
        # XDMFFile(self._mesh_store("mesh_facet_region.xdmf")).write(boundaries)
        # XDMFFile(self._mesh_store("mesh_interface_region.xdmf")).write(interfaces)

        # Restriction visualization (for debug)
        XDMFFile(self._mesh_store("restrictions/anode.rtc.xdmf")).write(anode)
        XDMFFile(self._mesh_store("restrictions/separator.rtc.xdmf")).write(separator)
        XDMFFile(self._mesh_store("restrictions/cathode.rtc.xdmf")).write(cathode)
        XDMFFile(self._mesh_store("restrictions/positiveCC.rtc.xdmf")).write(positiveCC)
        XDMFFile(self._mesh_store("restrictions/negativeCC.rtc.xdmf")).write(negativeCC)
        XDMFFile(self._mesh_store("restrictions/electrodes.rtc.xdmf")).write(electrodes)
        XDMFFile(self._mesh_store("restrictions/electrolyte.rtc.xdmf")).write(electrolyte)
        XDMFFile(self._mesh_store("restrictions/current_colectors.rtc.xdmf")).write(current_colectors)
        XDMFFile(self._mesh_store("restrictions/electrode_cc_interfaces.rtc.xdmf")).write(electrode_cc_interfaces)
        XDMFFile(self._mesh_store("restrictions/positive_tab.rtc.xdmf")).write(positive_tab)
        t1.stop()
        t = Timer('Write XML mesh files')
        # Write out new-style xml files
        File(self._mesh_store("mesh.xml")) << mesh
        File(self._mesh_store("mesh_physical_region.xml")) << subdomains
        File(self._mesh_store("mesh_facet_region.xml")) << boundaries
        File(self._mesh_store("mesh_interface_region.xml")) << interfaces

        # File(self._mesh_store("restrictions/anode.rtc.xml")) << anode
        # File(self._mesh_store("restrictions/cathode.rtc.xml")) << cathode
        # File(self._mesh_store("restrictions/electrodes.rtc.xml")) << electrodes
        # File(self._mesh_store("restrictions/electrolyte.rtc.xml")) << electrolyte
        # File(self._mesh_store("restrictions/current_colectors.rtc.xml")) << current_colectors
        # File(self._mesh_store("restrictions/electrode_cc_interfaces.rtc.xml")) << electrode_cc_interfaces
        # File(self._mesh_store("restrictions/positive_tab.rtc.xml")) << positive_tab
        t.stop()

class TemplateMesher(GmshConverter):

    def prepare_mesh(self, mtype:str='standard'):
        template_path = self._mesh_template(mtype,'{}.geo'.format(self.mode.lower()))
        parameters = self.prepare_parameters(mtype)
        assert os.path.exists(template_path), "Cannot find template mesh in '{}'".format(template_path)
        if not self.mesh_updated(parameters):
            gm = GmshGenerator()
            gm.generate_mesh_from_template(template_path, self._mesh_store('mesh.msh2'), dim=int(self.mode[1])-1, parameters=parameters)
            self.gmsh_convert()
            with open(self._mesh_store('log'), 'w') as fout:
                json.dump(parameters, fout)


class GmshMesher(GmshConverter):
    def prepare_mesh(self):
        # Note: This should be run allways in non-parallel model
        parameters = self.prepare_parameters()
        is_updated = self.mesh_updated(parameters)
        if not is_updated:
            self.clean_mesh_files()
            self.generate_gmsh_mesh() 
            self.gmsh_convert()
            with open(self._mesh_store('log'), 'w') as fout:
                json.dump(parameters, fout)
        return not is_updated

    def generate_gmsh_mesh(self):
        filename = self._mesh_store('mesh.msh2')
        N_x = self.options.N_x
        N_y = self.options.N_y
        N_z = self.options.N_z
        gm = GmshGenerator()
        L, H, W = self.get_dims()
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
        