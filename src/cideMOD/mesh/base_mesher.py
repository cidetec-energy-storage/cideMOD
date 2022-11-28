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
from dolfin import (
    DOLFIN_EPS,
    BoxMesh,
    CompiledSubDomain,
    Function,
    FunctionSpace,
    IntervalMesh,
    Measure,
    MeshFunction,
    Point,
    RectangleMesh,
    Timer,
    assemble,
    cells,
    Constant
)
from multiphenics import BlockFunctionSpace, MeshRestriction

from collections import namedtuple

from numpy import array, concatenate, ndarray, zeros

from cideMOD.helpers.config_parser import CellParser
from cideMOD.helpers.miscellaneous import inside_element_expression
from cideMOD.models.model_options import ModelOptions


class SubdomainGenerator:
    def __init__(self):
        self.boundary = "on_boundary && near(x[0], ref, tol)"

    def set_domain(self, list):
        domain = inside_element_expression(list)
        return CompiledSubDomain(domain,tol=DOLFIN_EPS)

    def set_boundary(self, ref):
        return CompiledSubDomain(self.boundary, ref=ref, tol=DOLFIN_EPS)

    def set_boundaries(self, a, b):
        boundary = "on_boundary && (near(x[0], a, tol) || near(x[0], b, tol))"
        return CompiledSubDomain(boundary, a=a, b=b, tol=DOLFIN_EPS)

    def set_tab(self, ref, dim:int, initial:bool):
        assert dim > 1, "Can't use tabs in 1D cell"
        assert dim < 4, "Max dimension is 3"
        boundary = " on_boundary && x[0] >= ref-tol && x[0] <= ref+1+tol && near(x[1], 1, tol) "
        if dim == 3:
            boundary += "&& x[2] >= 0.5*ini-tol && x[2] <= 1-0.5*ini+tol"
        return CompiledSubDomain(boundary, ref=ref, tol=DOLFIN_EPS, ini=int(initial))

    def set_interface(self, list):
        domain = ""
        for i_index, e_index in enumerate(list):
            domain += f"(near(x[0], {e_index}, tol))"
            if i_index < (len(list) - 1):
                domain += " || "
        if domain == "":
            domain = "false"
        return CompiledSubDomain(domain, tol=DOLFIN_EPS)

    def solid_conductor(self, structure):
        index_list = [index for index, element in enumerate(structure) if (element == 'a' or element == 'c' or element == 'pcc' or element == 'ncc' or element == 'li')]
        sub_domain = inside_element_expression(index_list)
        return CompiledSubDomain(sub_domain, tol= DOLFIN_EPS)

    def current_collectors(self, structure):
        index_list = [index for index, element in enumerate(structure) if (element == 'li' or element == 'pcc' or element == 'ncc')]
        sub_domain = inside_element_expression(index_list)
        return CompiledSubDomain(sub_domain, tol= DOLFIN_EPS)

    def electrolyte(self, structure):
        index_list = [index for index, element in enumerate(structure) if (element == 'a' or element == 'c' or element == 's')]
        sub_domain = inside_element_expression(index_list)
        return CompiledSubDomain(sub_domain, tol= DOLFIN_EPS)

    def electrodes(self, structure):
        index_list = [index for index, element in enumerate(structure) if (element == 'a' or element == 'c')]
        sub_domain = inside_element_expression(index_list)
        return CompiledSubDomain(sub_domain, tol= DOLFIN_EPS)


def get_local_dofs_on_restriction(V, restriction):
    """
    Computes dofs of W[component] which are on the provided restriction, which can be smaller or equal to the restriction
    provided at construction time of W (or it can be any restriction if W[component] is unrestricted). 
    Returns a list that stores local dof numbering with respect to W[component], e.g. to be used to fetch data
    from FEniCS solution vectors.
    """
    # Prepare an auxiliary block function space, restricted on the boundary
    W_restricted = BlockFunctionSpace([V], restrict=[restriction])
    component_restricted = 0 # there is only one block in the W_restricted space
    # Get list of all local dofs on the restriction, numbered according to W_restricted. This will be a contiguous list
    # [1, 2, ..., # local dofs on the restriction]
    restricted_dofs = W_restricted.block_dofmap().block_owned_dofs__local_numbering(component_restricted)
    # Get the mapping of local dofs numbering from W_restricted[0] to V
    restricted_to_original = W_restricted.block_dofmap().block_to_original(component_restricted)
    # Get list of all local dofs on the restriction, but numbered according to V. Note that this list will not be
    # contiguous anymore, because there are DOFs on V other than the ones in the restriction (i.e., the ones in the
    # interior)
    original_dofs = [restricted_to_original[restricted] for restricted in restricted_dofs]
    return original_dofs

class SubdomainMapper:
    def __init__(self, field_restriction, function_space):
        t = Timer('Build SubdomainMapper')
        self.domain_vertex_map = {}
        self.domain_dof_map = {}
        self.ow_range = function_space.dofmap().ownership_range()
        # print(self.ow_range)
        # for field_name, sd_id in field_data.items():
        #     vertex_stack = []
        #     for c in cells(subdomains.mesh()):
        #         if subdomains[c] == sd_id:
        #             for v_index in c.entities(0):
        #                 vertex_stack.append(v_index)
        #     if len(vertex_stack) > 0:
        #         self.domain_vertex_map[field_name] = array(list(set(vertex_stack)))
        # for field_name, vertex_map in self.domain_vertex_map.items():
        #     dofs = array(function_space.dofmap().entity_dofs(subdomains.mesh(), 0, vertex_map))
        #     self.domain_dof_map[field_name] = dofs[dofs<self.ow_range[1]-self.ow_range[0]]
        for field_name, res in field_restriction.items():
            self.domain_dof_map[field_name] = get_local_dofs_on_restriction(function_space, res)
        self.base_array = zeros(len(function_space.dofmap().dofs()))
        self.base_function = Function(function_space)
        t.stop()

    def generate_vector(self, source_dict:dict):
        out_array = self.base_array.copy()
        for domain_name, source in source_dict.items():
            if domain_name in self.domain_dof_map.keys():
                if isinstance(source,Function):
                    if len(source.vector()) == len(out_array):
                        out_array[self.domain_dof_map[domain_name]] = source.vector()[self.domain_dof_map[domain_name]]
                    elif len(source.vector()) == 1:
                        out_array[self.domain_dof_map[domain_name]] = source.vector()[0]
                    else:
                        raise Exception('Invalid source for domain mapper, number of dofs have to coincide')
                elif isinstance(source, (float, int)):
                    out_array[self.domain_dof_map[domain_name]] = source
                elif isinstance(source, (list,tuple,ndarray)):
                    if len(source) == len(out_array):
                        out_array[self.domain_dof_map[domain_name]] = source[self.domain_dof_map[domain_name]]
                    elif len(source) == len(self.domain_dof_map[domain_name]):
                        out_array[self.domain_dof_map[domain_name]] = source
                    elif len(source) == 1:
                        out_array[self.domain_dof_map[domain_name]] = source[0]
                    else:
                        raise Exception('Invalid source for domain mapper, number of dofs have to coincide')
                else:
                    raise Exception('Invalid source type for domain mapper')
        return out_array

    def generate_function(self, source_dict:dict):
        # TODO: avoid using this deepcopy if possible
        ou_funct = self.base_function.copy(deepcopy=True) 
        vec = ou_funct.vector()
        vec.set_local(self.generate_vector(source_dict))
        vec.apply("insert")
        return ou_funct


class BaseMesher:
    def __init__(self, options:ModelOptions, cell:CellParser):
        self.options = options
        self.cell = cell
        self.mode = self.options.mode
        self.structure = cell.structure
        self.num_components = len(self.structure)
        self.field_data = {}
        
    def get_dims(self):
        domain_dict = {
            'a': self.cell.negative_electrode,
            's': self.cell.separator,
            'c': self.cell.positive_electrode,
            'ncc': self.cell.negative_curent_colector,
            'pcc': self.cell.positive_curent_colector,
        } 
        L = []
        H = []
        W = []
        for element in self.structure:
            data = domain_dict[element]
            L.append(data.thickness)
            H.append(data.height)
            W.append(data.width)
        return L, H, W

    def get_measures(self):
        d = namedtuple('Measures', ['x', 'x_a', 'x_s', 'x_c', 'x_pcc', 'x_ncc', 's', 's_a', 's_c', 'S_a_s', 'S_s_a', 'S_c_s', 'S_s_c', 'S_a_ncc', 'S_ncc_a', 'S_c_pcc', 'S_pcc_c'])
        return d._make([self.dx, self.dx_a, self.dx_s, self.dx_c, self.dx_pcc, self.dx_ncc, self.ds, self.ds_a, self.ds_c, self.dS_as, self.dS_sa, self.dS_cs, self.dS_sc, self.dS_a_cc, self.dS_cc_a, self.dS_c_cc, self.dS_cc_c])

    def get_subdomains_coord(self, P1_map):
        empty = array([],dtype=int)
        coord_list = namedtuple('subdomainCoord', ' '.join(['negativeCC','anode','separator','cathode','positiveCC','electrolyte','solid_conductor','electrodes']))
        negCC = P1_map.domain_dof_map.get('negativeCC', empty)
        anod = P1_map.domain_dof_map.get('anode', empty)
        sep = P1_map.domain_dof_map.get('separator', empty)
        cathod = P1_map.domain_dof_map.get('cathode', empty)
        posCC = P1_map.domain_dof_map.get('positiveCC', empty)
        electrolyte = concatenate( (anod,sep,cathod) )
        solid_conductor = concatenate( (negCC, anod, cathod, posCC) )
        electrodes = concatenate( (anod, cathod) )
        return coord_list._make([negCC, anod, sep, cathod, posCC, electrolyte, solid_conductor, electrodes])

    def check_subdomains(self, subdomains, field_data):
        # DEBUG only - Check subdomains components
        # if MPI.size(MPI.comm_world) == 1:
        #     sd_array = self.subdomains.array().copy()
        #     assert len(set(sd_array)) == len(set(self.structure)), 'Some elements not inside a subdomain'
        #     for i, sd in enumerate(sd_array):
        #         if i == 0:
        #             assert sd == sd_array[i+1], 'A subdomain has only one element'
        #         elif i == len(sd_array)-1:
        #             assert sd == sd_array[i-1], 'A subdomain has only one element'
        #         else:
        #             assert sd in (sd_array[i+1], sd_array[i-1]), 'A subdomain has only one element'
        # Ensure electrode subdomains have higest values
        assert max([val for val in field_data.values() if isinstance(val,int)]) < 10, 'Unusualy high subdomain id'
        subdomains.array()[subdomains.array()==field_data['anode']] = 11
        subdomains.array()[subdomains.array()==field_data['cathode']] = 12
        field_data['anode'] = 11
        field_data['cathode'] = 12
        return subdomains, field_data

    def calc_area_ratios(self, scale):
        if self.options.mode == 'P4D':
            self.area_ratio_a = self.cell.area / (assemble(1*self.ds_a) * scale ** 2)
            self.area_ratio_c = self.cell.area / (assemble(1*self.ds_c) * scale ** 2)
        elif self.options.mode == 'P3D':
            self.area_ratio_a = self.cell.area / (assemble(self.cell.width*self.ds_a) * scale)
            self.area_ratio_c = self.cell.area / (assemble(self.cell.width*self.ds_c) * scale)
        elif self.options.mode == 'P2D':
            self.area_ratio_a = 1
            self.area_ratio_c = 1

    def _compute_volumes(self):
        d = self.get_measures()
        volumes = namedtuple('ScaledVolumes', d._fields)
        self.volumes = volumes._make([assemble(Constant(1)*dx) for dx in d])

class DolfinMesher(BaseMesher):
    def build_mesh(self):
        N_x = self.options.N_x
        N_y = self.options.N_y
        N_z = self.options.N_z
        n = self.num_components
        L, _, _ = self.get_dims()
        self.scale = L
        timer = Timer('Building mesh')
        nodes = n*N_x if self.mode == "P2D" else (n*N_x*(N_y or N_x) if self.mode == "P3D" else n*N_x*(N_y or N_x)*(N_z or N_x))
        print('Building mesh for {} problem with {} components and {} nodes.'.format(self.mode, n, nodes))

        if self.mode == "P4D":
            p1 = Point(0,0,0)
            p2 = Point(n,1,1)
            self.mesh = BoxMesh(p1,p2, N_x*n, N_y or N_x, N_z or N_x)
        elif self.mode == "P3D":
            p1 = Point(0,0,0)
            p2 = Point(n,1,0)
            self.mesh = RectangleMesh(p1,p2, N_x*n, N_y or N_x)

        elif self.mode == "P2D":
            self.mesh = IntervalMesh(N_x*n, 0, n)

        self.dimension = self.mesh.geometric_dimension()

        subdomain_generator = SubdomainGenerator()
        self.field_data['anode'] = 1
        self.field_data['separator'] = 2
        self.field_data['cathode'] = 3
        self.field_data['negativeCC'] = 4
        self.field_data['positiveCC'] = 5
        self.field_data['negativePlug'] = 6
        self.field_data['positivePlug'] = 7
        self.field_data['interfaces'] = {
            'anode-separator': 1,
            'cathode-separator': 2,
            'anode-CC': 3,
            'cathode-CC': 4,
        }

        # Mark boundaries
        boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1, 0)

        if self.structure[-1] in ('c','pcc') or not 'c' in self.structure:
            negativetab = subdomain_generator.set_boundary(0)
            positivetab = subdomain_generator.set_boundary(n)
        else:
            negativetab = subdomain_generator.set_boundary(n)
            positivetab = subdomain_generator.set_boundary(0)
        negativetab.mark(boundaries, self.field_data['negativePlug'])
        positivetab.mark(boundaries, self.field_data['positivePlug'])

        tabs = subdomain_generator.set_boundaries(0, n)

        self.boundaries = boundaries

        #Mark subdomains
        subdomains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim(), 99)

        anode_list = [index for index, element in enumerate(self.structure) if element == 'a']
        cathode_list = [index for index, element in enumerate(self.structure) if element == 'c']
        separator_list = [index for index, element in enumerate(self.structure) if element == 's']
        positive_cc_list = [index for index, element in enumerate(self.structure) if element == 'pcc']
        negative_cc_list = [index for index, element in enumerate(self.structure) if element == 'ncc']

        negative_cc = subdomain_generator.set_domain(negative_cc_list)
        anode = subdomain_generator.set_domain(anode_list)
        separator = subdomain_generator.set_domain(separator_list)
        cathode = subdomain_generator.set_domain(cathode_list)
        positive_cc = subdomain_generator.set_domain(positive_cc_list)

        electrodes = subdomain_generator.electrodes(self.structure)
        electrolyte = subdomain_generator.electrolyte(self.structure)
        solid_conductor = subdomain_generator.solid_conductor(self.structure)
        current_colectors = subdomain_generator.current_collectors(self.structure)

        negative_cc.mark(subdomains, self.field_data['negativeCC'])
        anode.mark(subdomains,self.field_data['anode'])
        separator.mark(subdomains,self.field_data['separator'])
        cathode.mark(subdomains,self.field_data['cathode'])
        positive_cc.mark(subdomains, self.field_data['positiveCC'])

        self.subdomains = subdomains
        self.check_subdomains(self.subdomains, self.field_data)

        #Mark interfaces
        interfaces = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() -1, 0)

        negativeCC_interfaces = set(negative_cc_list).union([i+1 for i in negative_cc_list])
        anode_interfaces = set(anode_list).union([i+1 for i in anode_list])
        separator_interfaces = set(separator_list).union([i+1 for i in separator_list])
        cathode_interfaces = set(cathode_list).union([i+1 for i in cathode_list])
        positiveCC_interfaces = set(positive_cc_list).union([i+1 for i in positive_cc_list])
        anode_separator_interface_list = anode_interfaces.intersection(separator_interfaces)
        anode_CC_interface_list = anode_interfaces.intersection(negativeCC_interfaces)
        cathode_separator_interface_list = cathode_interfaces.intersection(separator_interfaces)
        cathode_CC_interface_list = cathode_interfaces.intersection(positiveCC_interfaces)

        anode_separator_interface = subdomain_generator.set_interface(anode_separator_interface_list)
        cathode_separator_interface = subdomain_generator.set_interface(cathode_separator_interface_list)
        anode_CC_interface = subdomain_generator.set_interface(anode_CC_interface_list)
        cathode_CC_interface = subdomain_generator.set_interface(cathode_CC_interface_list)
        
        anode_separator_interface.mark(interfaces,self.field_data['interfaces']['anode-separator'])
        cathode_separator_interface.mark(interfaces,self.field_data['interfaces']['cathode-separator'])
        anode_CC_interface.mark(interfaces,self.field_data['interfaces']['anode-CC'])
        cathode_CC_interface.mark(interfaces,self.field_data['interfaces']['cathode-CC'])

        self.interfaces = interfaces

        # Restrictions 
        self.anode = MeshRestriction(self.mesh, anode)
        self.separator = MeshRestriction(self.mesh, separator)
        self.cathode = MeshRestriction(self.mesh, cathode)
        self.positiveCC = MeshRestriction(self.mesh, positive_cc)
        self.negativeCC = MeshRestriction(self.mesh, negative_cc)
        self.field_restrictions = {
            'anode':self.anode, 'separator':self.separator, 'cathode':self.cathode, 'positiveCC':self.positiveCC, 'negativeCC':self.negativeCC
        }
        self.electrodes = MeshRestriction(self.mesh, electrodes)
        self.electrolyte = MeshRestriction(self.mesh, electrolyte)
        self.solid_conductor = MeshRestriction(self.mesh, solid_conductor)
        self.current_colectors = MeshRestriction(self.mesh, current_colectors)
        self.electrode_cc_interfaces = MeshRestriction(self.mesh, [anode_CC_interface, cathode_CC_interface])
        self.positive_tab = MeshRestriction(self.mesh, positivetab)
        self.tabs = MeshRestriction(self.mesh, tabs)

        # Measures
        a_s_c_order = all([self.structure[i+1]=='s' for i, el in enumerate(self.structure) if el is 'a'])
        def int_dir(default_dir="+"):
            assert default_dir in ("+","-")
            reversed_dir = "-" if default_dir == "+" else "-"
            return default_dir if a_s_c_order else reversed_dir
        meta = {"quadrature_degree":2}
        self.dx = Measure('dx', domain=self.mesh, subdomain_data=subdomains, metadata=meta)
        self.dx_a = self.dx(self.field_data['anode'])
        self.dx_s = self.dx(self.field_data['separator'])
        self.dx_c = self.dx(self.field_data['cathode'])
        self.dx_pcc = self.dx(self.field_data['positiveCC'])
        self.dx_ncc = self.dx(self.field_data['negativeCC'])
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=boundaries, metadata=meta)
        self.ds_a = self.ds(self.field_data['negativePlug'])
        self.ds_c = self.ds(self.field_data['positivePlug'])
        self.dS = Measure('dS', domain=self.mesh, subdomain_data=interfaces, metadata=meta)
        self.dS_as = self.dS(self.field_data['interfaces']['anode-separator'], metadata={**meta, "direction": int_dir("+")})
        self.dS_sa = self.dS(self.field_data['interfaces']['anode-separator'], metadata={**meta, "direction": int_dir("-")})
        self.dS_sc = self.dS(self.field_data['interfaces']['cathode-separator'], metadata={**meta, "direction": int_dir("+")})
        self.dS_cs = self.dS(self.field_data['interfaces']['cathode-separator'], metadata={**meta, "direction": int_dir("-")})
        self.dS_cc_a = self.dS(self.field_data['interfaces']['anode-CC'], metadata={**meta, "direction": int_dir("+")})
        self.dS_a_cc = self.dS(self.field_data['interfaces']['anode-CC'], metadata={**meta, "direction": int_dir("-")})
        self.dS_cc_c = self.dS(self.field_data['interfaces']['cathode-CC'], metadata={**meta, "direction": int_dir("-")})
        self.dS_c_cc = self.dS(self.field_data['interfaces']['cathode-CC'], metadata={**meta, "direction": int_dir("+")})

        # Compute volumes 
        self._compute_volumes()

        timer.stop()
        print('Finished building mesh')
