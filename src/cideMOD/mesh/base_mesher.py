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

import dolfinx as dfx
from dolfinx.common import timed
from mpi4py import MPI
from ufl import Measure, FacetNormal, VectorElement, Cell, Mesh, grad, as_vector
from ufl.core.operator import Operator

from collections import namedtuple

import numpy as np

from cideMOD.helpers.logging import VerbosityLevel, _print
from cideMOD.cell.parser import CellParser
from cideMOD.numerics.fem_handler import interpolate, assemble_scalar as assemble


def mult_logical(*args, operator=np.logical_or):
    if len(args) == 0:
        return False
    else:
        res = args[0]
        for i in range(1, len(args)):
            res = operator(res, args[i])
        return res


def inside_element_expression(lst, x):
    if not lst:
        return np.zeros(x.shape[1], dtype=bool)
    conditions = []
    for element in lst:
        conditions.append(
            np.logical_and(np.greater_equal(x[0], element), np.less_equal(x[0], element + 1)))
    return mult_logical(*conditions, operator=np.logical_or)


def near_element_expression(lst, x):
    if not lst:
        return np.zeros(x.shape[1], dtype=bool)
    else:
        return mult_logical(*[np.isclose(x[0], element) for element in lst],
                            operator=np.logical_or)


def mark(mesh: dfx.mesh.Mesh, dim: int, subdomains):
    element_indices, element_markers = [], []
    assert dim <= mesh.topology.dim and dim >= 0
    for (marker, locator) in subdomains:
        facets = dfx.mesh.locate_entities(mesh, dim, locator)
        element_indices.append(facets)
        element_markers.append(np.full(len(facets), marker))
    element_indices = np.array(np.hstack(element_indices), dtype=np.int32)
    element_markers = np.array(np.hstack(element_markers), dtype=np.int32)
    sorted_elements = np.argsort(element_indices)
    element_tag = dfx.mesh.meshtags(
        mesh, dim, element_indices[sorted_elements], element_markers[sorted_elements])
    return element_tag


class SubdomainGenerator:
    def set_domain(self, lst):
        return lambda x: inside_element_expression(lst, x)

    def set_boundary(self, ref):
        return lambda x: np.isclose(x[0], ref)

    def set_boundaries(self, a, b):
        return lambda x: np.logical_or(np.isclose(x[0], a), np.isclose(x[0], b))

    def set_tab(self, ref, dim: int, initial: bool):
        assert dim > 1, "Can't use tabs in 1D cell"
        assert dim < 4, "Max dimension is 3"

        def tab(x):
            conditions = [np.greater_equal(x[0], ref),
                          np.less_equal(x[0], ref + 1),
                          np.isclose(x[1], 1)]
            if dim == 3:
                conditions += [np.greater_equal(x[2], 0.5 * int(initial)),
                               np.less_equal(x[2], 1 - 0.5 * int(initial))]
            return mult_logical(*conditions, operator=np.logical_and)
        return tab

    def set_interface(self, lst):
        return lambda x: near_element_expression(lst, x)

    def solid_conductor(self, structure):
        solid_conductors = ['a', 'c', 'pcc', 'ncc', 'li']
        index_list = [idx for idx, element in enumerate(structure) if element in solid_conductors]
        return lambda x: inside_element_expression(index_list, x)

    def current_collectors(self, structure):
        collectors = ['li', 'pcc', 'ncc']
        index_list = [idx for idx, element in enumerate(structure) if element in collectors]
        return lambda x: inside_element_expression(index_list, x)

    def electrolyte(self, structure):
        electrolyte = ['a', 's', 'c']
        index_list = [idx for idx, element in enumerate(structure) if element in electrolyte]
        return lambda x: inside_element_expression(index_list, x)

    def electrodes(self, structure):
        electrodes = ['a', 'c']
        index_list = [idx for idx, element in enumerate(structure) if element in electrodes]
        return lambda x: inside_element_expression(index_list, x)


class SubdomainMapper:
    @timed('Build SubdomainMapper')
    def __init__(self, field_restriction, function_space):
        index_map = function_space.dofmap.index_map
        self.domain_entities_map = {}
        self.domain_dof_map = {}
        self.ow_range = index_map.local_range
        self.base_array = np.zeros(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
        self.base_function = dfx.fem.Function(function_space)
        self._dummy_function = self.base_function.copy()
        for field_name, res in field_restriction.items():
            self.domain_entities_map[field_name] = {'dim': res[0], 'entities': res[1]}
            dofs = dfx.fem.locate_dofs_topological(function_space, res[0], res[1], False)
            self.domain_dof_map[field_name] = dofs
        self._dofs = None
        self._switches = None

    def generate_vector(self, source_dict: dict):
        out_array = self.base_array.copy()
        for domain_name, source in source_dict.items():
            dofs = self.domain_dof_map[domain_name]
            if domain_name in self.domain_dof_map.keys():
                if isinstance(source, dfx.fem.Function):
                    if source.vector.local_size == self.base_function.vector.local_size:
                        out_array[dofs] = source.vector.getValues([dofs])
                    elif source.vector.local_size == 1:
                        out_array[dofs] = source.vector.array[0]
                    else:
                        raise ValueError(
                            "Invalid source for domain mapper, number of dofs have to coincide")
                elif isinstance(source, (float, int)):
                    out_array[dofs] = source
                elif isinstance(source, (list, tuple, np.ndarray)):
                    if len(source) == len(out_array):
                        out_array[dofs] = source[dofs]
                    elif len(source) == len(dofs):
                        out_array[dofs] = source
                    elif len(source) == 1:
                        out_array[dofs] = source[0]
                    else:
                        raise ValueError(
                            "Invalid source for domain mapper, number of dofs have to coincide")
                elif isinstance(source, (Operator, dfx.fem.Expression)):
                    cells = self.domain_entities_map[domain_name]['entities']
                    interpolate(source, self._dummy_function, cells=cells)
                    out_array[dofs] = self._dummy_function.vector.getValues([dofs])
                else:
                    raise TypeError("Invalid source type for domain mapper")
        return out_array

    def generate_function(self, source_dict: dict):
        return self.interpolate(source_dict, self.base_function.copy())

    def interpolate(self, source_dict: dict, f: dfx.fem.Function, clear: bool = False):
        if f.vector.local_size != self.base_function.vector.local_size:
            raise ValueError("Invalid function for domain mapper, number of dofs have to coincide")
        if clear:
            interpolate(0., f)
        for domain_name, domain_source in source_dict.items():
            if domain_name in self.domain_dof_map.keys():
                dofs = self.domain_dof_map[domain_name]
                cells = self.domain_entities_map[domain_name]['entities']
                interpolate(domain_source, f, cells=cells, dofs=dofs)
            else:
                raise KeyError(f"Unrecognized domain '{domain_name}'. Available options are: '"
                               + "' '".join(self.domain_dof_map.keys()) + "'")
        return f

    def get_subdomains_dofs(self):
        # TODO: Needs sort + unique
        if self._dofs is not None:
            return self._dofs
        empty = np.array([], dtype=np.int32)
        dofs = namedtuple('SubdomainDofs', ' '.join(
            ['negativeCC', 'anode', 'separator', 'cathode', 'positiveCC', 'electrolyte',
             'solid_conductor', 'electrodes', 'collectors']))
        negCC = self.domain_dof_map.get('negativeCC', empty)
        anod = self.domain_dof_map.get('anode', empty)
        sep = self.domain_dof_map.get('separator', empty)
        cathod = self.domain_dof_map.get('cathode', empty)
        posCC = self.domain_dof_map.get('positiveCC', empty)
        electrolyte = np.concatenate((anod, sep, cathod))
        solid_conductor = np.concatenate((negCC, anod, cathod, posCC))
        electrodes = np.concatenate((anod, cathod))
        collectors = np.concatenate((negCC, posCC))
        self._dofs = dofs._make([negCC, anod, sep, cathod, posCC,
                                electrolyte, solid_conductor, electrodes, collectors])
        return self._dofs

    def get_subdomain_switches(self):
        if self._switches is not None:
            return self._switches
        subdomain_dofs = self.get_subdomains_dofs()
        switches = []
        for i, dofs in enumerate(subdomain_dofs):
            switches.append(self.base_function.copy())
            interpolate(1., switches[i], dofs=dofs)
        subdomain_switches = namedtuple('SubdomainSwitches', subdomain_dofs._fields)
        self._switches = subdomain_switches._make(switches)
        return self._switches


class BaseMesher:
    def __init__(self, options, cell: CellParser):
        self.options = options
        self.cell = cell
        self._comm = options.comm
        self.verbose = options.verbose
        self.model = self.options.model
        self.structure = cell.structure
        self.num_components = len(self.structure)
        self.field_data = {}
        self._measures = None
        self._set_subdomains()

    def _set_subdomains(self):
        # TODO: Make this implementation more generic, using cell components or delegate it
        #       to the models. This is only valid for the basic PXD electrochemical model and its
        #       submodels.
        self._subdomains_dict = {
            'anode': ['anode'],
            'separator': ['separator'],
            'cathode': ['cathode'],
            'negativeCC': ['negativeCC'],
            'positiveCC': ['positiveCC'],
            'cell': ['negativeCC', 'anode', 'separator', 'cathode', 'positiveCC'],
            'electrodes': ['anode', 'cathode'],
            'electrolyte': ['anode', 'separator', 'cathode'],
            'current_collectors': ['negativeCC', 'positiveCC'],
            'solid_conductor': ['negativeCC', 'anode', 'cathode', 'positiveCC']
        }

    def get_subdomains(self, subdomain: str):
        if subdomain not in self._subdomains_dict.keys():
            raise ValueError(f"Unrecognized subdomain '{subdomain}'. Available options: '"
                             + "' '".join(self._subdomains_dict.keys()) + "'")
        else:
            return self._subdomains_dict[subdomain]

    def get_dims(self, scale=1):
        domain_dict = {
            'a': self.cell.anode,
            's': self.cell.separator,
            'c': self.cell.cathode,
            'ncc': self.cell.negativeCC,
            'pcc': self.cell.positiveCC,
        }
        # TODO: Check if get_reference_value is the appropiate method here
        L, H, W = [], [], []
        for element in self.structure:
            data = domain_dict[element]
            L.append(data.thickness.get_reference_value() / scale)
            H.append(data.height.get_reference_value() / scale
                     if data.height.was_provided else None)
            W.append(data.width.get_reference_value() / scale
                     if data.width.was_provided else None)
        return L, H, W

    def get_measures(self):
        if self._measures is None:
            d = namedtuple('Measures', [
                'x', 'x_a', 'x_s', 'x_c', 'x_pcc', 'x_ncc', 's', 's_a', 's_c',
                'S_a_s', 'S_s_a', 'S_c_s', 'S_s_c', 'S_a_ncc', 'S_ncc_a', 'S_c_pcc', 'S_pcc_c'])
            self._measures = d._make([
                self.dx, self.dx_a, self.dx_s, self.dx_c, self.dx_pcc, self.dx_ncc, self.ds,
                self.ds_a, self.ds_c, self.dS_as, self.dS_sa, self.dS_cs, self.dS_sc, self.dS_a_cc,
                self.dS_cc_a, self.dS_c_cc, self.dS_cc_c])
        return self._measures

    def get_restrictions(self):
        res_dict = {
            'anode': self.anode,
            'separator': self.separator,
            'cathode': self.cathode,
            'positiveCC': self.positiveCC,
            'negativeCC': self.negativeCC,
            'electrolyte': self.electrolyte,
            'electrodes': self.electrodes,
            'current_collectors': self.current_collectors,
            'solid_conductor': self.solid_conductor,
            'anode_cc_facets': self.anode_CC_facets,
            'cathode_cc_facets': self.cathode_CC_facets,
            'electrode_cc_facets': self.electrode_CC_facets,
            'positive_tab': self.positive_tab,
            'negative_tab': self.negative_tab,
            'tabs': self.tabs,
            'cell': None
        }
        res = namedtuple('MeshRestrictions', res_dict.keys())
        return res._make(res_dict.values())

    def check_subdomains(self, subdomains, field_data):
        # DEBUG only - Check subdomains components
        # if MPI.size(MPI.comm_world) == 1:
        #     sd_array = self.subdomains.array().copy()
        #     assert len(set(sd_array)) == len(set(self.structure)), \
        #           'Some elements not inside a subdomain'
        #     for i, sd in enumerate(sd_array):
        #         if i == 0:
        #             assert sd == sd_array[i+1], 'A subdomain has only one element'
        #         elif i == len(sd_array)-1:
        #             assert sd == sd_array[i-1], 'A subdomain has only one element'
        #         else:
        #             assert sd in (sd_array[i+1], sd_array[i-1]), \
        #                   'A subdomain has only one element'
        # Ensure electrode subdomains have higher values
        if max([val for val in field_data.values() if isinstance(val, int)]) >= 20:
            raise RuntimeError("Unusualy high subdomain id")
        subdomains.values[subdomains.values == field_data['anode']] = 21
        subdomains.values[subdomains.values == field_data['cathode']] = 22
        field_data['anode'] = 21
        field_data['cathode'] = 22
        return subdomains, field_data

    def _compute_volumes(self):
        d = self.get_measures()
        values = [assemble(1 * dx) for dx in d]
        volumes = namedtuple('ScaledVolumes', d._fields)
        self.volumes = volumes._make(values)

    def get_component_gradient(self, component, L, H=None, W=None, dimless_model=False):
        """
        dimless_model is True is the model is dimensionless so the mesh is dimensional
        """
        components = ['negativeCC', 'anode', 'separator', 'cathode', 'positiveCC']
        if component not in components:
            raise ValueError(f"Unrecognized component '{component}'. Available options: '"
                             + "' '".join(components) + "'")

        if dimless_model:
            return grad

        norm = [L]
        if self.mesh.geometry.dim > 1:
            if H is None:
                raise ValueError(
                    "Unable to compute the normalized gradient. Height has not been provided.")
            norm.append(H)

        if self.mesh.geometry.dim > 2:
            if W is None:
                raise ValueError(
                    "Unable to compute the normalized gradient. Width has not been provided.")
            norm.append(W)

        def normalized_grad(arg):
            "Return normalized gradient for normalized domains"
            return as_vector([arg.dx(i) / norm[i] for i in range(self.mesh.geometry.dim)])

        return normalized_grad


class DolfinMesher(BaseMesher):

    def get_component_gradient(self, component, L, H=None, W=None, dimless_model=False):
        if dimless_model:
            raise NotImplementedError("Dolfin mesher does not support dimensionless model.")
        return super().get_component_gradient(component, L, H, W, dimless_model)

    @timed('Building mesh')
    def build_mesh(self, **kwargs):
        N_x = self.options.N_x
        N_y = self.options.N_y or N_x
        N_z = self.options.N_z or N_x
        n = self.num_components
        L, _, _ = self.get_dims()
        self.scale = L
        if isinstance(N_x, list):  # NOTE: maybe should allow it to be also np.array
            assert self.model == 'P2D', 'Different discretization in x only supported in P2D'
            assert len(N_x) == n, 'N_x must have the same number of elements as the cell structure'
            nodes = sum(N_x)
        else:
            nodes = (n * N_x if self.model == 'P2D' else (
                n * N_x * N_y if self.model == 'P3D' else
                n * N_x * N_y * N_z))
        if self.verbose >= VerbosityLevel.BASIC_PROBLEM_INFO:
            _print(f"Building mesh for {self.model} problem with {n} components and {nodes} nodes",
                   comm=self._comm)

        if self.model == 'P4D':
            p1 = (0, 0, 0)
            p2 = (n, 1, 1)
            self.mesh = dfx.mesh.create_box(self._comm, [p1, p2], [N_x * n, N_y, N_z])
        elif self.model == 'P3D':
            p1 = (0, 0, 0)
            p2 = (n, 1, 0)
            self.mesh = dfx.mesh.create_rectangle(self._comm, [p1, p2], [N_x * n, N_y])
        elif self.model == 'P2D':
            if isinstance(N_x, list):
                domain = Mesh(VectorElement("Lagrange", Cell('interval', 1), 1))
                vertices = [[i] for i in np.linspace(0, 1, N_x[0] + 1)]
                for ii in range(1, len(N_x)):
                    vertices += [[i] for i in np.linspace(ii, ii + 1, N_x[ii] + 1)][1:]
                vertices = np.array(vertices, dtype=np.float64)
                cells = np.array([[i, i + 1] for i in range(sum(N_x))], dtype=np.int64)
                self.mesh = dfx.mesh.create_mesh(MPI.COMM_WORLD, cells, vertices, domain)
            else:
                self.mesh = dfx.mesh.create_interval(self._comm, N_x * n, [0, n])

        else:
            raise ValueError(f"Unable to build the mesh. Unrecognized model '{self.model}'")

        cells_dim = self.mesh.topology.dim
        facets_dim = cells_dim - 1
        self.mesh.topology.create_connectivity(facets_dim, cells_dim)
        self.f_to_c = self.mesh.topology.connectivity(facets_dim, cells_dim)

        subdomain_generator = SubdomainGenerator()
        self.field_data['anode'] = 1
        self.field_data['separator'] = 2
        self.field_data['cathode'] = 3
        self.field_data['negativeCC'] = 4
        self.field_data['positiveCC'] = 5
        self.field_data['anode-separator'] = 6
        self.field_data['cathode-separator'] = 7
        self.field_data['anode-CC'] = 8
        self.field_data['cathode-CC'] = 9
        self.field_data['negative_plug'] = 10
        self.field_data['positive_plug'] = 11

        # Mark boundaries

        if self.structure[-1] in ('c', 'pcc') or 'c' not in self.structure:
            negativetab = subdomain_generator.set_boundary(0)
            positivetab = subdomain_generator.set_boundary(n)
        else:
            negativetab = subdomain_generator.set_boundary(n)
            positivetab = subdomain_generator.set_boundary(0)

        bounds = [
            (self.field_data['negative_plug'], negativetab),
            (self.field_data['positive_plug'], positivetab),
        ]
        boundaries = mark(self.mesh, facets_dim, bounds)
        tabs = subdomain_generator.set_boundaries(0, n)

        self.boundaries = boundaries

        # Mark subdomains
        anode_list = [idx for idx, element in enumerate(self.structure) if element == 'a']
        cathode_list = [idx for idx, element in enumerate(self.structure) if element == 'c']
        separator_list = [idx for idx, element in enumerate(self.structure) if element == 's']
        positive_cc_list = [idx for idx, element in enumerate(self.structure) if element == 'pcc']
        negative_cc_list = [idx for idx, element in enumerate(self.structure) if element == 'ncc']

        negative_cc = subdomain_generator.set_domain(negative_cc_list)
        anode = subdomain_generator.set_domain(anode_list)
        separator = subdomain_generator.set_domain(separator_list)
        cathode = subdomain_generator.set_domain(cathode_list)
        positive_cc = subdomain_generator.set_domain(positive_cc_list)

        electrodes = subdomain_generator.electrodes(self.structure)
        electrolyte = subdomain_generator.electrolyte(self.structure)
        solid_conductor = subdomain_generator.solid_conductor(self.structure)
        current_collectors = subdomain_generator.current_collectors(self.structure)

        subs = [
            (self.field_data['negativeCC'], negative_cc),
            (self.field_data['anode'], anode),
            (self.field_data['separator'], separator),
            (self.field_data['cathode'], cathode),
            (self.field_data['positiveCC'], positive_cc),
        ]
        subdomains = mark(self.mesh, cells_dim, subs)
        self.subdomains = subdomains
        self.check_subdomains(self.subdomains, self.field_data)

        # Mark interfaces
        negativeCC_interfaces = set(negative_cc_list).union([i + 1 for i in negative_cc_list])
        anode_interfaces = set(anode_list).union([i + 1 for i in anode_list])
        separator_interfaces = set(separator_list).union([i + 1 for i in separator_list])
        cathode_interfaces = set(cathode_list).union([i + 1 for i in cathode_list])
        positiveCC_interfaces = set(positive_cc_list).union([i + 1 for i in positive_cc_list])
        anode_separator_interface_list = anode_interfaces.intersection(separator_interfaces)
        anode_CC_interface_list = anode_interfaces.intersection(negativeCC_interfaces)
        cathode_separator_interface_list = cathode_interfaces.intersection(separator_interfaces)
        cathode_CC_interface_list = cathode_interfaces.intersection(positiveCC_interfaces)

        anode_separator_interface = subdomain_generator.set_interface(
            anode_separator_interface_list)
        cathode_separator_interface = subdomain_generator.set_interface(
            cathode_separator_interface_list)
        anode_CC_interface = subdomain_generator.set_interface(anode_CC_interface_list)
        cathode_CC_interface = subdomain_generator.set_interface(cathode_CC_interface_list)

        inters = [
            (self.field_data['anode-separator'], anode_separator_interface),
            (self.field_data['cathode-separator'], cathode_separator_interface),
            (self.field_data['anode-CC'], anode_CC_interface),
            (self.field_data['cathode-CC'], cathode_CC_interface),
        ]
        interfaces = mark(self.mesh, facets_dim, inters)
        self.interfaces = interfaces

        def _locate_entities(subdomain_locator, dim):
            subdomain_cells = dfx.mesh.locate_entities(self.mesh, dim, subdomain_locator)
            num_local = self.mesh.topology.index_map(dim).size_local
            return subdomain_cells[subdomain_cells < num_local]  # NOTE: Needed for parallelization

        self.anode = (cells_dim, _locate_entities(anode, cells_dim))

        self.separator = (cells_dim, _locate_entities(separator, cells_dim))

        self.cathode = (cells_dim, _locate_entities(cathode, cells_dim))

        self.positiveCC = (cells_dim, _locate_entities(positive_cc, cells_dim))

        self.negativeCC = (cells_dim, _locate_entities(negative_cc, cells_dim))

        self.field_restrictions = {
            'anode': self.anode,
            'separator': self.separator,
            'cathode': self.cathode,
            'positiveCC': self.positiveCC,
            'negativeCC': self.negativeCC
        }
        self.electrodes = (cells_dim, _locate_entities(electrodes, cells_dim))
        self.electrolyte = (cells_dim, _locate_entities(electrolyte, cells_dim))
        self.solid_conductor = (cells_dim, _locate_entities(solid_conductor, cells_dim))
        self.current_collectors = (cells_dim, _locate_entities(current_collectors, cells_dim))
        self.anode_CC_facets = (facets_dim, _locate_entities(anode_CC_interface, facets_dim))
        self.cathode_CC_facets = (facets_dim, _locate_entities(cathode_CC_interface, facets_dim))
        self.electrode_CC_facets = (
            facets_dim,
            np.unique(np.concatenate([self.anode_CC_facets[1], self.cathode_CC_facets[1]]))
        )
        # TODO: Consider using dfx.mesh.locate_entities_boundary
        self.positive_tab = (facets_dim, _locate_entities(positivetab, facets_dim))
        self.negative_tab = (facets_dim, _locate_entities(negativetab, facets_dim))
        self.tabs = (facets_dim, _locate_entities(tabs, facets_dim))

        # Measures
        a_s_c_order = all([self.structure[i + 1] == 's'
                           for i, el in enumerate(self.structure) if el == 'a'])

        def int_dir(default_dir="+"):
            assert default_dir in ("+", "-")
            reversed_dir = "-" if default_dir == "+" else "+"
            return default_dir if a_s_c_order else reversed_dir
        meta = {"quadrature_degree": 2}
        self.dx = Measure('dx', domain=self.mesh, subdomain_data=subdomains, metadata=meta)
        self.dx_a = self.dx(self.field_data['anode'])
        self.dx_s = self.dx(self.field_data['separator'])
        self.dx_c = self.dx(self.field_data['cathode'])
        self.dx_pcc = self.dx(self.field_data['positiveCC'])
        self.dx_ncc = self.dx(self.field_data['negativeCC'])
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=boundaries, metadata=meta)
        self.ds_a = self.ds(self.field_data['negative_plug'])
        self.ds_c = self.ds(self.field_data['positive_plug'])
        self.dS = Measure('dS', domain=self.mesh, subdomain_data=interfaces, metadata=meta)
        self.dS_as = self.dS(
            self.field_data['anode-separator'], metadata={**meta, "direction": int_dir("+")})
        self.dS_sa = self.dS(
            self.field_data['anode-separator'], metadata={**meta, "direction": int_dir("-")})
        self.dS_sc = self.dS(
            self.field_data['cathode-separator'], metadata={**meta, "direction": int_dir("+")})
        self.dS_cs = self.dS(
            self.field_data['cathode-separator'], metadata={**meta, "direction": int_dir("-")})
        self.dS_cc_a = self.dS(
            self.field_data['anode-CC'], metadata={**meta, "direction": int_dir("+")})
        self.dS_a_cc = self.dS(
            self.field_data['anode-CC'], metadata={**meta, "direction": int_dir("-")})
        self.dS_cc_c = self.dS(
            self.field_data['cathode-CC'], metadata={**meta, "direction": int_dir("-")})
        self.dS_c_cc = self.dS(
            self.field_data['cathode-CC'], metadata={**meta, "direction": int_dir("+")})

        # Compute volumes
        self._compute_volumes()

        # Facet normal directions
        self.normal = FacetNormal(self.mesh)

        if self.verbose >= VerbosityLevel.BASIC_PROBLEM_INFO:
            _print('Finished mesh construction', comm=self._comm)
