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
import tempfile

import gmsh
import meshio
import numpy as np
import pygmsh


class GmshGenerator:
    def __init__(self):
        self.geom = pygmsh.geo.Geometry()
        self.labels = {
            'a': 'anode',
            's': 'separator',
            'c': 'cathode',
            'ncc': 'negativeCC',
            'pcc': 'positiveCC'
        }
        
        self.discretization = {
            'a': {'n': 1, 'type': 'Progression', 'par': 1},
            's': {'n': 0.33, 'type': 'Progression', 'par': 1},
            'c': {'n': 1, 'type': 'Progression', 'par': 1},
            'pcc': {'n': 0.2, 'type': 'Progression', 'par': 1},
            'ncc': {'n': 0.2, 'type': 'Progression', 'par': 1},
        }

    def _adapt_discretization(self, structure, L, nL: int = 30):
        min_cells = {
            'a': 10,
            's': 5,
            'c': 10,
            'pcc': 5,
            'ncc': 5
        }
        for i, el in enumerate(structure):
            self.discretization[el]['n'] = int(nL * self.discretization[el]['n'])
            # size = L[i] / self.discretization[el]['n']
            # if size < 2e-6:
            #     self.discretization[el]['n'] = int(max(min_cells[el], np.ceil(L[i] / 2e-6)) + 1)

            # elif size > 2e-6:
            #     sep_index = structure.index('s')
            #     sep_size = L[i] / self.discretization[el]['n']
            #     if el in ['a', 'c']:
            #         for r in [1.025, 1.05, 1.075, 1.1, 1.15, 1.2]:
            #             n = 2 * np.log(1 + (r - 1) * L[i] / (sep_size * 2)) / np.log(r)
            #             if n < 45:
            #                 break
            #         self.discretization[el]['n'] = int(round(n))
            #         self.discretization[el]['type'] = 'Bump'
            #         self.discretization[el]['par'] = r**((-round(n) - 1) / 2)

    def gmshEnvironment(dim):
        def gmshDecorator(func):
            def decorated_method(self, *args, **kwargs):
                self.geom.__enter__()
                gmsh.option.setNumber("General.ExpertMode", 1)
                gmsh.option.setNumber("General.Verbosity", 0)
                func(self, *args, **kwargs)
                self.geom.__exit__()
            return decorated_method
        return gmshDecorator

    def generate_mesh_from_template(self, filename, output, dim=3, parameters: dict = {}):
        gmsh.initialize()
        for key, value in parameters.items():
            if isinstance(value, (float, int)):
                gmsh.onelab.setNumber('Parameters/{}'.format(key), [value])
            elif isinstance(value, (str)):
                gmsh.onelab.setString('Parameters/{}'.format(key), [value])
        gmsh.open(filename)
        gmsh.model.mesh.generate(dim)
        gmsh.write(output)
        gmsh.finalize()

    @gmshEnvironment(1)
    def create_1D_mesh(self, filename: str = '', structure=['a', 's', 'c'],
                       L=[76e-6, 25e-6, 68e-6], nL: int = 30):
        self._adapt_discretization(structure, L, nL)
        n_elements = len(structure)
        points = self._draw_point_series(start=[0, 0, 0], L=L, structure=structure, direction=0)

        lines = np.zeros(n_elements).tolist()
        for i in range(n_elements):
            lines[i] = self.geom.add_line(points[i], points[i + 1])

        self._label_physical_elements(lines, structure)

        ncc = self.geom.add_physical(points[0], label='negativePlug')
        pcc = self.geom.add_physical(points[-1], label='positivePlug')

        for i in range(n_elements):
            trans_pars = self.discretization[structure[i]]
            self.geom.set_transfinite_curve(
                lines[i], trans_pars['n'], trans_pars['type'], trans_pars['par'])

        self.geom.generate_mesh(dim=1, verbose=True)
        if filename:
            self.write_gmsh_file(filename)
        else:
            self.gmsh_mesh = self._get_gmsh_mesh()

    @gmshEnvironment(2)
    def create_2D_mesh(self, filename: str = '', structure=['a', 's', 'c'],
                       L=[76e-6, 25e-6, 68e-6], nL: int = 30, H=0.01, nH: int = 30):
        n_elements = len(structure)
        self._adapt_discretization(structure, L, nL)
        # Generate geometry
        # Points
        points_down = self._draw_point_series(
            start=[0, 0, 0], L=L, structure=structure, direction=0, sign=1)
        points_up = self._draw_point_series(
            start=[0, H, 0], L=L, structure=structure, direction=0, sign=1)
        # Lines
        lines = np.zeros(3 * (n_elements) + 1).tolist()
        for i in range(n_elements):
            lines[3 * i] = self.geom.add_line(points_down[i], points_up[i])
            lines[3 * i + 1] = self.geom.add_line(points_down[i], points_down[i + 1])
            lines[3 * i + 2] = self.geom.add_line(points_up[i + 1], points_up[i])
        lines[-1] = self.geom.add_line(points_down[-1], points_up[-1])
        # Curve Loops & Surfaces
        surfaces = np.zeros(n_elements).tolist()
        for i in range(n_elements):
            surfaces[i] = self.geom.add_plane_surface(
                self.geom.add_curve_loop(
                    [lines[3 * i + 1], 
                     lines[3 * (i + 1)], 
                     lines[3 * i + 2], 
                     -lines[3 * i]]))

        lines_to_label = [lines[3 * i] for i in range(n_elements + 1)]
        # Label physical entities
        self._label_physical_elements(surfaces, structure)

        ncc = self.geom.add_physical(lines[0], label='negativePlug')
        pcc = self.geom.add_physical(lines[-1], label='positivePlug')
        Y_m = self.geom.add_physical(
            self._flatten_list([lines[3 * i + 1] for i in range(n_elements)]), 
            label='Y_m')

        # Define mesh
        for i in range(n_elements):
            trans_pars = self.discretization[structure[i]]
            self.geom.set_transfinite_curve(
                lines[3 * i + 1], trans_pars['n'], trans_pars['type'], trans_pars['par'])
            self.geom.set_transfinite_curve(
                lines[3 * i + 2], trans_pars['n'], trans_pars['type'], trans_pars['par'])
        for i in range(n_elements + 1):
            self.geom.set_transfinite_curve(lines[3 * i], nH, 'Progression', 1)
        for i in range(n_elements):
            self.geom.set_transfinite_surface(surfaces[i], 'Left', [])

        # Generate and export mesh
        self.geom.generate_mesh(dim=2, verbose=True)
        if filename:
            self.write_gmsh_file(filename)
        else:
            self.gmsh_mesh = self._get_gmsh_mesh()

    @gmshEnvironment(3)
    def create_3D_mesh_with_tabs(
            self, filename: str = '', structure=['a', 's', 'c'],
            L=[76e-6, 25e-6, 68e-6], nL: int = 30, H=0.01, nH: int = 10, Z=0.01, nZ: int = 10,
            tab_locations=[('up', 'left'), ('up', 'right')]):
        n_elements = len(structure)
        self._adapt_discretization(structure, L, nL)
        # Check structure
        if 'pcc' in structure or 'ncc' in structure:
            if structure[0] not in ('pcc', 'ncc') or structure[-1] not in ('pcc', 'ncc'):
                raise ValueError("Current collectors must be at the extremes of the cell")
            tab_indexes = []
            for i, element in enumerate(structure):
                if element not in ('ncc', 'pcc'):
                    continue
                h_ind, z_ind = self._get_tab_location(tab_locations[0 if element == 'ncc' else 1])
                tab_indexes.append((h_ind, z_ind, i, element))

        # Generate geometries
        # Points
        distribution = [0 - 2 / 10, 0, 1 / 10, 4 / 10, 6 / 10, 9 / 10, 1, 1 + 2 / 10]
        points = np.zeros((6, 6)).tolist()
        for i in range(6):
            for j in range(6):
                points[i][j] = self._draw_point_series(
                    start=[0, distribution[i + 1] * H, distribution[j + 1] * Z],
                    L=L, structure=structure, direction=0, sign=1)
        points = np.transpose(np.array(points), (2, 0, 1)).tolist()
        # - Tab points
        if 'pcc' in structure or 'ncc' in structure:
            tab_points = np.zeros((len(tab_indexes), 4)).tolist()
            for i, tab in enumerate(tab_indexes):
                l_pos = [sum([L[j] for j in range(tab[2])]),
                         sum([L[j] for j in range(tab[2] + 1)])]
                if tab[0] in [0, 7]:  # Vertical tabs
                    y_pos = [H * distribution[tab[0]]]
                    z_pos = [Z * distribution[tab[1]], Z * distribution[tab[1] + 1]]
                    tab_points[i][0] = self.geom.add_point(
                        [l_pos[0], y_pos[0], z_pos[0]])
                    tab_points[i][1] = self.geom.add_point(
                        [l_pos[0], y_pos[0], z_pos[1]])
                    tab_points[i][2] = self.geom.add_point(
                        [l_pos[1], y_pos[0], z_pos[0]])
                    tab_points[i][3] = self.geom.add_point(
                        [l_pos[1], y_pos[0], z_pos[1]])
                else:  # Horizontal tabs
                    raise NotImplementedError("Only vertical tabs are available.")

        # Lines: [transversal, vertical (in-plane), horizontal (in-plane)]
        lines = [
            np.zeros((n_elements, 6, 6)).tolist(),  # Transversal
            np.zeros((n_elements + 1, 5, 6)).tolist(),  # V
            np.zeros((n_elements + 1, 6, 5)).tolist()  # H
        ]
        for i in range(n_elements):
            for j in range(6):
                for k in range(6):
                    lines[0][i][j][k] = self.geom.add_line(points[i][j][k], points[i + 1][j][k])
        for i in range(n_elements + 1):
            for j in range(5):
                for k in range(6):
                    lines[1][i][j][k] = self.geom.add_line(points[i][j][k], points[i][j + 1][k])
        for i in range(n_elements + 1):
            for j in range(6):
                for k in range(5):
                    lines[2][i][j][k] = self.geom.add_line(points[i][j][k], points[i][j][k + 1])

        # - Tab lines
        if 'pcc' in structure or 'ncc' in structure:
            tab_lines = np.zeros((len(tab_indexes), 8)).tolist()
            for i, tab in enumerate(tab_indexes):
                if tab[0] in [0, 7]:
                    i_tab = tab[2]
                    j_tab = max(tab[0] - 2, 0)
                    k_tab = tab[1]
                    tab_lines[i][0] = self.geom.add_line(tab_points[i][0], tab_points[i][1])
                    tab_lines[i][1] = self.geom.add_line(tab_points[i][0], tab_points[i][2])
                    tab_lines[i][2] = self.geom.add_line(tab_points[i][2], tab_points[i][3])
                    tab_lines[i][3] = self.geom.add_line(tab_points[i][1], tab_points[i][3])
                    tab_lines[i][4] = self.geom.add_line(
                        points[i_tab][j_tab][k_tab - 1], tab_points[i][0])
                    tab_lines[i][5] = self.geom.add_line(
                        points[i_tab][j_tab][k_tab], tab_points[i][1])
                    tab_lines[i][6] = self.geom.add_line(
                        points[i_tab + 1][j_tab][k_tab - 1], tab_points[i][2])
                    tab_lines[i][7] = self.geom.add_line(
                        points[i_tab + 1][j_tab][k_tab], tab_points[i][3])
                else:
                    raise NotImplementedError("Only vertical tabs are available.")

        # Curve_loops & Surfaces
        # Curve_loops: [In-plane (Front-back), Up-Down (H+, H-), Sides (Z+, Z-)]
        surfaces = [
            np.zeros((5, 5, n_elements + 1)).tolist(),  # In-plane
            np.zeros((6, 5, n_elements)).tolist(),  # Up-Down
            np.zeros((5, 6, n_elements)).tolist(),  # Sides
        ]

        for i in range(5):
            for j in range(5):
                for k in range(n_elements + 1):
                    surfaces[0][i][j][k] = self.geom.add_plane_surface(
                        self.geom.add_curve_loop(
                            [lines[2][k][i][j],
                             lines[1][k][i][j + 1],
                             -lines[2][k][i + 1][j],
                             -lines[1][k][i][j]]))
        for i in range(6):
            for j in range(5):
                for k in range(n_elements):
                    surfaces[1][i][j][k] = self.geom.add_plane_surface(
                        self.geom.add_curve_loop(
                            [lines[0][k][i][j],
                             lines[2][k + 1][i][j],
                             -lines[0][k][i][j + 1],
                             -lines[2][k][i][j]]))
        for i in range(5):
            for j in range(6):
                for k in range(n_elements):
                    surfaces[2][i][j][k] = self.geom.add_plane_surface(
                        self.geom.add_curve_loop(
                            [lines[1][k][i][j],
                             lines[0][k][i + 1][j],
                             -lines[1][k + 1][i][j],
                             -lines[0][k][i][j]]))

        # - Tab curve loops & surfaces
        if 'pcc' in structure or 'ncc' in structure:
            tab_surfaces = np.zeros((len(tab_indexes), 5)).tolist()
            for i, tab in enumerate(tab_indexes):
                if tab[0] in [0, 7]:
                    i_tab = tab[2]
                    j_tab = max(tab[0] - 2, 0)
                    k_tab = tab[1]
                    # Top surface
                    tab_surfaces[i][0] = self.geom.add_plane_surface(
                        self.geom.add_curve_loop(
                            [tab_lines[i][1],
                             tab_lines[i][2],
                             -tab_lines[i][3],
                             -tab_lines[i][0]]))
                    # rear
                    tab_surfaces[i][1] = self.geom.add_plane_surface(
                        self.geom.add_curve_loop(
                            [-tab_lines[i][0],
                             -tab_lines[i][4],
                             lines[2][i_tab][j_tab][k_tab - 1],
                             tab_lines[i][5]]))
                    # back
                    tab_surfaces[i][2] = self.geom.add_plane_surface(
                        self.geom.add_curve_loop(
                            [tab_lines[i][1],
                             -tab_lines[i][6],
                             -lines[0][i_tab][j_tab][k_tab - 1],
                             tab_lines[i][4],
                             ]))
                    # side
                    tab_surfaces[i][3] = self.geom.add_plane_surface(
                        self.geom.add_curve_loop(
                            [-tab_lines[i][2],
                             -tab_lines[i][6],
                             lines[2][i_tab + 1][j_tab][k_tab - 1],
                             tab_lines[i][7],
                             ]))
                    # front
                    tab_surfaces[i][4] = self.geom.add_plane_surface(
                        self.geom.add_curve_loop(
                            [tab_lines[i][3],
                             -tab_lines[i][7],
                             -lines[0][i_tab][j_tab][k_tab],
                             tab_lines[i][5],
                             ]))
                else:
                    raise NotImplementedError("Only vertical tabs are available.")
                  
        # Surface Loops & Volumes
        volumes = np.zeros((5, 5, n_elements)).tolist()
        for i in range(5):
            for j in range(5):
                for k in range(n_elements):
                    volumes[i][j][k] = self.geom.add_volume(
                        self.geom.add_surface_loop(
                            [surfaces[0][i][j][k],
                             surfaces[1][i][j][k],
                             surfaces[2][i][j][k],
                             surfaces[0][i][j][k + 1],
                             surfaces[1][i + 1][j][k],
                             surfaces[2][i][j + 1][k]]))

        # - Tab surface loops & volumes
        if 'pcc' in structure or 'ncc' in structure:
            tab_volumes = np.zeros(len(tab_indexes)).tolist()
            for i, tab in enumerate(tab_indexes):
                tab_volumes[i] = self.geom.add_volume(
                    self.geom.add_surface_loop(
                        [tab_surfaces[i][j] for j in range(5)]
                        + [surfaces[1][max(tab[0] - 2, 0)][tab[1] - 1][tab[2]]]))

        # Re-structure volumes array to label according element number
        volumes_to_label = [[[volumes[i][j][k] for i in range(5)] for j in range(5)]
                            for k in range(n_elements)]
        if 'pcc' in structure or 'ncc' in structure:
            for i, tab in enumerate(tab_indexes):
                volumes_to_label[tab[2]].append(tab_volumes[i])

        # Label physical entities
        self._label_physical_elements(volumes_to_label, structure)

        if 'pcc' in structure or 'ncc' in structure:
            ncc = self.geom.add_physical(
                self._flatten_list([tab_surfaces[i][j] for j in range(1)
                                    for i, tab in enumerate(tab_indexes) if tab[3] == 'ncc']),
                label='negativePlug')
            pcc = self.geom.add_physical(
                self._flatten_list([tab_surfaces[i][j] for j in range(1)
                                    for i, tab in enumerate(tab_indexes) if tab[3] == 'pcc']),
                label='positivePlug')
        else:
            ncc = self.geom.add_physical(
                self._flatten_list([surfaces[0][i][j][0] for i in range(5) for j in range(5)]),
                label='negativePlug')
            pcc = self.geom.add_physical(
                self._flatten_list([surfaces[0][i][j][-1] for i in range(5) for j in range(5)]),
                label='positivePlug')

        Y_m = self.geom.add_physical(
            self._flatten_list([surfaces[1][0][j][k]
                                for k in range(n_elements) for j in range(5)]),
            label='Y_m')

        # Define mesh
        H_disc = [
            1 + int(max(1, np.ceil(nH / 10))),
            1 + int(max(2, np.ceil(nH * 3 / 10))),
            1 + int(max(1, np.ceil(nH * 2 / 10))),
            1 + int(max(2, np.ceil(nH * 3 / 10))),
            1 + int(max(1, np.ceil(nH / 10)))]
        Z_disc = [
            1 + int(max(1, np.ceil(nZ / 10))),
            1 + int(max(2, np.ceil(nZ * 3 / 10))),
            1 + int(max(1, np.ceil(nZ * 2 / 10))),
            1 + int(max(2, np.ceil(nZ * 3 / 10))),
            1 + int(max(1, np.ceil(nZ / 10)))]
        
        for i in range(n_elements):
            trans_pars = self.discretization[structure[i]]
            for j in range(6):
                for k in range(6):
                    self.geom.set_transfinite_curve(lines[0][i][j][k], trans_pars['n'],
                                                    trans_pars['type'], trans_pars['par'])

        for i in range(n_elements + 1):
            for j in range(5):
                for k in range(6):
                    self.geom.set_transfinite_curve(lines[1][i][j][k], H_disc[j], 'Progression', 1)

        for i in range(n_elements + 1):
            for j in range(6):
                for k in range(5):
                    self.geom.set_transfinite_curve(lines[2][i][j][k], Z_disc[k], 'Progression', 1)

        all_surfaces = self._flatten_list(surfaces)
        for surf in all_surfaces:
            self.geom.set_transfinite_surface(surf, 'Left', [])
        all_volumes = self._flatten_list(volumes)
        for vol in all_volumes:
            self.geom.set_transfinite_volume(vol, [])

        # - Tab mesh definition
        if 'pcc' in structure or 'ncc' in structure:
            for i, tab in enumerate(tab_indexes):
                trans_pars = self.discretization[structure[tab[2]]]
                if tab[0] in (7, 0):
                    self.geom.set_transfinite_curve(
                        tab_lines[i][0], Z_disc[1], 'Progression', 1)
                    self.geom.set_transfinite_curve(
                        tab_lines[i][1],
                        trans_pars['n'], trans_pars['type'], trans_pars['par'])
                    self.geom.set_transfinite_curve(
                        tab_lines[i][2], Z_disc[1], 'Progression', 1)
                    self.geom.set_transfinite_curve(
                        tab_lines[i][3],
                        trans_pars['n'], trans_pars['type'], trans_pars['par'])
                    for j in range(4):
                        self.geom.set_transfinite_curve(
                            tab_lines[i][4 + j], H_disc[2], 'Progression', 1)
                else:
                    raise NotImplementedError("Only vertical tabs are available.")
                
                for j in range(5):
                    self.geom.set_transfinite_surface(tab_surfaces[i][j], 'Left', [])
                self.geom.set_transfinite_volume(tab_volumes[i], [])
        self.geom.generate_mesh(dim=3, verbose=True)
        if filename:
            self.write_gmsh_file(filename)
        else:
            self.gmsh_mesh = self._get_gmsh_mesh()

    def _label_physical_elements(self, elements: list, structure: list):
        assert len(elements) == len(structure), "Element number incorrect"
        keys = np.array(structure)
        ind = np.arange(len(structure))
        key_dict = {}
        for k in set(keys):
            key_dict[k] = ind[k == keys]

        for k in key_dict:
            objects = []
            for index in key_dict[k]:
                flattened_elements = self._flatten_list(elements[index])
                for item in flattened_elements:
                    objects.append(item)
            self.geom.add_physical(objects, label=self.labels[k])

    def _flatten_list(self, elements):
        if not isinstance(elements, list):
            return [elements]
        else:
            flattened_list = []
            for element in elements:
                if isinstance(element, list):
                    flattened_sublist = self._flatten_list(element)
                    for subelement in flattened_sublist:
                        flattened_list.append(subelement)
                else:
                    flattened_list.append(element)
            return flattened_list

    def _get_tab_location(self, tab_location):
        locations = {
            'up': {
                'left': (7, 2),
                'right': (7, 4),
            },
            'down': {
                'left': (0, 2),
                'right': (0, 4),
            },
            'left': {
                'up': (4, 0),
                'down': (2, 0),
            },
            'right': {
                'up': (4, 7),
                'down': (2, 7),
            }
        }
        return locations[tab_location[0]][tab_location[1]]

    def _draw_point_series(self, start: list, L: list, structure: list,
                           direction: int, sign: int = 1):
        assert len(start) in (2, 3), "Start point must have 2 or 3 coordinates"
        assert direction < len(start)
        assert sign in (1, -1)
        lcar = max(L)
        n_elem = len(L)
        points = np.zeros(n_elem + 1).tolist()
        x = start
        points[0] = self.geom.add_point(x.copy())
        for i in range(n_elem - 1):
            x[direction] += sign * L[i]
            points[i + 1] = self.geom.add_point(x.copy())
        x[direction] += sign * L[-1]
        points[-1] = self.geom.add_point(x.copy())
        return points

    def _get_gmsh_mesh(self):
        temp = tempfile.NamedTemporaryFile(suffix='.msh')
        try:
            gmsh.write(temp.name)
            gmsh_mesh = meshio.read(temp.name)
        except Exception:
            raise Exception('Error in gmsh write or meshio read')
        finally:
            temp.close()
        return gmsh_mesh

    def write_gmsh_file(self, filename):
        gmsh.write(filename)

    def get_field_data(self):
        return {key: value[0] for key, value in self.field_data.items()}