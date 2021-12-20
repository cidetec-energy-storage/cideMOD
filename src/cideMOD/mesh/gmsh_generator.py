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
        self.discretization={
            'a': {'n':30, 'type': 'Progression', 'par': 1},
            's': {'n':15, 'type': 'Progression', 'par': 1},
            'c': {'n':30, 'type': 'Progression', 'par': 1},
            'pcc': {'n':10, 'type': 'Progression', 'par': 1},
            'ncc': {'n':10, 'type': 'Progression', 'par': 1},
        }

    def _adapt_discretization(self, structure, L):
        min_cells = {
            'a': 10,
            's': 5,
            'c': 10,
            'pcc': 5,
            'ncc': 5
        }
        for i, el in enumerate(structure):
            size = L[i]/self.discretization[el]['n']
            if size < 2e-6:
                self.discretization[el]['n'] = int(max(min_cells[el], np.ceil(L[i]/2e-6))+1)

            elif size > 2e-6:
                if el in ['a','c']:
                    for r in [1.025, 1.05, 1.075, 1.1, 1.15, 1.2]:
                        n = 2 * np.log(1+(r-1)*L[i]/4e-6)/np.log(r)
                        if n<45:
                            break
                    self.discretization[el]['n']=int(round(n))
                    self.discretization[el]['type'] = 'Bump'
                    self.discretization[el]['par']=r**((-round(n)-1)/2)
                    
    def gmshEnvironment(func):
        def decorated(self, *args, **kwargs):
            self.geom.__enter__()
            gmsh.option.setNumber("General.ExpertMode",1)
            gmsh.option.setNumber("General.Verbosity",0)
            try:
                func(self, *args, **kwargs)
            finally:
                self.geom.__exit__()
        return decorated

    def generate_mesh_from_template(self, filename, output, dim=3, parameters:dict={}):
        gmsh.initialize()
        for key, value in parameters.items():
            if isinstance(value, (float,int)):
                gmsh.onelab.setNumber('Parameters/{}'.format(key),[value])
            elif isinstance(value, (str)):
                gmsh.onelab.setString('Parameters/{}'.format(key),[value])
        gmsh.open(filename)
        gmsh.model.mesh.generate(dim)
        gmsh.write(output)
        gmsh.finalize()

    @gmshEnvironment
    def create_1D_mesh(self, filename:str='', structure = ['a','s','c'], L = [76e-6, 25e-6, 68e-6]):
        self._adapt_discretization(structure, L)
        n_elements = len(structure)
        points = self._draw_point_series(start = [0,0,0], L=L, structure=structure, direction=0)
        
        lines = [0 for i in range(n_elements)]
        for i in range(n_elements):
            lines[i] = self.geom.add_line(points[i], points[i+1]) 
        
        self._label_physical_elements(lines, structure)

        ncc = self.geom.add_physical(points[0],label='negativePlug')
        pcc = self.geom.add_physical(points[-1],label='positivePlug')

        for i in range(n_elements):
            trans_pars = self.discretization[structure[i]]
            self.geom.set_transfinite_curve(lines[i], trans_pars['n'], trans_pars['type'], trans_pars['par'])

        self.geom.generate_mesh(dim = 1, verbose=True)
        if filename:
            self.write_gmsh_file(filename)
        else:
            self.gmsh_mesh = self._get_gmsh_mesh()

    @gmshEnvironment
    def create_2D_mesh(self, filename:str='', structure = ['a','s','c'], L = [76e-6, 25e-6, 68e-6], H = 0.01, nH:int=30):
        n_elements = len(structure)
        self._adapt_discretization(structure, L)
        # Generate geometry
        # Points
        points_down = self._draw_point_series(start = [0,0,0], L=L, structure=structure, direction=0, sign=1)
        points_up = self._draw_point_series(start = [0,H,0], L=L, structure=structure, direction=0, sign=1)
        # Lines
        lines = [0 for i in range(3*(n_elements)+1)]
        for i in range(n_elements):
            lines[3*i] = self.geom.add_line(points_down[i], points_up[i])
            lines[3*i+1] = self.geom.add_line(points_down[i], points_down[i+1])
            lines[3*i+2] = self.geom.add_line(points_up[i+1], points_up[i])
        lines[-1] = self.geom.add_line(points_down[-1], points_up[-1])
        # Curve Loops & Surfaces
        curve_loops = [0 for i in range(n_elements)]
        surfaces = [0 for i in range(n_elements)]
        for i in range(n_elements):
            curve_loops[i] = self.geom.add_curve_loop([lines[3*i+1],lines[3*(i+1)],lines[3*i+2],-lines[3*i]])
            surfaces[i] = self.geom.add_plane_surface(curve_loops[i])

        # Label physical entities
        self._label_physical_elements(surfaces, structure)

        ncc = self.geom.add_physical(lines[0],label='negativePlug')
        pcc = self.geom.add_physical(lines[-1],label='positivePlug')

        # Define mesh
        for i in range(n_elements):
            trans_pars = self.discretization[structure[i]]
            self.geom.set_transfinite_curve(lines[3*i+1], trans_pars['n'], trans_pars['type'], trans_pars['par'])
            self.geom.set_transfinite_curve(lines[3*i+2], trans_pars['n'], trans_pars['type'], trans_pars['par'])
        for i in range(n_elements+1):
            self.geom.set_transfinite_curve(lines[3*i], nH, 'Progression', 1)
        for i in range(n_elements):
            self.geom.set_transfinite_surface(surfaces[i],'Left', [])
        
        # Generate and export mesh
        self.geom.generate_mesh(dim = 2, verbose=True)
        if filename:
            self.write_gmsh_file(filename)
        else:
            self.gmsh_mesh = self._get_gmsh_mesh()

    @gmshEnvironment
    def create_3D_mesh(self, filename:str='', structure = ['a','s','c'], L = [76e-6, 25e-6, 68e-6], H = 0.01, nH:int=10, Z = 0.01, nZ:int=10):
        n_elements = len(structure)
        self._adapt_discretization(structure, L)
        # Check structure structure
        if 'pcc' in structure or 'ncc' in structure:
            assert structure[0] in ('pcc','ncc'), "Current colectors must be at the extremes of the cell"
            assert structure[-1] in ('pcc','ncc'), "Current colectors must be at the extremes of the cell"
        # Generate geometries
        # Points
        points_down_0 = self._draw_point_series(start = [0,0,0], L=L, structure=structure, direction=0, sign=1)
        points_down_1 = self._draw_point_series(start = [0,0,Z], L=L, structure=structure, direction=0, sign=1)
        points_up_0 = self._draw_point_series(start = [0,H,0], L=L, structure=structure, direction=0, sign=1)
        points_up_1 = self._draw_point_series(start = [0,H,Z], L=L, structure=structure, direction=0, sign=1)
        # Lines
        lines_front = [0 for i in range(3*(n_elements)+1)]
        for i in range(n_elements):
            lines_front[3*i] = self.geom.add_line(points_down_0[i], points_up_0[i])
            lines_front[3*i+1] = self.geom.add_line(points_down_0[i], points_down_0[i+1])
            lines_front[3*i+2] = self.geom.add_line(points_up_0[i], points_up_0[i+1])
        lines_front[-1] = self.geom.add_line(points_down_0[-1], points_up_0[-1])

        lines_back = [0 for i in range(3*(n_elements)+1)]
        for i in range(n_elements):
            lines_back[3*i] = self.geom.add_line(points_down_1[i], points_up_1[i])
            lines_back[3*i+1] = self.geom.add_line(points_down_1[i], points_down_1[i+1])
            lines_back[3*i+2] = self.geom.add_line(points_up_1[i], points_up_1[i+1])
        lines_back[-1] = self.geom.add_line(points_down_1[-1], points_up_1[-1])

        lines_joint = [0 for i in range(2*(n_elements+1))]
        for i in range(n_elements+1):
            lines_joint[2*i] = self.geom.add_line(points_down_0[i], points_down_1[i])
            lines_joint[2*i+1] = self.geom.add_line(points_up_0[i], points_up_1[i])
        
        # Curve Loops & Surfaces
        curve_loops = [0 for i in range(5*n_elements+1)]
        surfaces = [0 for i in range(5*n_elements+1)]
        for i in range(n_elements):
            curve_loops[5*i] = self.geom.add_curve_loop([-lines_joint[2*i],lines_front[3*i],lines_joint[2*i+1],-lines_back[3*i]]) 
            curve_loops[5*i+1] = self.geom.add_curve_loop([-lines_back[3*i+1],lines_back[3*i],lines_back[3*i+2],-lines_back[3*(i+1)]])
            curve_loops[5*i+2] = self.geom.add_curve_loop([-lines_front[3*i+1],lines_joint[2*i],lines_back[3*i+1],-lines_joint[2*(i+1)]])
            curve_loops[5*i+3] = self.geom.add_curve_loop([lines_front[3*i+2],lines_joint[2*(i+1)+1],-lines_back[3*i+2],-lines_joint[2*i+1]])
            curve_loops[5*i+4] = self.geom.add_curve_loop([lines_front[3*i+1],lines_front[3*(i+1)],-lines_front[3*i+2],-lines_front[3*i]])
            surfaces[5*i] = self.geom.add_surface(curve_loops[5*i])
            surfaces[5*i+1] = self.geom.add_surface(curve_loops[5*i+1])
            surfaces[5*i+2] = self.geom.add_surface(curve_loops[5*i+2])
            surfaces[5*i+3] = self.geom.add_surface(curve_loops[5*i+3])
            surfaces[5*i+4] = self.geom.add_surface(curve_loops[5*i+4])
        curve_loops[5*n_elements] = self.geom.add_curve_loop([-lines_joint[2*n_elements],lines_front[3*n_elements],lines_joint[2*n_elements+1],-lines_back[3*n_elements]])
        surfaces[5*n_elements] = self.geom.add_surface(curve_loops[5*n_elements])

        # Surface Loops & Volumes
        surface_loops = [0 for i in range(n_elements)]
        volumes = [0 for i in range(n_elements)]
        for i in range(n_elements):
            surface_loops[i] = self.geom.add_surface_loop([surfaces[5*i+j] for j in range(6)])
            volumes[i] = self.geom.add_volume(surface_loops[i])

        # Label physical entities
        self._label_physical_elements(volumes, structure)

        ncc = self.geom.add_physical(surfaces[0],label='negativePlug')
        pcc = self.geom.add_physical(surfaces[-1],label='positivePlug')

        # Define mesh
        for i in range(n_elements):
            trans_pars = self.discretization[structure[i]]
            self.geom.set_transfinite_curve(lines_front[3*i+1], trans_pars['n'], trans_pars['type'], trans_pars['par'])
            self.geom.set_transfinite_curve(lines_front[3*i+2], trans_pars['n'], trans_pars['type'], trans_pars['par'])
            self.geom.set_transfinite_curve(lines_back[3*i+1], trans_pars['n'], trans_pars['type'], trans_pars['par'])
            self.geom.set_transfinite_curve(lines_back[3*i+2], trans_pars['n'], trans_pars['type'], trans_pars['par'])
        for i in range(n_elements+1):
            self.geom.set_transfinite_curve(lines_front[3*i], nH, 'Progression', 1)
            self.geom.set_transfinite_curve(lines_back[3*i], nH, 'Progression', 1)
            self.geom.set_transfinite_curve(lines_joint[2*i], nZ, 'Progression', 1)
            self.geom.set_transfinite_curve(lines_joint[2*i+1], nZ, 'Progression', 1)            
        for surf in surfaces:
            self.geom.set_transfinite_surface(surf,'Left', [])
        for vol in volumes:
            self.geom.set_transfinite_volume(vol, [])
        self.geom.generate_mesh(dim = 3, verbose=True)
        if filename:
            self.write_gmsh_file(filename)
        else:
            self.gmsh_mesh = self._get_gmsh_mesh()

    @gmshEnvironment
    def create_3D_mesh_with_tabs(self, filename:str='', structure = ['a','s','c'], L = [76e-6, 25e-6, 68e-6], H = 0.01, nH:int=10, Z = 0.01, nZ:int=10, tab_locations=[('up','left'),('up','right')]):
        n_elements = len(structure)
        self._adapt_discretization(structure, L)
        # Check structure structure
        if 'pcc' in structure or 'ncc' in structure:
            assert structure[0] in ('pcc','ncc'), "Current colectors must be at the extremes of the cell"
            assert structure[-1] in ('pcc','ncc'), "Current colectors must be at the extremes of the cell"
            tab_indexes = []
            for i, element in enumerate(structure):
                if element in ('ncc','pcc'):
                    h_ind, z_ind = self._get_tab_location( tab_locations[0 if element=='ncc' else 1])
                    tab_indexes.append( (h_ind, z_ind, i, element) )

        # Generate geometries
        # Points
        distribution = [0-2/10, 0, 1/10, 4/10, 6/10, 9/10, 1, 1+2/10]
        points = [[0 for j in range(6)] for i in range(6)]
        for i in range(6):
            for j in range(6):
                points[i][j] = self._draw_point_series(start = [0,distribution[i+1]*H,distribution[j+1]*Z], L=L, structure=structure, direction=0, sign=1)
        # - Tab points
        if 'pcc' in structure or 'ncc' in structure:
            tab_points = [[0 for i in range(4)] for ind in tab_indexes]
            for i, tab in enumerate(tab_indexes):
                if tab[0] in [0,7]:
                    tab_points[i][0] = self.geom.add_point([sum([L[j] for j in range(tab[2])]), H*distribution[tab[0]], Z*distribution[tab[1]]], L[tab[2]]/10)
                    tab_points[i][1] = self.geom.add_point([sum([L[j] for j in range(tab[2])]), H*distribution[tab[0]], Z*distribution[tab[1]+1]], L[tab[2]]/10)
                    tab_points[i][2] = self.geom.add_point([sum([L[j] for j in range(tab[2]+1)]), H*distribution[tab[0]], Z*distribution[tab[1]]], L[tab[2]]/10)
                    tab_points[i][3] = self.geom.add_point([sum([L[j] for j in range(tab[2]+1)]), H*distribution[tab[0]], Z*distribution[tab[1]+1]], L[tab[2]]/10)
                else:
                    tab_points[i][0] = self.geom.add_point([sum([L[j] for j in range(tab[2])]), H*distribution[tab[0]], Z*distribution[tab[1]]], L[tab[2]]/10)
                    tab_points[i][1] = self.geom.add_point([sum([L[j] for j in range(tab[2])]), H*distribution[tab[0]+1], Z*distribution[tab[1]]], L[tab[2]]/10)
                    tab_points[i][2] = self.geom.add_point([sum([L[j] for j in range(tab[2]+1)]), H*distribution[tab[0]], Z*distribution[tab[1]]], L[tab[2]]/10)
                    tab_points[i][3] = self.geom.add_point([sum([L[j] for j in range(tab[2]+1)]), H*distribution[tab[0]+1], Z*distribution[tab[1]]], L[tab[2]]/10)

        # Lines: [transversal, vertical, horizontal]
        lines = [
            [ [ [0 for k in range(n_elements)] for j in range(6)] for i in range(6) ],  # Transversal
            [ [ [0 for k in range(n_elements+1)] for j in range(6)] for i in range(5) ],  # Vertical, in-plane
            [ [ [0 for k in range(n_elements+1)] for j in range(5)] for i in range(6) ]  # Horizontal, in-plane
        ]
        for i in range(6):
            for j in range(6):
                for k in range(n_elements):
                    lines[0][i][j][k] = self.geom.add_line(points[i][j][k], points[i][j][k+1])
        for i in range(5):
            for j in range(6):
                for k in range(n_elements+1):
                    lines[1][i][j][k] = self.geom.add_line(points[i][j][k], points[i+1][j][k])
        for i in range(6):
            for j in range(5):
                for k in range(n_elements+1):
                    lines[2][i][j][k] = self.geom.add_line(points[i][j][k], points[i][j+1][k])
        
        # - Tab lines
        if 'pcc' in structure or 'ncc' in structure:
            tab_lines = [[0 for i in range(8)] for ind in tab_indexes]
            for i, tab in enumerate(tab_indexes):
                if tab[0] in [0,7]:
                    h = max(tab[0]-2,0)
                    tab_lines[i][0] = self.geom.add_line( tab_points[i][0], tab_points[i][1])
                    tab_lines[i][1] = self.geom.add_line( tab_points[i][0], tab_points[i][2])
                    tab_lines[i][2] = self.geom.add_line( tab_points[i][2], tab_points[i][3])
                    tab_lines[i][3] = self.geom.add_line( tab_points[i][1], tab_points[i][3])
                    
                    tab_lines[i][4] = self.geom.add_line( points[h][tab[1]-1][tab[2]], tab_points[i][0])
                    tab_lines[i][5] = self.geom.add_line( points[h][tab[1]][tab[2]], tab_points[i][1])
                    tab_lines[i][6] = self.geom.add_line( points[h][tab[1]-1][tab[2]+1], tab_points[i][2])
                    tab_lines[i][7] = self.geom.add_line( points[h][tab[1]][tab[2]+1], tab_points[i][3])
                else:
                    z = max(tab[0]-2,0)
                    tab_lines[i][0] = self.geom.add_line( tab_points[i][0], tab_points[i][1])
                    tab_lines[i][1] = self.geom.add_line( tab_points[i][0], tab_points[i][2])
                    tab_lines[i][2] = self.geom.add_line( tab_points[i][2], tab_points[i][3])
                    tab_lines[i][3] = self.geom.add_line( tab_points[i][1], tab_points[i][3])
                    
                    tab_lines[i][4] = self.geom.add_line( points[tab[0]-1][z][tab[2]], tab_points[i][0])
                    tab_lines[i][5] = self.geom.add_line( points[tab[0]][z][tab[2]], tab_points[i][1])
                    tab_lines[i][6] = self.geom.add_line( points[tab[0]-1][z][tab[2]+1], tab_points[i][2])
                    tab_lines[i][7] = self.geom.add_line( points[tab[0]][z][tab[2]+1], tab_points[i][3])
                    
        #Curve_loops & Surfaces
        curve_loops = [
            [ [ [0 for k in range(n_elements+1)] for j in range(5)] for i in range(5) ],  # In-plane (Front-back)
            [ [ [0 for k in range(n_elements)] for j in range(5)] for i in range(6) ],  # Up-Down (H+, H-)
            [ [ [0 for k in range(n_elements)] for j in range(6)] for i in range(5) ]  # Sides (Z+, Z-)
        ]
        surfaces = curve_loops.copy()

        for i in range(5):
            for j in range(5):
                for k in range(n_elements+1):
                    curve_loops[0][i][j][k] = self.geom.add_curve_loop([ lines[2][i][j][k], lines[1][i][j+1][k], -lines[2][i+1][j][k], -lines[1][i][j][k] ])
                    surfaces[0][i][j][k] = self.geom.add_plane_surface( curve_loops[0][i][j][k])
        for i in range(6):
            for j in range(5):
                for k in range(n_elements):
                    curve_loops[1][i][j][k] = self.geom.add_curve_loop([ lines[0][i][j][k], lines[2][i][j][k+1], -lines[0][i][j+1][k], -lines[2][i][j][k] ])
                    surfaces[1][i][j][k] = self.geom.add_plane_surface( curve_loops[1][i][j][k])
        for i in range(5):
            for j in range(6):
                for k in range(n_elements):
                    curve_loops[2][i][j][k] = self.geom.add_curve_loop([ lines[1][i][j][k], lines[0][i+1][j][k], -lines[1][i][j][k+1], -lines[0][i][j][k] ])
                    surfaces[2][i][j][k] = self.geom.add_plane_surface( curve_loops[2][i][j][k])

        # - Tab curve loops & surfaces
        if 'pcc' in structure or 'ncc' in structure:
            tab_curve_loops = [[0 for i in range(5)] for ind in tab_indexes]
            tab_surfaces = tab_curve_loops.copy()
            for i, tab in enumerate(tab_indexes):
                if tab[0] in [0,7]:
                    h = max(tab[0]-2,0)
                    tab_curve_loops[i][0] = self.geom.add_curve_loop([ tab_lines[i][1], tab_lines[i][2], -tab_lines[i][3], -tab_lines[i][0] ]) # Top surface

                    tab_curve_loops[i][1] = self.geom.add_curve_loop([ -tab_lines[i][0], -tab_lines[i][4] , lines[2][h][tab[1]-1][tab[2]], tab_lines[i][5] ]) # rear
                    tab_curve_loops[i][2] = self.geom.add_curve_loop([ tab_lines[i][1], -tab_lines[i][6], -lines[0][h][tab[1]-1][tab[2]], tab_lines[i][4] ]) # back
                    tab_curve_loops[i][3] = self.geom.add_curve_loop([ -tab_lines[i][2], -tab_lines[i][6], lines[2][h][tab[1]-1][tab[2]+1], tab_lines[i][7] ]) # side
                    tab_curve_loops[i][4] = self.geom.add_curve_loop([ tab_lines[i][3], -tab_lines[i][7], -lines[0][h][tab[1]][tab[2]], tab_lines[i][5] ]) # front

                    for j in range(5):
                        tab_surfaces[i][j] = self.geom.add_plane_surface(tab_curve_loops[i][j])

        # Surface Loops & Volumes
        surface_loops = [ [[0 for k in range(n_elements)] for j in range(5)] for i in range(5)]
        volumes = surface_loops.copy()
        for i in range(5):
            for j in range(5):
                for k in range(n_elements):
                    surface_loops[i][j][k] = self.geom.add_surface_loop([ surfaces[0][i][j][k], surfaces[1][i][j][k], surfaces[2][i][j][k], surfaces[0][i][j][k+1], surfaces[1][i+1][j][k], surfaces[2][i][j+1][k] ])
                    volumes[i][j][k] = self.geom.add_volume(surface_loops[i][j][k])

        # - Tab surface loops & volumes
        if 'pcc' in structure or 'ncc' in structure:
            tab_surface_loops = [0 for ind in tab_indexes]
            tab_volumes = tab_surface_loops.copy()
            for i, tab in enumerate(tab_indexes):
                tab_surface_loops[i] = self.geom.add_surface_loop([tab_surfaces[i][j] for j in range(5)] + [surfaces[1][max(tab[0]-2,0)][tab[1]-1][tab[2]]])
                tab_volumes[i] = self.geom.add_volume(tab_surface_loops[i])

        # Re-structure volumes array to label according element number
        volumes_to_label = [[[volumes[i][j][k] for i in range(5)] for j in range(5)] for k in range(n_elements)]
        if 'pcc' in structure or 'ncc' in structure:
            for i, tab in enumerate(tab_indexes):
                volumes_to_label[tab[2]].append(tab_volumes[i])

        # Label physical entities
        self._label_physical_elements(volumes_to_label, structure)

        if 'pcc' in structure or 'ncc' in structure:
            ncc = self.geom.add_physical(self._flatten_list([ tab_surfaces[i][j] for j in range(5) for i, tab in enumerate(tab_indexes) if tab[3] == 'ncc' ]), label='negativePlug')
            pcc = self.geom.add_physical(self._flatten_list([ tab_surfaces[i][j] for j in range(5) for i, tab in enumerate(tab_indexes) if tab[3] == 'pcc' ]), label='positivePlug')
        else:
            ncc = self.geom.add_physical(self._flatten_list([surfaces[0][i][j][0] for i in range(5) for j in range(5)]),label='negativePlug')
            pcc = self.geom.add_physical(self._flatten_list([surfaces[0][i][j][-1] for i in range(5) for j in range(5)]),label='positivePlug')

        # Define mesh
        H_disc = [int(max(1,np.ceil(nH/10))), int(max(2,np.ceil(nH*3/10))), int(max(1,np.ceil(nH*2/10))), int(max(2,np.ceil(nH*3/10))), int(max(1,np.ceil(nH/10)))]
        Z_disc = [int(max(1,np.ceil(nZ/10))), int(max(2,np.ceil(nZ*3/10))), int(max(1,np.ceil(nZ*2/10))), int(max(2,np.ceil(nZ*3/10))), int(max(1,np.ceil(nZ/10)))]
        for i in range(6):
            for j in range(6):
                for k in range(n_elements):
                    trans_pars = self.discretization[structure[k]]
                    self.geom.set_transfinite_curve(lines[0][i][j][k], trans_pars['n'], trans_pars['type'], trans_pars['par'])
        for i in range(5):
            for j in range(6):
                for k in range(n_elements+1):
                    self.geom.set_transfinite_curve(lines[1][i][j][k], H_disc[i], 'Progression', 1)
        for i in range(6):
            for j in range(5):
                for k in range(n_elements+1):
                    self.geom.set_transfinite_curve(lines[2][i][j][k], Z_disc[j], 'Progression', 1)

        all_surfaces = self._flatten_list(surfaces)
        for surf in all_surfaces:
            self.geom.set_transfinite_surface(surf,'Left', []) 
        all_volumes = self._flatten_list(volumes)
        for vol in all_volumes:
            self.geom.set_transfinite_volume(vol, [])

        # - Tab mesh definition
        if 'pcc' in structure or 'ncc' in structure:
            for i, tab in enumerate(tab_indexes):
                trans_pars = self.discretization[structure[tab[2]]]
                if tab[0] in (7,0):
                    self.geom.set_transfinite_curve( tab_lines[i][0], Z_disc[1], 'Progression', 1 )
                    self.geom.set_transfinite_curve( tab_lines[i][1], trans_pars['n'], trans_pars['type'], trans_pars['par'])
                    self.geom.set_transfinite_curve( tab_lines[i][2], Z_disc[1], 'Progression', 1 )
                    self.geom.set_transfinite_curve( tab_lines[i][3], trans_pars['n'], trans_pars['type'], trans_pars['par'])
                    for j in range(4):
                        self.geom.set_transfinite_curve( tab_lines[i][4+j], H_disc[2], 'Progression', 1 )
                else:
                    self.geom.set_transfinite_curve( tab_lines[i][0], H_disc[1], 'Progression', 1 )
                    self.geom.set_transfinite_curve( tab_lines[i][1], trans_pars['n'], trans_pars['type'], trans_pars['par'])
                    self.geom.set_transfinite_curve( tab_lines[i][2], H_disc[1], 'Progression', 1 )
                    self.geom.set_transfinite_curve( tab_lines[i][3], trans_pars['n'], trans_pars['type'], trans_pars['par'])
                    for j in range(4):
                        self.geom.set_transfinite_curve( tab_lines[i][4+j], Z_disc[2], 'Progression', 1 )
                for j in range(5):
                    self.geom.set_transfinite_surface(tab_surfaces[i][j], 'Left', [])
                self.geom.set_transfinite_volume(vol, [])
        self.geom.generate_mesh(dim = 3, verbose=True)
        if filename:
            self.write_gmsh_file(filename)
        else:
            self.gmsh_mesh = self._get_gmsh_mesh()

    def _label_physical_elements(self, elements:list, structure:list):
        assert len(elements) == len(structure), "Element number incorrect" 
        keys = np.array(structure)
        ind = np.arange(len(structure))
        key_dict = {}
        for k in set(keys):
            key_dict[k] = ind[k==keys]

        for k in key_dict:
            objects = []
            for index in key_dict[k]:
                flattened_elements = self._flatten_list(elements[index])
                for item in flattened_elements:
                    objects.append(item)
            self.geom.add_physical(objects,label=self.labels[k])

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
                'down': (4, 7),
            }
        }
        return locations[tab_location[0]][tab_location[1]]

    def _draw_point_series(self, start:list, L:list, structure:list, direction:int, sign:int=1):
        assert len(start) in (2,3), "Start point must have 2 or 3 coordinates"
        assert direction < len(start)
        assert sign in (1, -1) 
        lcar = max(L)
        n_elem = len(L)
        points = [0 for i in range(n_elem+1)]
        x = start
        points[0] = self.geom.add_point(x.copy(), min(lcar,L[0]/self.discretization[structure[0]]['n']))
        for i in range(n_elem-1):
            x[direction] += sign * L[i]
            points[i+1] = self.geom.add_point(x.copy(), min(lcar,L[i]/self.discretization[structure[i]]['n'], L[i+1]/self.discretization[structure[i+1]]['n'])) 
        x[direction] += sign * L[-1]
        points[-1] = self.geom.add_point(x.copy(), min(lcar,L[-1]/self.discretization[structure[-1]]['n']))
        return points

    def custom_mesh(self, file_name):
        self.gmsh_mesh = meshio.read(file_name)
        self.write_to_fenics()
        
    def _get_gmsh_mesh(self):
        temp = tempfile.NamedTemporaryFile(suffix='.msh')
        try:
            gmsh.write(temp.name)
            gmsh_mesh = meshio.read(temp.name)
        except:
            raise Exception('Error in gmsh write or meshio read')
        finally:
            temp.close()
        return gmsh_mesh

    def write_gmsh_file(self, filename):
        gmsh.write(filename)

    def get_field_data(self):
        return {key: value[0] for key, value in self.field_data.items()}


