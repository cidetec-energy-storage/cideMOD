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

from cideMOD.cell.warehouse import Warehouse
from cideMOD.cell.variables import ProblemVariables
from cideMOD.cell.components import BatteryCell
from cideMOD.cell.dimensional_analysis import DimensionalAnalysis
from cideMOD.mesh.base_mesher import BaseMesher
from cideMOD.models.PXD.base_model import BasePXDModelOutputs


class DiffusionSEIModelOutputs(BasePXDModelOutputs):
    """
    A class that contains the mandatory methods to be overrided
    related to the outputs of :class:`cideMOD.models.ElectrochemicalModel`.
    """

    def get_outputs_info(self, warehouse: Warehouse) -> None:
        """
        This method modifies a dictionary containing the information of
        both the global and internal variables that can be outputed by
        the SEI model.

        Parameters
        ----------
        warehouse: Warehouse
            Object that postprocess, store and write the outputs.
        """
        # Global variables
        header = "Instantaneous capacity loss to SEI {i} [Ah]"
        warehouse.add_global_variable_info('Q_sei_instant_a', fnc=self.get_Q_sei_instant,
                                           default=False, dtype='list_of_scalar', header=header)
        header = "Average SEI {i} thickness [m]"
        warehouse.add_global_variable_info('delta_sei_a', fnc=self.get_L_sei,
                                           default=True, dtype='list_of_scalar', header=header)
        header = "Capacity loss to SEI {i} [Ah]"
        warehouse.add_global_variable_info('Q_sei_a', fnc=self.get_Q_sei,
                                           default=True, dtype='list_of_scalar', header=header)

        # Internal variables
        warehouse.add_internal_variable_info('c_EC_0_a', subdomains='anode',
                                             dtype='list_of_scalar', default=False)
        warehouse.add_internal_variable_info('delta_sei_a', subdomains='anode',
                                             dtype='list_of_scalar', default=True)
        warehouse.add_internal_variable_info('j_sei_a', subdomains='anode',
                                             dtype='list_of_scalar', default=True)

    def prepare_outputs(self, warehouse: Warehouse, var: ProblemVariables, cell: BatteryCell,
                        mesher: BaseMesher, DA: DimensionalAnalysis, problem) -> None:
        """
        This method computes the expression of the requested internal
        variables to be ready for being evaluated and stored.

        Parameters
        ----------
        warehouse: Warehouse
            Object that postprocess, store and write the outputs.
        var: ProblemVariables
            Object containing the problem variables.
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
        mesher: BaseMesher
            Object that store the mesh information.
        DA: DimensionalAnalysis
            Object where the dimensional analysis is performed.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        # Prepare global variables
        self.problem = problem  # Needed within global variables methods
        self.mesher = mesher
        self.cell = cell

        self.dx = problem.mesher.dx_a
        self._j_instant_sei_forms, self._delta_sei_forms = [], []
        for k, am in enumerate(cell.anode.active_materials):
            j_instant_sei = (var.f_1(f'j_sei_a{k}') + var.f_0(f'j_sei_a{k}')) / 2
            self._j_instant_sei_forms.append(dfx.fem.form(j_instant_sei * self.dx))
            self._delta_sei_forms.append(dfx.fem.form(var.f_1(f'delta_sei_a{k}') * self.dx))

        # Prepare internal variables
        warehouse.setup_internal_variable('j_sei_a', var.j_Li_a.SEI, length=var.n_mat_a)
        warehouse.setup_internal_variable('delta_sei_a', var.delta_sei_a, length=var.n_mat_a)
        warehouse.setup_internal_variable('c_EC_0_a', var.c_EC_0_a, length=var.n_mat_a)

    def get_Q_sei_instant(self):
        Q_sei_instant = []
        for am_idx, am in enumerate(self.cell.anode.active_materials):
            volume = self.cell.anode.L * self.cell.area * self.cell.anode.N
            j_instant_sei_am = self.problem.get_avg(self._j_instant_sei_forms[am_idx], self.dx)
            Q_sei_instant.append(-self.problem.get_timestep()
                                 * j_instant_sei_am * am.a_s * volume / 3600)
        return Q_sei_instant

    def get_L_sei(self):
        return [self.problem.get_avg(form, self.dx) for form in self._delta_sei_forms]

    def get_Q_sei(self):
        return self.Q_sei.copy()
