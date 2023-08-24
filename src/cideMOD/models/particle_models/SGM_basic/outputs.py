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


class ParticleModelSGMOutputs(BasePXDModelOutputs):
    """
    A class that contains the mandatory methods to be overrided
    related to the outputs of :class:`cideMOD.models.ParticlelModel`.
    """

    def get_outputs_info(self, warehouse: Warehouse) -> None:
        """
        This method modifies a dictionary containing the information of
        both the global and internal variables that can be outputed by
        the particle model.

        Parameters
        ----------
        warehouse: Warehouse
            Object that postprocess, store and write the outputs.
        """
        # Global variables
        warehouse.add_global_variable_info('cathode_SOC', fnc=self.get_soc_c, default=True,
                                           header="Cathode AM {i} SOC [-]", dtype='list_of_scalar')
        warehouse.add_global_variable_info('total_lithium', fnc=self.calculate_total_lithium,
                                           header="Total_lithium [mol]")

        # Internal variables
        for component in warehouse.problem.cell_parser._components_.values():
            if not component.name == 'electrode':
                continue
            label = component.label
            tag = component.tag
            warehouse.add_internal_variable_info(f'c_s_{label}_surf', subdomains=tag,
                                                 dtype='list_of_scalar')
            warehouse.add_internal_variable_info(f'c_s_{label}_avg', subdomains=tag,
                                                 dtype='list_of_scalar')
            warehouse.add_internal_variable_info(f'x_{label}_surf', subdomains=tag,
                                                 dtype='list_of_scalar')
            warehouse.add_internal_variable_info(f'x_{label}_avg', subdomains=tag,
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
        self.problem = problem
        self.mesher = mesher
        self.cell = cell
        self.var = var
        self._SoC_forms = [dfx.fem.form(x * mesher.dx_c) for x in var.x_c_avg]

       # Prepare internal variables
        for component in cell._components_.values():
            if not component.name == 'electrode':
                continue
            label = component.label
            n_mat = component.n_mat
            warehouse.setup_internal_variable(
                f'c_s_{label}_surf', var(f'c_s_{label}_surf'), length=n_mat)
            warehouse.setup_internal_variable(
                f'c_s_{label}_avg', var(f'c_s_{label}_avg'), length=n_mat)
            warehouse.setup_internal_variable(
                f'x_{label}_surf', var(f'x_{label}_surf'), length=n_mat)
            warehouse.setup_internal_variable(
                f'x_{label}_avg', var(f'x_{label}_avg'), length=n_mat)

    def get_soc_c(self):
        dx = self.mesher.dx_c
        return [self.problem.get_avg(form, dx) for form in self._SoC_forms]

    def _get_electrode_lithium(self, electrode, dx):
        c_s_avg = getattr(self.var, f'c_s_{electrode.label}_avg')
        li_total = 0
        for k, material in enumerate(electrode.active_materials):
            li_total += c_s_avg[k] * material.eps_s
        return self.problem.get_avg(li_total, dx)

    def calculate_total_lithium(self):
        # TODO: Make this more efficient compiling the form within self.prepare_outputs

        d = self.mesher.get_measures()

        internal_li = self.cell.anode.L * self._get_electrode_lithium(self.cell.anode, d.x_a)
        internal_li += self.cell.cathode.L * self._get_electrode_lithium(self.cell.cathode, d.x_c)

        disolved_li = self.problem.get_avg(
            self.cell.anode.eps_e * self.cell.anode.L * self.var.c_e, d.x_a)
        disolved_li += self.problem.get_avg(
            self.cell.separator.eps_e * self.cell.separator.L * self.var.c_e * d.x_s)
        disolved_li += self.problem.get_avg(
            self.cell.cathode.eps_e * self.cell.cathode.L * self.var.c_e * d.x_c)

        total_li = (internal_li + disolved_li) * self.cell.area
        return total_li
