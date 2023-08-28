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
from cideMOD.cell.warehouse import Warehouse
from cideMOD.cell.variables import ProblemVariables
from cideMOD.cell.components import BatteryCell
from cideMOD.cell.dimensional_analysis import DimensionalAnalysis
from cideMOD.mesh.base_mesher import BaseMesher
from cideMOD.models.PXD.base_model import BasePXDModelOutputs


class ThermalModelOutputs(BasePXDModelOutputs):
    """
    A class that contains the mandatory methods to be overrided
    related to the outputs of :class:`cideMOD.models.ThemalModel`.
    """

    def get_outputs_info(self, warehouse: Warehouse) -> None:
        """
        This method modifies a dictionary containing the information of
        both the global and internal variables that can be provided by
        the thermal model.

        Parameters
        ----------
        warehouse: Warehouse
            Object that postprocess, store and write the outputs.
        """

        # Global variables
        warehouse.add_global_variable_info('T_max', fnc=self.get_max_temperature, default=True,
                                           header="T max [K]")

        # Internal variables
        warehouse.add_internal_variable_info('temperature', subdomains='cell',
                                             dtype='scalar', default=True)
        warehouse.add_internal_variable_info('heat_flux', subdomains='cell', dtype='vector')
        warehouse.add_internal_variable_info('q_ohm_e', subdomains='electrolyte',
                                             dtype='scalar')
        warehouse.add_internal_variable_info('q_ohm_s', subdomains='solid_conductor',
                                             dtype='scalar')
        warehouse.add_internal_variable_info('q_rev', subdomains='electrolyte', dtype='scalar')
        warehouse.add_internal_variable_info('q_irrev', subdomains='electrolyte', dtype='scalar')

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
        # Global variables
        self.temp_ = var.temp_
        self.DA = DA

        # Internal variables
        warehouse.setup_internal_variable('temperature', var.temp)
        if cell.has_collectors:
            warehouse.setup_internal_variable('heat_flux', {
                'anode': var.heat_flux_a,
                'cathode': var.heat_flux_c,
                'separator': var.heat_flux_s,
                'negativeCC': var.heat_flux_ncc,
                'positiveCC': var.heat_flux_pcc
            })
        else:
            warehouse.setup_internal_variable('heat_flux', {
                'anode': var.heat_flux_a,
                'cathode': var.heat_flux_c,
                'separator': var.heat_flux_s
            })
        warehouse.setup_internal_variable('q_ohm_e', {
            'anode': var.q_ohm_e_a,
            'cathode': var.q_ohm_e_c,
            'separator': var.q_ohm_e_s
        })
        if cell.has_collectors:
            warehouse.setup_internal_variable('q_ohm_s', {
                'anode': var.q_ohm_s_a,
                'cathode': var.q_ohm_s_c,
                'negativeCC': var.q_ohm_s_ncc,
                'positiveCC': var.q_ohm_s_pcc
            })
        else:
            warehouse.setup_internal_variable('q_ohm_s', {
                'anode': var.q_ohm_s_a,
                'cathode': var.q_ohm_s_c
            })
        warehouse.setup_internal_variable('q_rev', {
            'anode': var.q_rev_a_total,
            'cathode': var.q_rev_c_total
        })
        warehouse.setup_internal_variable('q_irrev', {
            'anode': var.q_irrev_a_total,
            'cathode': var.q_irrev_c_total
        })

    def get_max_temperature(self):
        # NOTE: Dimensionless state variables will be dolfinx.fem.Function always
        temp_ = self.temp_.vector.array.max()
        return self.DA.unscale_variable('temp', temp_)
