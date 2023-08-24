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
from abc import ABC, abstractmethod
from collections import OrderedDict

from cideMOD.cell.warehouse import Warehouse
from cideMOD.cell.variables import ProblemVariables
from cideMOD.cell.components import BatteryCell
from cideMOD.cell.dimensional_analysis import DimensionalAnalysis
from cideMOD.mesh.base_mesher import BaseMesher


class BaseCellModelOutputs(ABC):
    """
    Base mixin class that contains the mandatory methods to be overrided
    related to the outputs of :class:`cideMOD.models.BaseCellModel`.
    """

    def get_outputs_info(self, warehouse: Warehouse) -> None:
        """
        This method modifies a dictionary containing the information of
        both the global and internal variables that can be outputed by
        this specific model.

        Parameters
        ----------
        warehouse: Warehouse
            Object that postprocess, store and write the outputs.

        Examples
        --------
        To add information about a new global variable:

        >>> warehouse.add_global_variable_info(
            'voltage', fnc = self.get_voltage, default = True,
            header = "Voltage [V]")

        To add information about a new internal variable:

        >>> warehouse.add_internal_variable_info(
            'ionic_current', subdomains = 'electrolyte',
            function_space = 'P1', dtype = 'vector')
        """

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

        Examples
        --------
        To add the source of an specific variable:

        >>> warehouse.setup_internal_variable('temperature', var.temp)

        If it is defined in different subdomains:

        >>> warehouse.setup_internal_variable('phi_s', {
                'anode': var.phi_s,
                'cathode': var.phi_s,
                'negativeCC': var.phi_s_cc,
                'positiveCC': var.phi_s_cc,
            })
        """
        # TODO: In phase 2, this could be done automatically and thus, this method will
        #       be deprecated. Even in phase 1 could be carried out.

    def get_cell_state(self, cell_state: OrderedDict, problem) -> None:
        """
        This method updates the cell state dictionary with the current
        cell state variables of this specific model.

        Parameters
        ----------
        cell_state: OrderedDict
            Dictionary containing the current cell state variables
        problem: Problem
            Object that handles the battery cell simulation.

        Notes
        -----
        The name of the cell state variables should be the ones
        registered in the triggers in order to be detected.
        """
