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
from abc import ABC, abstractmethod

from cideMOD.numerics.time_scheme import TimeScheme
from cideMOD.cell.dimensional_analysis import DimensionalAnalysis
from cideMOD.cell.parser import CellParser
from cideMOD.cell.components import BatteryCell
from cideMOD.cell.variables import ProblemVariables
from cideMOD.mesh.base_mesher import BaseMesher


class BaseCellModelPreprocessing(ABC):
    """
    Base mixin class that contains the mandatory methods to be overrided
    related to the preprocessing of the model inputs.
    """

    # ******************************************************************************************* #
    # ***                                 DimensionalAnalysis                                 *** #
    # ******************************************************************************************* #

    def build_reference_parameters(self, DA: DimensionalAnalysis, cell: CellParser) -> None:
        """
        This method computes the reference parameters that will be used
        to perform the dimensional analysis.

        Parameters
        ----------
        DA: DimensionalAnalysis
            Object where the dimensional analysis is performed.
        cell: CellParser
            Parser of the cell dictionary.
        """

    def build_dimensionless_parameters(self, DA: DimensionalAnalysis, cell: CellParser) -> None:
        """
        This method computes the dimensionless numbers that arise from
        the dimensional analysis.

        Parameters
        ----------
        DA: DimensionalAnalysis
            Object where the dimensional analysis is performed.
        cell: CellParser
            Parser of the cell dictionary.
        """

    def scale_variables(self, variables: dict):
        """
        This method scales the given variables.

        Parameters
        ----------
        variables: Dict[str, Any]
            Dictionary containing the names and the values of the
            variables to be scaled.

        Returns
        -------
        dict
            Dictionary containing the scaled variables. If a variable is
            not recognized, then do not include it.

        Examples
        --------
        >>> variables = {'c_e': 1000, 'c_s_a': 28700}
        >>> models.scale_variables(variables)
        {'c_e': 0, 'c_s_a': 1}
        """
        return {}

    def unscale_variables(self, variables: dict):
        """
        This method unscales the given variables.

        Parameters
        ----------
        variables: Dict[str, Any]
            Dictionary containing the names and the values of the
            variables to be unscaled.

        Returns
        -------
        dict
            Dictionary containing the unscaled variables. If a variable
            is not recognized, then do not include it.

        Examples
        --------
        >>> variables = {'c_e': 0, 'c_s_a': 1}
        >>> models.unscale_variables(variables)
        {'c_e': 1000, 'c_s_a': 28700}
        """
        return {}

    # ******************************************************************************************* #
    # ***                                     BatteryCell                                     *** #
    # ******************************************************************************************* #

    # def set_component_parameters(self, component, problem) -> None:
    #     """
    #     This method preprocesseses the component parameters of this
    #     specific model.

    #     Parameters
    #     ----------
    #     component: BaseCellComponent
    #         Object where the component parameters are preprocessed and
    #         stored.
    #     problem: Problem
    #         Object that handles the battery cell simulation.

    #     Notes
    #     -----
    #     This method is just a template for the family of methods
    #     :meth:`set_{component}_parameters`. The component name
    #     will be given by the components added during the registration
    #     process.
    #     """

    def compute_cell_properties(self, cell: BatteryCell) -> None:
        """
        This method computes the general cell properties of this
        specific model.

        Parameters
        ----------
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.

        Notes
        -----
        This method is called once the cell parameters has been
        preprocessed.
        """

    # ******************************************************************************************* #
    # ***                                       Problem                                       *** #
    # ******************************************************************************************* #

    def set_state_variables(self, state_vars: list, mesher: BaseMesher,
                            V: dfx.fem.FunctionSpace, V_vec: dfx.fem.FunctionSpace,
                            problem) -> None:
        """
        This method sets the state variables of this specific model.

        Parameters
        ----------
        state_vars : List(Tuple(str, numpy.ndarray, dolfinx.fem.FunctionSpace))
            List of tuples, each one containing the name, the
            subdomain and the function space of the state variable.
        mesher : BaseMesher
            Object that contains the mesh information.
        V : dolfinx.fem.FunctionSpace
            Common FunctionSpace to be used for each model.
        V_vec : dolfinx.fem.VectorFunctionSpace
            Common VectorFunctionSpace to be used for each model.
        problem: Problem
            Object that handles the battery cell simulation.

        Examples
        --------
        >>> res = mesher.get_restrictions()
        >>> state_vars.append(('new_var', res.electrolyte, V.clone()))
        """

    def set_problem_variables(self, var: ProblemVariables, DT: TimeScheme, problem) -> None:
        """
        This method sets the problem variables.

        Parameters
        ----------
        var: ProblemVariables
            Object that store the preprocessed problem variables.
        DT: TimeScheme
            Object that provide the temporal derivatives with the
            specified scheme.
        problem: Problem
            Object that handles the battery cell simulation.

        Notes
        -----
        This method is called within :class:`ProblemVariables` right
        after setting up the state variables and before the
        :class:`BatteryCell` is created. In this class is meant to
        create the control variables and those ones that will help
        cell parameters preprocessing.
        """
        # TODO: Extend the documentation with examples.

    def set_dependent_variables(self, var: ProblemVariables, cell: BatteryCell,
                                DT: TimeScheme, problem) -> None:
        """
        This method sets the dependent variables of this specific model.

        Parameters
        ----------
        var: ProblemVariables
            Object that store the preprocessed problem variables.
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
        DT: TimeScheme
            Object that provide the temporal derivatives with the
            specified scheme.
        problem: Problem
            Object that handles the battery cell simulation.
        """

    def initial_guess(self, f, var, cell, problem) -> None:
        """
        This method initializes the state variables based on the initial
        conditions and assuming that the simulation begins after a
        stationary state.

        Parameters
        ----------
        f: BlockFunction
            Block function that contain the state variables to be
            initialized.
        var: ProblemVariables
            Object that store the preprocessed problem variables.
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """

    def setup(self, problem) -> None:
        """
        This method setup this specific model if needed.

        Parameters
        ----------
        problem: Problem
            Object that handles the battery cell simulation.
        """

    def update_control_variables(self, var, problem, **kwargs):
        """
        This method updates the control variables of this specific
        model.

        Parameters
        ----------
        var: ProblemVariables
            Object that store the preprocessed problem variables.
        problem: Problem
            Object that handles the battery cell simulation.
        kwargs: dict
            Dictionary containing the control variables to be updated
            by this specific model.
        """

    def update_reference_values(self, updated_values: dict,
                                cell: CellParser, problem=None) -> None:
        """
        This method updates the reference cell cell properties of this
        specific model.

        Parameters
        ----------
        updated_values: Dict[str, float]
            Dictionary containing the cell parameters that have already
            been updated.
        cell: CellParser
            Parser of the cell dictionary.
        problem: Problem, optional
            Object that handles the battery cell simulation.
        Notes
        -----
        This method is called each time a set of dynamic parameters have
        been updated. If problem is not given, then it is assumed that
        it have not been already defined.
        """

    def reset(self, problem, new_parameters=None, deep_reset=False) -> None:
        """
        This method resets the problem variables related with this
        specific model in order to be ready for running another
        simulation with the same initial conditions, and maybe using
        different parameters.

        Parameters
        ----------
        problem: Problem
            Object that handles the battery cell simulation.
        new_parameters: Dict[str, float], optional
            Dictionary containing the cell parameters that have already
            been updated.
        deep_reset: bool
            Whether or not a deep reset will be performed. It means
            that the Problem setup stage will be run again as the mesh
            has been changed. Default to False.
        """
