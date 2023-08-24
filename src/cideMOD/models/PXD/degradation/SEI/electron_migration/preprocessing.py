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
from collections import namedtuple

from cideMOD.numerics.time_scheme import TimeScheme
from cideMOD.numerics.fem_handler import BlockFunction
from cideMOD.cell.components import BatteryCell
from cideMOD.cell.variables import ProblemVariables
from cideMOD.models.PXD.base_model import BasePXDModelPreprocessing


class MigrationSEIModelPreprocessing(BasePXDModelPreprocessing):
    """
    Base mixin class that contains the mandatory methods to be overrided
    related to the preprocessing of the model inputs.
    """
    # ******************************************************************************************* #
    # ***                                       Problem                                       *** #
    # ******************************************************************************************* #

    def set_state_variables(self, state_vars: list, mesher, V, V_vec, problem) -> None:
        """
        This method sets the state variables of the compact SEI
        model.

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
        """
        res = mesher.get_restrictions()

        state_vars.extend([(f'j_sei_a{i}', V.clone(), res.anode)
                          for i in range(problem.cell_parser.anode.n_mat)])
        state_vars.extend([(f'delta_porous_sei_a{i}', V.clone(), res.anode)
                          for i in range(problem.cell_parser.anode.n_mat)])
        state_vars.extend([(f'delta_compact_sei_a{i}', V.clone(), res.anode)
                           for i in range(problem.cell_parser.anode.n_mat)])

        state_var_names = [var_name for var_name, _, _ in state_vars]
        self._state_vars = state_var_names[state_var_names.index('j_sei_a0'):]

    def set_dependent_variables(self, var: ProblemVariables,
                                cell: BatteryCell, DT: TimeScheme, problem):
        """
        This method sets the dependent variables of the electron
        migration SEI model.

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
        # build j_Li
        var._J_Li = namedtuple('J_Li', var._J_Li._fields + ('SEI',))
        electrode = cell.anode
        var.j_Li_a = var._J_Li._make([var.j_Li_a.total, var.j_Li_a.int, var.j_Li_a.LLI,
                                      var.j_Li_a.C_dl, var.j_Li_a.total_0, list()])
        for idx in range(len(electrode.active_materials)):
            # Compact Solid Electrolyte Interphase
            var.j_Li_a.SEI.append(var.f_1(f'j_sei_{electrode.label}{idx}'))
            # Lost of Lithium Inventory
            var.j_Li_a.LLI[idx] += var.j_Li_a.SEI[idx]
            # Total Li flux
            var.j_Li_a.total_0[idx] += var.j_Li_a.SEI[idx]
            var.j_Li_a.total[idx] += var.j_Li_a.SEI[idx]

        # build j_Li term = sum(j_Li * a_s)
        j_Li_term = []
        for ff, field in enumerate(var.j_Li_a._fields):
            j_Li_term.append(0)
            for i, am in enumerate(electrode.active_materials):
                j_Li_term[ff] += var.j_Li_a._asdict()[field][i] * am.a_s
        setattr(var, f'j_Li_{electrode.label}_term', var._J_Li._make(j_Li_term))

        # Overpotential
        var.overpotential_sei = []
        var.delta_porous_sei_a, var.delta_compact_sei_a = [], []
        for i in range(var.n_mat_a):
            delta_porous_sei = var.f_1(f'delta_porous_sei_a{i}')
            var.delta_porous_sei_a.append(delta_porous_sei)
            delta_compact_sei = var.f_1(f'delta_compact_sei_a{i}')
            var.delta_compact_sei_a.append(delta_compact_sei)
            G = (cell.anode.SEI.porous.R
                 + var.delta_porous_sei_a[i] / cell.anode.SEI.porous.kappa)
            var.overpotential_a[i] -= var.j_Li_a.total[i] * G
            # SEI overpotential
            var.overpotential_sei.append(var.f_1.phi_s - var.f_1.phi_e - cell.anode.SEI.compact.U
                                         - var.j_Li_a.total[i] * G)

    def initial_guess(self, f: BlockFunction, var: ProblemVariables,
                      cell: BatteryCell, problem) -> None:
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
        if var.n_mat_a == 0:  # if not cell.anode.is_active:
            return

        P1_map = problem.P1_map
        for material in range(var.n_mat_a):
            P1_map.interpolate({'anode': cell.anode.SEI.compact.delta0},
                               f(f'delta_compact_sei_a{material}'), clear=True)
            P1_map.interpolate({'anode': cell.anode.SEI.porous.delta0},
                               f(f'delta_porous_sei_a{material}'), clear=True)

    def setup(self, problem):
        """
        This method setup the compact SEI model.

        Parameters
        ----------
        problem: Problem
            Object that handles the battery cell simulation.
        """
        self.Q_sei = [0 for _ in range(problem.cell.anode.n_mat)]

    # ******************************************************************************************* #
    # ***                                 DimensionalAnalysis                                 *** #
    # ******************************************************************************************* #

    # ******************************************************************************************* #
    # ***                                     BatteryCell                                     *** #
    # ******************************************************************************************* #

    def set_compactSEI_parameters(self, compact, problem) -> None:
        """
        This method preprocesses the electrode parameters of the
        compact SEI model.

        Parameters
        ----------
        compactSEI: BaseCellComponent
            Object where electrode parameters are preprocessed and
            stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        compact.U = compact.parser.reference_voltage.get_value(problem)
        compact.beta = compact.parser.charge_transfer_coefficient.get_value(problem)
        compact.kappa = compact.parser.electron_conductivity.get_value(problem)
