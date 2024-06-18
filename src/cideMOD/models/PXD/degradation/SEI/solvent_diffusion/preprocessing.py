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
import numpy as np
from numpy.polynomial.polynomial import polyval, polymul, polyint
from collections import namedtuple

from cideMOD.numerics.time_scheme import TimeScheme
from cideMOD.numerics.fem_handler import BlockFunction
from cideMOD.cell.components import BatteryCell
from cideMOD.cell.variables import ProblemVariables
from cideMOD.models.PXD.base_model import BasePXDModelPreprocessing
from cideMOD.numerics.polynomials import Lagrange


class DiffusionSEIModelPreprocessing(BasePXDModelPreprocessing):
    """
    Base mixin class that contains the mandatory methods to be overrided
    related to the preprocessing of the model inputs.
    """
    # ******************************************************************************************* #
    # ***                                       Problem                                       *** #
    # ******************************************************************************************* #

    def set_state_variables(self, state_vars: list, mesher, V, V_vec, problem) -> None:
        """
        This method sets the state variables of the SEI
        solvent-diffusion model.

        Parameters
        ----------
        state_vars : List(Tuple(str, numpy.ndarray, dolfinx.fem.functionspace))
            List of tuples, each one containing the name, the
            subdomain and the function space of the state variable.

        mesher : BaseMesher
            Object that contains the mesh information.

        V : dolfinx.fem.functionspace
            Common functionspace to be used for each model.

        V_vec : dolfinx.fem.VectorFunctionSpace
            Common VectorFunctionSpace to be used for each model.
        """
        res = mesher.get_restrictions()

        state_vars.extend([(f'j_sei_a{i}', V.clone(), res.anode)
                          for i in range(problem.cell_parser.anode.n_mat)])
        state_vars.extend([(f'delta_sei_a{i}', V.clone(), res.anode)
                          for i in range(problem.cell_parser.anode.n_mat)])

        for k in range(problem.cell_parser.anode.n_mat):
            for j in range(self.order):
                var_name = f'c_EC_{j}_a{k}'
                state_vars.append((var_name, V.clone(), res.anode))

        state_var_names = [var_name for var_name, _, _ in state_vars]
        self._state_vars = state_var_names[state_var_names.index('j_sei_a0'):]

    def set_dependent_variables(self, var: ProblemVariables,
                                cell: BatteryCell, DT: TimeScheme, problem):
        """
        This method sets the dependent variables of the SEI
        solvent-diffusion model.

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
            # Solid Electrolyte Interphase
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
        var.delta_sei_a = []
        # EC Concentration
        var.c_EC_0_a = []
        for i in range(var.n_mat_a):
            c_EC = var.f_1(f'c_EC_0_a{i}')
            var.c_EC_0_a.append(c_EC)
            delta_sei = var.f_1(f'delta_sei_a{i}')
            var.delta_sei_a.append(delta_sei)
            G = cell.anode.SEI.porous.R + delta_sei / cell.anode.SEI.porous.kappa
            var.overpotential_a[i] -= var.j_Li_a.total[i] * G
            # SEI overpotential
            var.overpotential_sei.append(var.f_1.phi_s - var.f_1.phi_e - cell.anode.SEI.porous.U
                                         - var.j_Li_a.total[i] * G)

        # FIXME: Implement var.overpotential_a_0 variable to use it in the transitory and
        #        stationary electrochemical equations.

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
        c_0 = cell.anode.SEI.porous.c_EC_sln * cell.anode.SEI.porous.eps
        for material in range(var.n_mat_a):
            for j in range(self.order):
                P1_map.interpolate({'anode': c_0}, f(f'c_EC_{j}_a{material}'), clear=True)
            P1_map.interpolate(
                {'anode': cell.anode.SEI.porous.delta0}, f(f'delta_sei_a{material}'), clear=True)

    def setup(self, problem):
        """
        This method setup the SEI solvent-diffusion model.

        Parameters
        ----------
        problem: Problem
            Object that handles the battery cell simulation.
        """
        self.Q_sei = [0 for _ in range(problem.cell.anode.n_mat)]

    def _build_lagrange(self):
        """
        Builds mass matrix, stiffness matrix and boundary vector using
        Legendre Polinomials. The domain used is [0,1] and only pair
        Legendre polinomials are used to enforce zero flux at x=0.

        Args:
            order (int): number of Legendre polinomials to use

        Returns:
            tuple: Mass matrix, Stiffness matrix, boundary vector
        """
        self.poly = Lagrange(self.order)
        self.f = self.poly.f
        self.df = self.poly.df
        self.xf = self.poly.xf
        self.xdf = self.poly.xdf

        J = np.zeros((self.order + 1, self.order + 1))
        H = np.zeros((self.order + 1, self.order + 1))
        K = np.zeros((self.order + 1, self.order + 1))
        L = np.zeros((self.order + 1, self.order + 1))
        M = np.zeros((self.order + 1, self.order + 1))
        N = np.zeros((self.order + 1, self.order + 1))
        P = np.zeros(self.order + 1)

        for i in range(self.order + 1):

            P[i] = polyval(0, self.f[i])

            for j in range(self.order + 1):

                J[i, j] = polyval(0, polymul(self.f[i], self.f[j]))
                H[i, j] = polyval(1, polymul(self.df[i], self.f[j]))
                K[i, j] = polyval(1, polyint(polymul(self.f[i], self.f[j])))
                L[i, j] = polyval(1, polyint(polymul(self.df[i], self.f[j])))
                M[i, j] = polyval(1, polyint(polymul(self.xdf[i], self.f[j])))
                N[i, j] = polyval(1, polyint(polymul(self.df[i], self.df[j])))

        J_d = J[:, 0:-1]
        K_d = K[:, 0:-1]
        L_d = L[:, 0:-1]
        M_d = M[:, 0:-1]
        N_d = N[:, 0:-1]
        H_d = H[:, 0:-1]
        P_d = P

        self.D = K_d
        self.K1 = L_d - M_d + J_d
        self.K2 = N_d - H_d
        self.P = P_d
    # ******************************************************************************************* #
    # ***                                 DimensionalAnalysis                                 *** #
    # ******************************************************************************************* #

    # ******************************************************************************************* #
    # ***                                     BatteryCell                                     *** #
    # ******************************************************************************************* #

    def set_porousSEI_parameters(self, porous, problem) -> None:
        """
        This method preprocesses the electrode parameters of the
        SEI solvent-diffusion model.

        Parameters
        ----------
        SEI: BaseCellComponent
            Object where electrode parameters are preprocessed and
            stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        porous.U = porous.parser.reference_voltage.get_value(problem)
        porous.beta = porous.parser.charge_transfer_coefficient.get_value(problem)
        porous.D_EC = porous.parser.solvent_diffusion.get_value(problem)
        porous.eps = porous.parser.solvent_porosity.get_value(problem)
        porous.c_EC_sln = porous.parser.solvent_surf_concentration.get_value(problem)
        porous.k_f_s = porous.parser.rate_constant.get_value(problem)
