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
import os
from mpi4py import MPI
from numpy import array, ndarray

from cideMOD.numerics.fem_handler import interpolate, assign
from cideMOD.numerics.triggers import SolverCrashed


class ErrorCheck:
    # TODO: Adapt ErrorCheck to the modular structure. Ask the models the check their variables.
    #       Move it to cell or models module.
    def __init__(self, problem, status, name='', debug=False):
        self.comm = problem._comm
        self.problem = problem
        self.subdomains = problem.P1_map.get_subdomains_dofs()
        if isinstance(status, SolverCrashed) and problem.save_path:
            self.log = []
            try:
                self.print('\nSolver crashed, performing failure checks...')
                self.print('\n------------LAST TIMESTEP------------\n')
                self.check_electrolyte_depleted()
                self._compute_electrode_charge()
                self.check_electrode_depleted()
                self.check_electrode_overloaded()
                self.check_coeffs()
                if debug:
                    self.print('\n------------PREVIOUS TIMESTEP------------\n')
                    # FIXME: Pass f to the method and do not reset f_1
                    assign(self.problem.u_1, self.problem.u_0)
                    self.check_electrolyte_depleted()
                    self._compute_electrode_charge()
                    self.check_electrode_depleted()
                    self.check_electrode_overloaded()
                    self.check_coeffs()
            except Exception as e:
                self.print('Error writing error_check file')
                self.print(str(e))
                # raise e # NOTE: Debug only
            finally:
                if self.comm.rank == 0:
                    filepath = os.path.join(problem.save_path, f'error_check_{name}.txt')
                    with open(filepath, 'w') as f:
                        f.writelines(self.log)

    def _compute_electrode_charge(self):
        if self.problem.model_options.particle_model.startswith('SGM'):
            var = self.problem._vars
            x_a = [interpolate(var.x_a_surf[i], self.problem.V) for i in range(var.n_mat_a)]
            x_c = [interpolate(var.x_c_surf[i], self.problem.V) for i in range(var.n_mat_c)]
        else:
            raise NotImplementedError
        self.x_a = [x.vector.array[self.subdomains.anode] for x in x_a]
        self.x_c = [x.vector.array[self.subdomains.cathode] for x in x_c]

    def check_electrolyte_depleted(self):
        """
        If electrolyte concentration is zero or below, numeric crashes.
        """
        c_e = interpolate(self.problem._vars.c_e, self.problem.V)
        c_e = c_e.vector.array[self.subdomains.electrolyte]
        min_c_e = min(1e12, min(c_e))
        min_c_e = self.comm.allreduce(min_c_e, MPI.MIN)
        if min_c_e <= 0:
            self.print(f"\tERROR - Electrolyte has depleted!! (min c_e = {min_c_e:.2e})")
        else:
            self.print(f"\tOK - Minimum electrolyte concentration {min_c_e:.2e}")

    def check_electrode_overloaded(self):
        """
        If electrode surface concentration exceeds maximum surface concentration, numeric crashes
        """
        for i, x in enumerate(self.x_a):
            if len(x) > 0:
                max_x = max(x)
            else:
                max_x = 0
            max_x_a = self.comm.allreduce(max_x, MPI.MAX)
            if max_x_a > 1:
                self.print(f"\tERROR - Anode material {i} overloaded! ",
                           f"(max c_s_a = {100*max_x_a:.2f}%)")
            else:
                self.print(f"\tOK - Max Anode material {i} concentration {100*max_x_a:.2f}%")
        for i, x in enumerate(self.x_c):
            if len(x) > 0:
                max_x = max(x)
            else:
                max_x = 0
            max_x_c = self.comm.allreduce(max_x, MPI.MAX)
            if max_x_c > 1:
                self.print(f"\tERROR - Cathode material {i} overloaded! ",
                           f"(max c_s_c = {100*max_x_c:.2f}%)")
            else:
                self.print(f"\tOK - Max Cathode material {i} concentration {100*max_x_c:.2f}%")

    def check_electrode_depleted(self):
        """
        If electrode surface concentration is below zero, numeric crashes
        """
        for i, x in enumerate(self.x_a):
            if len(x) > 0:
                min_x = min(x)
            else:
                min_x = 1
            min_x_a = self.comm.allreduce(min_x, MPI.MIN)
            if min_x_a <= 0:
                self.print(f"\tERROR - Anode material {i} depleted! ",
                           f"(min c_s_a = {100*min_x_a:.2f}%)")
            else:
                self.print(f"\tOK - Min Anode material {i} concentration {100*min_x_a:.2f}%")
        for i, x in enumerate(self.x_c):
            if len(x) > 0:
                min_x = min(x)
            else:
                min_x = 1
            min_x_c = self.comm.allreduce(min_x, MPI.MIN)
            if min_x_c <= 0:
                self.print(f"\tERROR - Cathode material {i} depleted! ",
                           f"(min c_s_c = {100*min_x_c:.2f}%)")
            else:
                self.print(f"\tOK - Min Cathode material {i} concentration {100*min_x_c:.2f}%")

    def check_temperatures(self):
        """
        Check temperatures are in a good range 0-60ºC
        """
        pass

    def check_coeffs(self):
        """
        Coefficients could be nonlinear and diverge for some values of internal variables, according to their expressions.
        """
        cell = self.problem.cell
        self.print('\nChecking coeffs')
        D_e_a = self._check_coeff(cell.anode.D_e, self.subdomains.anode)
        D_e_s = self._check_coeff(cell.separator.D_e, self.subdomains.separator)
        D_e_c = self._check_coeff(cell.cathode.D_e, self.subdomains.cathode)
        self.print("\tElectrolyte diffusivity:")
        self.print("\t\tAnode: max: {:.2e}, min: {:.2e}".format(D_e_a[0], D_e_a[1]))
        self.print("\t\tSeparator: max: {:.2e}, min: {:.2e}".format(D_e_s[0], D_e_s[1]))
        self.print("\t\tCathode: max: {:.2e}, min: {:.2e}".format(D_e_c[0], D_e_c[1]))

        k_e_a = self._check_coeff(cell.anode.kappa, self.subdomains.anode)
        k_e_s = self._check_coeff(cell.separator.kappa, self.subdomains.separator)
        k_e_c = self._check_coeff(cell.cathode.kappa, self.subdomains.cathode)
        self.print("\tElectrolyte ionic conductivity:")
        self.print("\t\tAnode: max: {:.2e}, min: {:.2e}".format(k_e_a[0], k_e_a[1]))
        self.print("\t\tSeparator: max: {:.2e}, min: {:.2e}".format(k_e_s[0], k_e_s[1]))
        self.print("\t\tCathode: max: {:.2e}, min: {:.2e}".format(k_e_c[0], k_e_c[1]))

        k_d_e_a = self._check_coeff(cell.anode.kappa_D, self.subdomains.anode)
        k_d_e_s = self._check_coeff(cell.separator.kappa_D, self.subdomains.separator)
        k_d_e_c = self._check_coeff(cell.cathode.kappa_D, self.subdomains.cathode)
        self.print("\tElectrolyte concentration effective conductivity:")
        self.print("\t\tAnode: max: {:.2e}, min: {:.2e}".format(k_d_e_a[0], k_d_e_a[1]))
        self.print("\t\tSeparator: max: {:.2e}, min: {:.2e}".format(k_d_e_s[0], k_d_e_s[1]))
        self.print("\t\tCathode: max: {:.2e}, min: {:.2e}".format(k_d_e_c[0], k_d_e_c[1]))

        sig_a = self._check_coeff(cell.anode.sigma, self.subdomains.anode)
        sig_c = self._check_coeff(cell.cathode.sigma, self.subdomains.cathode)
        self.print("\tElectrode electronic conductivity:")
        self.print("\t\tAnode: max: {:.2e}, min: {:.2e}".format(sig_a[0], sig_a[1]))
        self.print("\t\tCathode: max: {:.2e}, min: {:.2e}".format(sig_c[0], sig_c[1]))

    def _check_coeff(self, coeff, subdomain):
        if not isinstance(subdomain, ndarray):
            subdomain = array(subdomain)
        v = interpolate(coeff, self.problem.V)
        v = v.vector.array
        if subdomain.any():
            v = v[subdomain]
            vmax = v.max()
            vmin = v.min()
        else:
            vmax = -9e99
            vmin = 9e99
        vmax = self.comm.allreduce(vmax, MPI.MAX)
        vmin = self.comm.allreduce(vmin, MPI.MIN)
        return [vmax, vmin]

    def print(self, *args):
        self.log.append(' '.join(str(arg) for arg in args) + '\n')
