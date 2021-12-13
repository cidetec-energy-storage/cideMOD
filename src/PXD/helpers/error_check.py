#
# Copyright (c) 2021 CIDETEC Energy Storage.
#
# This file is part of PXD.
#
# PXD is free software: you can redistribute it and/or modify
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
from dolfin import MPI, project
from multiphenics import block_assign

import os

from numpy import array, ndarray

comm =MPI.comm_world

class ErrorCheck:
    def print(self, *args):
        self.log.append(' '.join(str(arg) for arg in args)+'\n')
                
    def __init__(self, problem, status, name='', debug=False):
        self.problem = problem
        self.subdomains = self.problem.mesher.get_subdomains_coord(problem.P1_map)
        if status != 0:
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
                    block_assign(self.problem.u_1,self.problem.u_0)
                    self.check_electrolyte_depleted()
                    self._compute_electrode_charge()
                    self.check_electrode_depleted()
                    self.check_electrode_overloaded()
                    self.check_coeffs()
            except Exception as e:
                self.print('Error writing error_check file')
                self.print(str(e))
                # raise e
            finally:
                if MPI.rank(comm)==0:
                    with open(os.path.join(problem.save_path,'error_check_{}.txt'.format(name)),'w') as f:
                        f.writelines(self.log)

    def _compute_electrode_charge(self):
        if self.problem.c_s_implicit_coupling:
            if 'nd_model' in self.problem.__dict__:
                x_a = [project(self.problem.SGM.c_s_surf(self.problem.f_1,'anode')[i],self.problem.V) for i, material in enumerate(self.problem.anode.active_material)]
                x_c = [project(self.problem.SGM.c_s_surf(self.problem.f_1,'cathode')[i],self.problem.V) for i, material in enumerate(self.problem.cathode.active_material)]
            else:
                x_a = [project(self.problem.SGM.c_s_surf(self.problem.f_1,'anode')[i]/material.c_s_max,self.problem.V) for i, material in enumerate(self.problem.anode.active_material)]
                x_c = [project(self.problem.SGM.c_s_surf(self.problem.f_1,'cathode')[i]/material.c_s_max,self.problem.V) for i, material in enumerate(self.problem.cathode.active_material)]
        else:
            x_a = [ project(self.problem.c_s_surf_1_anode[i]/material.c_s_max ,self.problem.V) for i, material in enumerate(self.problem.anode.active_material)]
            x_c = [ project(self.problem.c_s_surf_1_cathode[i]/material.c_s_max ,self.problem.V) for i, material in enumerate(self.problem.cathode.active_material)]
        self.x_a = [x.vector()[self.subdomains.anode] for x in x_a]
        self.x_c = [x.vector()[self.subdomains.cathode] for x in x_c]
        
    def check_electrolyte_depleted(self):
        """
        If electrolyte concentration is zero or below, numeric crashes.
        """
        if 'nd_model' in self.problem.__dict__:
            c_e = project(self.problem.dim_variables.c_e, self.problem.V).vector()[self.subdomains.electrolyte]
        else:
            c_e = self.problem.u_1.sub(0).vector()[self.subdomains.electrolyte]
        min_c_e = 1e12
        min_c_e = MPI.min(comm, min(min(c_e),min_c_e))
        if min_c_e<=0:
            self.print('\tERROR - Electrolyte has depleted!! (min c_e =',"{:.2e}".format(min_c_e),')')
        else:
            self.print('\tOK - Minimum electrolyte concentration:',min_c_e)

    def check_electrode_overloaded(self):
        """
        If electrode surface concentration excedes maximum surface concentration, numeric crashes
        """
        for i, x in enumerate(self.x_a):
            if len(x)>0:
                max_x = max(x)
            else:
                max_x = 0
            max_x_a = MPI.max(comm, max_x)
            if max_x_a > 1:
                self.print('\tERROR - Anode material {} overloaded!! (max c_s_a ='.format(i),"{:.2f}%".format(max_x_a*100),')')
            else:
                self.print('\tOK - Max Anode material {} concentration'.format(i),"{:.2f}%".format(max_x_a*100))
        for i, x in enumerate(self.x_c):
            if len(x)>0:
                max_x = max(x)
            else:
                max_x = 0
            max_x_c = MPI.max(comm, max_x)
            if max_x_c > 1:
                self.print('\tERROR - Cathode material {} overloaded!! (max c_s_c ='.format(i),"{:.2f}%".format(max_x_c*100),')')
            else:
                self.print('\tOK - Max Cathode material {} concentration'.format(i),"{:.2f}%".format(max_x_c*100))


    def check_electrode_depleted(self):
        """
        If electrode surface concentration is below zero, numeric crashes
        """
        for i, x in enumerate(self.x_a):
            if len(x)>0:
                min_x = min(x)
            else:
                min_x = 1
            min_x_a = MPI.min(comm, min_x)
            if min_x_a <= 0:
                self.print('\tERROR - Anode material {} depleted!! (min c_s_a ='.format(i),"{:.2f}%".format(min_x_a*100),')')
            else:
                self.print('\tOK - Min Anode material {} concentration'.format(i),"{:.2f}%".format(min_x_a*100))
        for i, x in enumerate(self.x_c):
            if len(x)>0:
                min_x = min(x)
            else:
                min_x = 1
            min_x_c = MPI.min(comm, min_x)
            if min_x_c <= 0:
                self.print('\tERROR - Cathode material {} depleted!! (min c_s_c ='.format(i),"{:.2f}%".format(min_x_c*100),')')
            else:
                self.print('\tOK - Min Cathode material {} concentration'.format(i),"{:.2f}%".format(min_x_c*100))

    def check_temperatures(self):
        """
        Check temperatures are in a good range 0-60ºC
        """
        pass

    def check_coeffs(self):
        """
        Coefficients could be nonlinear and diverge for some values of internal variables, according to their expressions.
        """
        self.print('\nChecking coeffs')
        D_e_a = self._check_coeff(self.problem.anode.D_e, self.subdomains.anode)
        D_e_s = self._check_coeff(self.problem.separator.D_e, self.subdomains.separator)
        D_e_c = self._check_coeff(self.problem.cathode.D_e, self.subdomains.cathode)
        self.print("\tElectrolyte diffusivity:")
        self.print("\t\tAnode: max: {:.2e}, min: {:.2e}".format(D_e_a[0],D_e_a[1]))
        self.print("\t\tSeparator: max: {:.2e}, min: {:.2e}".format(D_e_s[0],D_e_s[1]))
        self.print("\t\tCathode: max: {:.2e}, min: {:.2e}".format(D_e_c[0],D_e_c[1]))

        k_e_a = self._check_coeff(self.problem.anode.kappa, self.subdomains.anode)
        k_e_s = self._check_coeff(self.problem.separator.kappa, self.subdomains.separator)
        k_e_c = self._check_coeff(self.problem.cathode.kappa, self.subdomains.cathode)
        self.print("\tElectrolyte ionic conductivity:")
        self.print("\t\tAnode: max: {:.2e}, min: {:.2e}".format(k_e_a[0],k_e_a[1]))
        self.print("\t\tSeparator: max: {:.2e}, min: {:.2e}".format(k_e_s[0],k_e_s[1]))
        self.print("\t\tCathode: max: {:.2e}, min: {:.2e}".format(k_e_c[0],k_e_c[1]))

        k_d_e_a = self._check_coeff(self.problem.anode.kappa_D, self.subdomains.anode)
        k_d_e_s = self._check_coeff(self.problem.separator.kappa_D, self.subdomains.separator)
        k_d_e_c = self._check_coeff(self.problem.cathode.kappa_D, self.subdomains.cathode)
        self.print("\tElectrolyte concentration effective conductivity:")
        self.print("\t\tAnode: max: {:.2e}, min: {:.2e}".format(k_d_e_a[0],k_d_e_a[1]))
        self.print("\t\tSeparator: max: {:.2e}, min: {:.2e}".format(k_d_e_s[0],k_d_e_s[1]))
        self.print("\t\tCathode: max: {:.2e}, min: {:.2e}".format(k_d_e_c[0],k_d_e_c[1]))

        sig_a = self._check_coeff(self.problem.anode.sigma, self.subdomains.anode)
        sig_c = self._check_coeff(self.problem.cathode.sigma, self.subdomains.cathode)
        self.print("\tElectrode ionic conductivity:")
        self.print("\t\tAnode: max: {:.2e}, min: {:.2e}".format(sig_a[0],sig_a[1]))
        self.print("\t\tCathode: max: {:.2e}, min: {:.2e}".format(sig_c[0],sig_c[1]))


    def _check_coeff(self, coeff, subdomain):
        if not isinstance(subdomain, ndarray):
            subdomain = array(subdomain)
        v=project(coeff,self.problem.V, form_compiler_parameters={'quadrature_degree':2})
        v=v.vector()
        if subdomain.any():
            v = v[subdomain]
            vmax = v.max()
            vmin = v.min()
        else:
            vmax = -9e99
            vmin = 9e99
        vmax = MPI.max(comm, vmax)
        vmin = MPI.min(comm, vmin)
        return [vmax,vmin]
        