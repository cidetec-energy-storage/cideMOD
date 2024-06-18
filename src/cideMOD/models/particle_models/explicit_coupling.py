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
from dolfinx.common import Timer
from ufl import (exp, inner, grad, SpatialCoordinate,
                 Measure, TestFunction, TrialFunction, derivative)
from petsc4py.PETSc import ScalarType
from mpi4py import MPI

from typing import List

import numpy as np

from cideMOD.models.base.base_particle_models import WeakCoupledPM


class StandardParticleIntercalation(WeakCoupledPM):
    def __init__(self, active_material: list, F, R, N_s, DT, nodes: int):
        self.DT = DT
        self.F = F
        self.R = R
        self.particles = active_material
        self._build_super_variables()
        self.build_mesh(N_s)
        self.r2 = SpatialCoordinate(self.mesh)[0]**2
        self.build_fs()
        self.build_db(nodes)

    def _build_super_variables(self):
        self.c_e = dfx.fem.Constant(self.mesh, ScalarType(1))
        self.phi = dfx.fem.Constant(self.mesh, ScalarType(1))
        self.T = dfx.fem.Constant(self.mesh, ScalarType(1))

    def build_mesh(self, Ns):
        self.mesh = dfx.mesh.create_unit_interval(MPI.COMM_SELF, Ns)
        boundaries = [(1, lambda x: np.isclose(x[0], 0)),
                      (2, lambda x: np.isclose(x[0], 1)),]
        self.surf_facets = dfx.mesh.locate_entities(self.mesh, 0, boundaries[1][1])
        facet_indices, facet_markers = [], []
        fdim = self.mesh.topology.dim - 1
        for (marker, locator) in boundaries:
            facets = dfx.mesh.locate_entities(self.mesh, fdim, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full(len(facets), marker))
        facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
        facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = dfx.mesh.meshtags(
            self.mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

        self.dx = Measure('dx', domain=self.mesh)
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=facet_tag)

    def build_fs(self):
        P1 = dfx.fem.functionspace(self.mesh, ('Lagrange', 1))

        self.V = P1.clone()
        self.W = P1.clone()

        self.v = TestFunction(self.W)
        self.u = TrialFunction(self.W)

        self.dofs = np.unique(self.W.dofmap.list.array)
        self.u_0 = [dfx.fem.Function(self.W) for i in self.particles]
        self.u_1 = [dfx.fem.Function(self.W) for i in self.particles]

    def build_db(self, nodes: int):
        self.c_s_0_db = np.empty((nodes, len(self.particles), len(self.dofs)), dtype=float)
        self.c_s_1_db = np.empty((nodes, len(self.particles), len(self.dofs)), dtype=float)
        self.c_s__1_db = np.empty((nodes, len(self.particles), len(self.dofs)), dtype=float)
        self.c_surf_index = dfx.fem.locate_dofs_topological(self.W, 0, self.surf_facets)
        self._build_super_db(nodes)

    def _build_super_db(self, nodes):
        self.db_c_e = np.empty(nodes, dtype=float)
        self.db_phi = np.empty(nodes, dtype=float)
        self.db_T = np.empty(nodes, dtype=float)

    def initial_guess(self, c_s_ini: List[float] = []):
        if c_s_ini:
            if len(c_s_ini) != len(self.particles):
                raise ValueError(
                    f"Initial concentration list must be of length {len(self.particles)}")
        else:
            c_s_ini = [material.c_s_ini for material in self.particles]
        for i, c_s in enumerate(c_s_ini):
            if isinstance(c_s, dfx.fem.Constant):
                c_s = c_s.value
            if not isinstance(c_s, float):
                raise TypeError('Initial Concentration must be of type Float')
            self.c_s_0_db[:, i, :].fill(c_s)
            self.c_s_1_db[:, i, :].fill(c_s)
            self.c_s__1_db[:, i, :].fill(c_s)

    def setup(self, params=None):
        self.solvers = []
        for i, material in enumerate(self.particles):
            F = self.c_s_equation(material)
            J = derivative(F, self.u_1[i])
            problem = dfx.fem.petsc.NonlinearProblem(F, self.u_1[i], [], J)
            solver = dfx.nls.petsc.NewtonSolver(self.mesh.comm, problem)
            self.solvers.append(solver)

    def microscale_update(self, c_e: np.array, phi: np.array, T: np.array):
        self.db_c_e = c_e
        self.db_phi = phi
        self.db_T = T

    def _update_constants(self, super_dof):
        timer0 = Timer('Update constants')
        timer0.start()
        self.c_e.value = self.db_c_e[super_dof]
        self.phi.value = self.db_phi[super_dof]
        self.T.value = self.db_T[super_dof]
        timer0.stop()

    def _solve_particle(self, mat_index):
        self.solvers[mat_index].solve()

    def _solve(self):
        db_shape = self.c_s_0_db.shape
        result = np.empty(db_shape, dtype=float)
        for super_dof in range(db_shape[0]):
            self._update_constants(super_dof)
            for mat in range(db_shape[1]):
                timer1 = Timer('Update from db')
                timer1.start()
                self.u_0[mat].vector.array[:] = self.c_s_0_db[super_dof, mat, :]
                self.u_1[mat].interpolate(self.u_0[mat])
                timer1.stop()
                self._solve_particle(mat)
                result[super_dof, mat, :] = self.u_1[mat].vector.array
        return result

    def solve(self):
        self.c_s_1_db[:, :, :] = self._solve()[:, :, :]

    def c_s_surf(self):
        shape = self.c_s_1_db.shape
        c_surf = self.c_s_1_db[:, :, self.c_surf_index].reshape(shape[0], shape[1])
        return c_surf

    def Li_amount(self, electrode_thickness=1):
        db_shape = self.c_s_0_db.shape
        total_li = np.empty((db_shape[0]))
        for cell_dof in range(db_shape[0]):
            c_tot = 0
            for mat in range(db_shape[1]):
                self.u_1[mat].vector.array[:] = self.c_s_1_db[cell_dof, mat]
                CV = dfx.fem.assemble_scalar(self.r2 * self.u_1[mat].sub(0) * self.dx)
                c_tot += CV * self.particles[mat].eps_s
            total_li[cell_dof] = c_tot
        np.trapz(total_li, dx=1 / len(total_li - 1))

    def advance_problem(self):
        self.c_s__1_db[:, :, :] = self.c_s_0_db[:, :, :]
        self.c_s_0_db[:, :, :] = self.c_s_1_db[:, :, :]

    def get_time_filter_error(self, nu, tau):
        error = nu * (1 / (1 + tau) * self.c_s_1_db - self.c_s_0_db
                      + tau / (1 + tau) * self.c_s__1_db)
        return np.linalg.norm(error)

    def c_s_equation(self, material):
        c_s = self.u_1[material.index]
        c_s_0 = self.u_0[material.index]

        j_Li = self._j_li(c_s, self.c_e, self.phi, self.T, material.k_0, material.k_0_Ea,
                          material.k_0_Tref, material.alpha, self.F, self.R, material.U,
                          material.c_s_max)
        if not isinstance(material.D_s, str):
            D_s_eff = material.D_s
        else:
            D_s_eff = self.D_s_exp(material.D_s, c_s)
        D_s_eff = D_s_eff * exp(material.D_s_Ea / self.R * (1 / material.D_s_Tref - 1 / self.T))
        return self._c_s_equation(c_s, c_s_0, self.r2, self.v_0,
                                  self.dx, D_s_eff, material.R_s, j_Li, self.ds)

    def _c_s_equation(self, c_s, c_s_0, r2, test, dx, D_s, R_s, j_Li, ds):
        """
        Particle intercalarion equation for c_s according with Fick's
        Diffusion law. The domain is normalized to [0,1] being the
        normalized radius r=real_r/R_s. Euler implicit method is used
        to discretize time.

        Args:
            c_s (Function or TrialFunction): Lithium concentration in the particle
            c_s_0 (Function): c_s at prior timestep
            dt (Expression): Time step in seconds
            r2 (Expression): particle radius coordinate squared
            test (TestFunction): TestFunction for c_s equation
            dx (Measure): Domain Integral Measure
            D_s (Constant or Expression or Form): Diffusivity of lithium in the particles
                of the electrode
            R_s (Constant or Expression): Radius of the particles
            j_Li (Function or Form): Lithium intercalation Flux
            a_s (Constant or Expression or Form): Active area of electrode. Equals 3*eps_s/R_s
            F (Constant): Faraday's constant
            ds (Measure): Boundaries Integral Measure

        Returns:
            Form: weak form of c_s equation
        """
        return (self.DT.dt(c_s_0, c_s) * r2 * test * dx
                + (D_s * r2 / (R_s**2)) * inner(grad(c_s), grad(test)) * dx
                + (r2 / R_s) * j_Li * test * ds(2))

    def D_s_exp(self, expression, x):
        return eval(expression)

    def _j_li(self, c_s, c_e, phi, T, k_0, k_0_Ea, k_0_Tref, alpha, F, R, OCV, c_s_max):
        """Lithium reaction flux

        Args:
            c_s (Function or TrialFunction): Lithium concentration in the particle
            c_e (Expression): lithium concentration in the electrolyte surrounding the particle
            k_0_Ea (Expression): Activation energy
            a_s (Constant): Active area of the electrode particle
            alpha (Constant): charge transfer coefficient
            F (Constant): Faraday's constant
            c_s_max (Constant): Maximum concentration of electrode particle

        Returns:
            Form: Lithium Reaction Flux dependent on c_s
        """
        eta = phi - OCV(c_s / c_s_max)
        BV = exp((1 - alpha) * F * eta / (R * T)) - exp(-alpha * F * eta / (R * T))
        k_0_eff = k_0 * exp(k_0_Ea / R * (1 / k_0_Tref - 1 / T))
        i_0 = k_0_eff * c_e ** (1 - alpha) * (c_s_max - c_s) ** (1 - alpha) * c_s ** alpha
        return i_0 * BV

    def get_average_c_s(self, increment=False):
        """
        Calculates average concentration in the solid particle, useful
        for thickness change calculations

        :param c_s_ref: Reference concentration to substract if
        necessary, defaults to None
        :type c_s_ref: Constant or float, optional
        """
        shape = self.c_s_1_db.shape  # cell_dof, material, c_s_vector
        c_avg = np.empty(shape[:-1])
        for i, material in enumerate(self.particles):
            c_avg[:, i] = np.array([c.sum() / c.size for c in self.c_s_1_db[:, i, :]])
            if increment:
                c_avg[:, i] -= material.c_s_ini
        return c_avg


class StressEnhancedIntercalation(StandardParticleIntercalation):
    def theta(self, material, R):
        if None not in [material.omega, material.young, material.poisson]:
            return (material.omega * 2 * material.young * material.omega
                    / (R * 9 * (1 - material.poisson)) / self.T)
        else:
            raise Exception(f"Material {material.index} does not have mechanical properties")

    def c_s_equation(self, material):
        c_s = self.u_1[material.index]
        c_s_0 = self.u_0[material.index]

        j_Li = self._j_li(c_s, self.c_e, self.phi, self.T, material.k_0, material.k_0_Ea,
                          material.k_0_Tref, material.alpha, self.F, self.R, material.U,
                          material.c_s_max)
        if not isinstance(material.D_s, str):
            D_s_eff = material.D_s
        else:
            D_s_eff = self.D_s_exp(material.D_s, c_s)
        D_s_eff = D_s_eff * exp(material.D_s_Ea / self.R * (1 / material.D_s_Tref - 1 / self.T))
        theta = self.theta(material, self.R)
        return self._c_s_equation(c_s, c_s_0, material.c_s_ini, self.r2, self.v_0,
                                  self.dx, D_s_eff, material.R_s, j_Li, self.ds, theta)

    def _c_s_equation(self, c_s, c_s_0, c_ini, r2, test, dx, D_s, R_s, j_Li, ds, theta):
        """
        Particle intercalarion equation for c_s according with Fick's
        Diffusion law with stress contribution. The domain is normalized
        to [0,1] being the normalized radius r=real_r/R_s. Euler
        implicit method is used to discretize time.

        Args:
            c_s (Function or TrialFunction): Lithium concentration in the particle
            c_s_0 (Function): c_s at prior timestep
            c_ini (Constant): reference c_s at initial time (where mechanical parameters are given)
            dt (Expression): Time step in seconds
            r2 (Expression): particle radius coordinate squared
            test (TestFunction): TestFunction for c_s equation
            dx (Measure): Domain Integral Measure
            D_s (Constant or Expression or Form): Diffusivity of lithium in the particles of
                the electrode
            R_s (Constant or Expression): Radius of the particles
            j_Li (Function or Form): Lithium intercalation Flux
            a_s (Constant or Expression or Form): Active area of electrode. Equals 3*eps_s/R_s
            F (Constant): Faraday's constant
            ds (Measure): Boundaries Integral Measure
            theta (Constant or Expression or Form): Mechanical effect coefficient equals to
                (2*E*omega^2) / (9*R*T*(1-nu)) with E=Young's Modulus, nu=Poisson's ratio and
                omega=Partial molar volume

        Returns:
            Form: weak form of c_s equation
        """
        return (r2 * self.DT.dt(c_s_0, c_s) * test * dx
                + r2 * (D_s / R_s ** 2) * inner(grad(c_s), grad(test)) * dx
                + theta * (D_s / R_s ** 2) * r2 * inner(grad(c_s), grad(test)) * (c_s - c_ini) * dx
                + (r2 / (R_s)) * j_Li * test * ds(2))

    def get_average_c_s(self, increment=False):
        """
        Calculates average concentration in the solid particle, useful
        for thickness change calculations

        :param c_s_ref: Reference concentration to substract if necessary, defaults to None
        :type c_s_ref: Constant or float, optional
        """
        shape = self.c_s_1_db.shape  # cell_dof, material, c_s_vector
        c_avg = np.empty(shape[:-1])
        for i, material in enumerate(self.particles):
            c_avg[:, i] = np.array([c.sum() / c.size for c in self.c_s_1_db[:, i, :]])
            if increment:
                c_avg[:, i] -= material.c_s_ini
        return c_avg
