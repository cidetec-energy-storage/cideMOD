from dolfin import *
from multiphenics import *

from typing import List

import numpy

from PXD.models.base.base_particle_models import WeakCoupledPM


class Center(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.0 + DOLFIN_EPS and on_boundary

class Surface(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 1.0 - DOLFIN_EPS and on_boundary

class StandardParticleIntercalation(WeakCoupledPM):
    def __init__(self, active_material:list, F, alpha, R, N_s, DT, nodes:int):
        self.r2 = Expression("pow(x[0],2)", degree=2)
        self.DT = DT
        self.F = F
        self.alpha = alpha
        self.R = R
        self.particles = active_material
        self._build_super_variables()
        self.build_mesh(N_s)
        self.build_fs()
        self.build_db(nodes)

    def _build_super_variables(self):
        self.c_e = Constant(1)
        self.phi = Constant(1)
        self.T = Constant(1)

    def build_mesh(self, Ns):
        self.mesh = UnitIntervalMesh(Ns)
        boundaries = MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1, 0)
        center = Center()
        surf = Surface()
        center.mark(boundaries, 1)
        surf.mark(boundaries, 2)

        self.dx = Measure('dx', domain=self.mesh)
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=boundaries)

    def build_fs(self):
        P1 = FiniteElement('CG', self.mesh.ufl_cell(), 1)
        LM = FiniteElement('R', self.mesh.ufl_cell(), 0)
        c_s_elements = P1
        lm_D_s_elements = LM
        ME = MixedElement([c_s_elements, lm_D_s_elements])

        self.V = FunctionSpace(self.mesh, P1)
        self.W = FunctionSpace(self.mesh, ME)

        self.v = TestFunction(self.W)
        self.v_0, self.v_1 = split(self.v)
        self.u = TrialFunction(self.W)
        
        self.dofs = self.W.dofmap().dofs()
        self.u_0 = [Function(self.W) for i in self.particles]
        self.u_1 = [Function(self.W) for i in self.particles]

    def build_db(self, nodes: int):
        self.c_s_0_db = numpy.empty((nodes, len(self.particles), len(self.dofs)),dtype=float)
        self.c_s_1_db = numpy.empty((nodes, len(self.particles), len(self.dofs)),dtype=float)
        self.c_s__1_db = numpy.empty((nodes, len(self.particles), len(self.dofs)),dtype=float)
        self.c_surf_index = numpy.where(self.W.tabulate_dof_coordinates().flatten()>=1-DOLFIN_EPS)
        self._build_super_db(nodes)

    def _build_super_db(self, nodes):
        self.db_c_e = numpy.empty(nodes,dtype=float)
        self.db_phi = numpy.empty(nodes,dtype=float)
        self.db_T = numpy.empty(nodes,dtype=float)

    def initial_guess(self, c_s_ini: List[float] = []):
        if c_s_ini:
            assert len(c_s_ini) == len(self.particles), "Initial concentration list must be of lenght {}".format(len(self.particles))
        else:
            c_s_ini = [material.c_s_ini for material in self.particles]
        for i, c_s in enumerate(c_s_ini):
            if isinstance(c_s, Constant):
                c_s = c_s.values()[0]
            assert isinstance(c_s,float), 'Initial Concentration must be of type Float'
            self.c_s_0_db[:,i,:].fill(c_s)
            self.c_s_1_db[:,i,:].fill(c_s)
            self.c_s__1_db[:,i,:].fill(c_s)

    def setup(self, params=None):
        if not params:
            params = {
                'nonlinear_solver': 'snes',
                'snes_solver':
                {
                    'method': 'newtonls',
                    'line_search': 'basic',
                    'linear_solver': 'mumps',
                    'report': False,
                    'absolute_tolerance': 1E-6,
                }
            }
        self.solvers = []
        for i, material in enumerate(self.particles):
            F = self.c_s_equation(material) + self.lm_D_s_equation(material)
            J = derivative(F, self.u_1[i])
            problem = NonlinearVariationalProblem(F,  self.u_1[i], [], J)
            solver = NonlinearVariationalSolver(problem)
            solver.parameters.update(params)
            self.solvers.append(solver)

    def microscale_update(self, c_e:numpy.array, phi:numpy.array, T:numpy.array):
        self.db_c_e = c_e
        self.db_phi = phi
        self.db_T = T

    def _update_constants(self, super_dof):
        timer0 = Timer('Update constants')
        self.c_e.assign(self.db_c_e[super_dof])
        self.phi.assign(self.db_phi[super_dof])
        self.T.assign(self.db_T[super_dof])
        timer0.stop()

    def _solve_particle(self, mat_index):
        self.solvers[mat_index].solve()

    def _solve(self):
        db_shape = self.c_s_0_db.shape
        result = numpy.empty(db_shape,dtype=float)
        for super_dof in range(db_shape[0]):
            self._update_constants(super_dof)
            for mat in range(db_shape[1]):
                timer1 = Timer('Update from db')
                self.u_0[mat].sub(0).vector()[:] = self.c_s_0_db[super_dof,mat,:]
                assign(self.u_1[mat].sub(0), self.u_0[mat].sub(0))
                timer1.stop()
                self._solve_particle(mat)
                result[super_dof,mat,:] = self.u_1[mat].sub(0).vector()[:]
        return result

    def solve(self):
        self.c_s_1_db[:,:,:] = self._solve()[:,:,:]

    def c_s_surf(self):
        shape = self.c_s_1_db.shape
        c_surf = self.c_s_1_db[:,:,self.c_surf_index].reshape(shape[0],shape[1])
        return c_surf

    def Li_amount(self, electrode_thickness=1):
        db_shape = self.c_s_0_db.shape
        total_li = numpy.empty((db_shape[0]))
        for cell_dof in range(db_shape[0]):
            c_tot = 0
            for mat in range(db_shape[1]):
                self.u_1[mat].sub(0).vector()[:] = self.c_s_1_db[cell_dof,mat]
                CV = assemble(self.r2*self.u_1[mat].sub(0)*self.dx)
                c_tot += CV * self.particles[mat].eps_s
            total_li[cell_dof] = c_tot
        numpy.trapz(total_li,dx=1/len(total_li-1))

    def advance_problem(self):
        self.c_s__1_db[:,:,:] = self.c_s_0_db[:,:,:]
        self.c_s_0_db[:,:,:] = self.c_s_1_db[:,:,:]

    def get_time_filter_error(self, nu, tau):
        error = nu * ( 1/(1+tau) * self.c_s_1_db - self.c_s_0_db + tau/(1+tau)*self.c_s__1_db  )
        return numpy.linalg.norm(error)

    def c_s_equation(self, material):
        c_s, lm = split(self.u_1[material.index])
        c_s_0, _ = split(self.u_0[material.index])
        
        j_Li = self._j_li(c_s, self.c_e, self.phi, self.T, material.k_0, material.k_0_Ea, material.k_0_Tref,
                          self.alpha, self.F, self.R, material.U, material.c_s_max)
        if not isinstance(material.D_s,str):
            D_s_eff = material.D_s
        else:
            D_s_eff = self.D_s_exp(material.D_s, lm)
        D_s_eff = D_s_eff * exp(material.D_s_Ea/self.R * (1/material.D_s_Tref - 1/self.T))
        return self._c_s_equation(c_s, c_s_0, self.r2, self.v_0,
                                  self.dx, D_s_eff, material.R_s, j_Li, self.ds)

    def _c_s_equation(self, c_s, c_s_0, r2, test, dx, D_s, R_s, j_Li, ds):
        """Particle intercalarion equation for c_s according with Fick's Diffusion law.
        The domain is normalized to [0,1] being the normalized radius r=real_r/R_s.
        Euler implicit method is used to discretize time.

        Args:
            c_s (Function or TrialFunction): Lithium concentration in the particle
            c_s_0 (Function): c_s at prior timestep
            dt (Expression): Time step in seconds
            r2 (Expression): particle radius coordinate squared
            test (TestFunction): TestFunction for c_s equation
            dx (Measure): Domain Integral Measure
            D_s (Constant or Expression or Form): Diffusivity of lithium in the particles of the electrode
            R_s (Constant or Expression): Radius of the particles
            j_Li (Function or Form): Lithium intercalation Flux
            a_s (Constant or Expression or Form): Active area of electrode. Equals 3*eps_s/R_s
            F (Constant): Faraday's constant
            ds (Measure): Boundaries Integral Measure

        Returns:
            Form: weak form of c_s equation
        """
        return self.DT.dt(c_s_0, c_s)*r2*test*dx + (D_s*r2/(R_s**2))*inner(grad(c_s), grad(test))*dx + (r2/R_s)*j_Li*test*ds(2)

    def lm_D_s_equation(self, material):
        c_s, lm = split(self.u_1[material.index])
        return lm * self.v_1 * self.ds(2) - c_s / material.c_s_max *self.v_1 * self.ds(2)
    
    def D_s_exp(self, expression, x):
        return eval(expression)


class StressEnhancedIntercalation(StandardParticleIntercalation):
    def theta(self, material, R):
        if material.omega is not None and material.young is not None and material.poisson is not None:
            return Constant((material.omega*2*material.young*material.omega)/(R*9*(1-material.poisson)))/self.T
        else:
            raise Exception('Material {} does not have mechanical properties'.format(material.index))
    def c_s_equation(self, material):
        c_s, lm = split(self.u_1[material.index])
        c_s_0, _ = split(self.u_0[material.index])
        
        j_Li = self._j_li(c_s, self.c_e, self.phi, self.T, material.k_0, material.k_0_Ea, material.k_0_Tref,
                          self.alpha, self.F, self.R, material.U, material.c_s_max)
        if not isinstance(material.D_s,str):
            D_s_eff = material.D_s
        else:
            D_s_eff = self.D_s_exp(material.D_s, lm)
        D_s_eff = D_s_eff * exp(material.D_s_Ea/self.R * (1/material.D_s_Tref - 1/self.T))
        theta = self.theta(material, self.R)
        return self._c_s_equation(c_s, c_s_0, material.c_s_ini, self.r2, self.v_0,
                                  self.dx, D_s_eff, material.R_s, j_Li, self.ds, theta)

    def _c_s_equation(self, c_s, c_s_0, c_ini, r2, test, dx, D_s, R_s, j_Li, ds, thetha):
        """Particle intercalarion equation for c_s according with Fick's Diffusion law with stress contribution.
        The domain is normalized to [0,1] being the normalized radius r=real_r/R_s.
        Euler implicit method is used to discretize time.

        Args:
            c_s (Function or TrialFunction): Lithium concentration in the particle
            c_s_0 (Function): c_s at prior timestep
            c_ini (Constant): reference c_s at initial time (where mechanical parameters are given)
            dt (Expression): Time step in seconds
            r2 (Expression): particle radius coordinate squared
            test (TestFunction): TestFunction for c_s equation
            dx (Measure): Domain Integral Measure
            D_s (Constant or Expression or Form): Diffusivity of lithium in the particles of the electrode
            R_s (Constant or Expression): Radius of the particles
            j_Li (Function or Form): Lithium intercalation Flux
            a_s (Constant or Expression or Form): Active area of electrode. Equals 3*eps_s/R_s
            F (Constant): Faraday's constant
            ds (Measure): Boundaries Integral Measure
            thetha (Constant or Expression or Form): Mechanical effect coefficient equals to (2*E*omega^2) / (9*R*T*(1-nu)) with E=Young's Modulus, nu=Poisson's ratio and omega=Partial molar volume

        Returns:
            Form: weak form of c_s equation
        """
        return r2*self.DT.dt(c_s_0, c_s)*test*dx + r2*(D_s / R_s**2) * inner(grad(c_s), grad(test))*dx + \
                thetha*(D_s / R_s**2) * r2 * inner(grad(c_s), grad(test))*(c_s-c_ini)*dx + (r2/(R_s))*j_Li*test*ds(2)

    def get_average_c_s(self, increment=False):
        """Calculates average concentration in the solid particle, useful for thickness change calculations

        :param c_s_ref: Reference concentration to substract if necessary, defaults to None
        :type c_s_ref: Constant or float, optional
        """
        shape = self.c_s_1_db.shape # cell_dof, material, c_s_vector
        c_avg = numpy.empty(shape[:-1])
        for i, material in enumerate(self.particles):
            c_avg[:,i] = numpy.array([c.sum()/c.size for c in self.c_s_1_db[:,i,:]])
            if increment:
                c_avg[:,i] -= material.c_s_ini
        return c_avg

