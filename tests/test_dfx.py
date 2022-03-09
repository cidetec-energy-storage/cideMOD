from time import time
import dolfinx as dfx
import multiphenicsx.fem
import petsc4py
import typing
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import ufl
import scipy.constants as ctes
import numpy as np
from petsc4py import PETSc



def test_multiphenicsx(block:bool) -> np.ndarray:

    F = ctes.physical_constants["Faraday constant"][0]
    R = ctes.R

    # Params
    T = ctes.convert_temperature(25,'Celsius','Kelvin')
    dt = 10 #s
    L = 10e-5 #m
    A = 2.14e-4
    _I_app = 0.01 # A

    # Properties
    eps = 0.5
    D_eff = 1.8e-10
    K_eff = 0.1
    t_p = 0.25
    k_0 = 1e-9

    mesh = dfx.mesh.create_unit_interval(MPI.COMM_WORLD, 100)

    boundaries = [(1, lambda x: np.isclose(x[0], 0)),
                (2, lambda x: np.isclose(x[0], 1)),]

    left_facets = dfx.mesh.locate_entities(mesh, 0, boundaries[0][1]) 

    facet_indices, facet_markers = [], []
    fdim = mesh.topology.dim - 1
    for (marker, locator) in boundaries:
        facets = dfx.mesh.locate_entities(mesh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))
    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = dfx.mesh.MeshTags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

    # Function Space and Measures
    if block:
        C_E = dfx.fem.FunctionSpace(mesh, ('CG',1))
        PHI_E = dfx.fem.FunctionSpace(mesh, ('CG',1))

        c_e, phi_e = dfx.fem.Function(C_E), dfx.fem.Function(PHI_E)
        c_e_0, phi_e_0 = dfx.fem.Function(C_E), dfx.fem.Function(PHI_E)

        dc_e, dphi_e = ufl.TrialFunction(C_E), ufl.TrialFunction(PHI_E)
        v_c, v_phi = ufl.TestFunction(C_E), ufl.TestFunction(PHI_E)
    else:
        P1 = ufl.FiniteElement('CG', mesh.ufl_cell(), 1)
        FS = dfx.fem.FunctionSpace(mesh,ufl.MixedElement(P1,P1))

        u_1 = dfx.fem.Function(FS)
        u_0 = dfx.fem.Function(FS)
        du = ufl.TrialFunction(FS)
        dc_e, dphi_e = ufl.TrialFunctions(FS)

        c_e, phi_e = ufl.split(u_1)
        c_e_0, phi_e_0 = ufl.split(u_0)

        v_c, v_phi = ufl.TestFunctions(FS)


    # Measures
    dx = ufl.Measure('dx', mesh)
    ds = ufl.Measure('ds', mesh, subdomain_data=facet_tag)
    ds_0 = ds(1)
    ds_1 = ds(2)

    I_app = dfx.fem.Constant(mesh, ScalarType(0))

    # Define residuals
    residuals = [
        eps * (c_e-c_e_0)/ dt * v_c * dx + D_eff/L**2 * ufl.inner(ufl.grad(c_e),ufl.grad(v_c)) * dx - 1/L*(1-t_p)/F * I_app/A * v_c * ds_0 + 1/L*(1-t_p)/F * I_app/A * v_c * ds_1,
        K_eff/L * ufl.inner(ufl.grad(phi_e),ufl.grad(v_phi)) * dx - 2*R*T/F *(1-t_p)*K_eff/L * ufl.inner(ufl.grad(c_e),ufl.grad(v_phi))/c_e * dx - F*k_0*c_e**0.5*2*ufl.sinh(-0.5*F/(R*T)*phi_e) * v_phi * ds_1
    ]

    # Create problem and solver

    class NonlinearBlockProblem(object):
        """Define a nonlinear problem, interfacing with SNES."""

        def __init__(
            self, F: typing.List[ufl.Form], u: typing.Tuple[dfx.fem.Function], bcs: typing.List[dfx.fem.DirichletBCMetaClass], 
            J: typing.List[typing.List[ufl.Form]], restriction: typing.Optional[typing.List[multiphenicsx.fem.DofMapRestriction]] = None, 
            P: typing.Optional[ufl.Form] = None
        ) -> None:
            self._F = dfx.fem.form(F)
            self._J = dfx.fem.form(J)
            self._obj_vec = multiphenicsx.fem.petsc.create_vector_block(self._F, restriction)
            self._solutions = u
            self._bcs = bcs
            self._restriction = restriction
            self._P = dfx.fem.form(P)

        def create_snes_solution(self) -> petsc4py.PETSc.Vec:
            """Create SNES solution vector"""
            x = multiphenicsx.fem.petsc.create_vector_block(self._F, restriction=self._restriction)
            with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(
                    x, [c.function_space.dofmap for c in self._solutions], self._restriction) as x_wrapper:
                for x_wrapper_local, component in zip(x_wrapper, self._solutions):
                    with component.vector.localForm() as component_local:
                        x_wrapper_local[:] = component_local
            return x

        def update_solutions(self, x: petsc4py.PETSc.Vec) -> None:
            """Update `self._solutions` with data in `x`."""
            x.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
            with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(
                    x, [c.function_space.dofmap for c in self._solutions], self._restriction) as x_wrapper:
                for x_wrapper_local, component in zip(x_wrapper, self._solutions):
                    with component.vector.localForm() as component_local:
                        component_local[:] = x_wrapper_local

        def obj(self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec) -> np.float64:
            """Compute the norm of the residual."""
            self.F(snes, x, self._obj_vec)
            return self._obj_vec.norm()

        def F(self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec, b: petsc4py.PETSc.Vec) -> None:
            """Assemble the residual."""
            self.update_solutions(x)
            with b.localForm() as b_local:
                b_local.set(0.0)
            multiphenicsx.fem.petsc.assemble_vector_block(
                b, self._F, self._J, self._bcs, x0=x, scale=-1.0,
                restriction=self._restriction, restriction_x0=self._restriction)

        def J(self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec, A: petsc4py.PETSc.Mat, P_mat: petsc4py.PETSc.Mat) -> None:
            """Assemble the jacobian."""
            A.zeroEntries()
            if self._restriction is None:
                restriction = None
            else:
                restriction = (self._restriction, self._restriction)
            multiphenicsx.fem.petsc.assemble_matrix_block(
                A, self._J, self._bcs, diagonal=1.0, restriction=restriction)
            A.assemble()
            if self._P is not None:
                P_mat.zeroEntries()
                multiphenicsx.fem.petsc.assemble_matrix_block(P_mat, self._P, self._bcs, diagonal=1.0, restriction=restriction)
                P_mat.assemble()


    class NewtonBlockSolver:
        def __init__(self, comm: MPI.Intracomm, problem: NonlinearBlockProblem):
            """A Newton solver for non-linear block problems."""
            self.problem = problem
            self.snes = petsc4py.PETSc.SNES().create(comm)
            self.snes.setTolerances(max_it=50, atol=1e-10, rtol=1e-9)

            # Create matrix and vector to be used for assembly
            # of the non-linear problem
            self._A = multiphenicsx.fem.petsc.create_matrix_block(problem._J, restriction=(problem._restriction, problem._restriction) if problem._restriction is not None else None)
            self._b = multiphenicsx.fem.petsc.create_vector_block(problem._F, restriction=problem._restriction)
            self.solution = self.problem.create_snes_solution()

            self.snes.getKSP().setType("gmres")
            self.snes.getKSP().getPC().setType("hypre")
            self.snes.getKSP().getPC().setHYPREType("boomeramg")
            # self.snes.ksp.
            
            self.snes.setObjective(problem.obj)
            self.snes.setFunction(problem.F, self._b)
            self.snes.setJacobian(problem.J, J=self._A, P=None)
            self.snes.setMonitor(lambda _, it, residual: print(it, residual))

        def solve(self):
            """Solve non-linear problem into function u. Returns the number
            of iterations and if the solver converged."""
            # self.problem.init_solution(self.solution)
            self.snes.solve(None, self.solution)
            self.problem.update_solutions(self.solution)
            return self.snes.its, self.snes.converged


    if block:
        jacobian = [[ufl.derivative(r_i, u_i, du_i) for u_i, du_i in zip((c_e, phi_e),(dc_e, dphi_e))] for r_i in residuals]
        left_dofs = dfx.fem.locate_dofs_topological(PHI_E, 0, left_facets)
        bc = [dfx.fem.dirichletbc(ScalarType(0), left_dofs, PHI_E)]
    else:
        jacobian = ufl.derivative(sum(residuals), u_1, du)
        left_dofs = dfx.fem.locate_dofs_topological(FS.sub(1), 0, left_facets)
        bc = [dfx.fem.dirichletbc(ScalarType(0), left_dofs, FS.sub(1))]

    if block:
        # initial condition
        c_e.interpolate(lambda x: 0*x[0]+1000) 
        phi_e.interpolate(lambda x: 0*x[0])
        c_e_0.interpolate(c_e); phi_e_0.interpolate(phi_e)

        problem = NonlinearBlockProblem(residuals, (c_e, phi_e), bc, jacobian)
        solver = NewtonBlockSolver(mesh.comm, problem)

        # Initialize solution vector
        sol_c_e = np.empty((21,c_e_0.vector.array.size))
        sol_phi_e = np.empty((21,phi_e_0.vector.array.size))

        c_e_dofs = np.arange(c_e.vector.array.size)
        phi_e_dofs = np.arange(c_e.vector.array.size)

        real_c_e, real_phi_e = c_e, phi_e
    else:
        problem = dfx.fem.petsc.NonlinearProblem(dfx.fem.form(sum(residuals)), u_1, bc, dfx.fem.form(jacobian))
        solver = dfx.nls.petsc.NewtonSolver(mesh.comm, problem)

        # initial condition
        u_1.sub(0).interpolate(lambda x: 0*x[0]+1000) 
        u_1.sub(1).interpolate(lambda x: 0*x[0])
        u_0.interpolate(u_1)

        _, c_e_dofs = FS.sub(0).collapse()
        _, phi_e_dofs = FS.sub(1).collapse()

        # Initialize solution vector
        sol_c_e = np.empty((21,len(c_e_dofs)))
        sol_phi_e = np.empty((21,len(phi_e_dofs)))
        
        c_e, phi_e = u_1.split()
        c_e_0, phi_e_0 = u_0.split()

    I_app.value = _I_app

    ### Simulation ###

    # Apply current
    sol_c_e[0]=c_e.vector.array[c_e_dofs]
    sol_phi_e[0]=phi_e.vector.array[phi_e_dofs]
    for i in range(10):
        if block:
            n, converged = solver.solve()
        else:
            n, converged = solver.solve(u_1)
        print(n, converged)
        c_e_0.interpolate(c_e)
        phi_e_0.interpolate(phi_e)
        sol_c_e[i+1]=c_e.vector.array[c_e_dofs]
        sol_phi_e[i+1]=phi_e.vector.array[phi_e_dofs]

    I_app.value = 0

    # Relax
    for i in range(10):
        if block:
            n, converged = solver.solve()
        else:
            n, converged = solver.solve(u_1)
        print(n, converged)
        c_e_0.interpolate(c_e)
        phi_e_0.interpolate(phi_e)
        sol_c_e[i+11]=c_e.vector.array[c_e_dofs]
        sol_phi_e[i+11]=phi_e.vector.array[phi_e_dofs]

    voltage = sol_phi_e[:,0]-sol_phi_e[:,-1]
    return voltage

t = time()
v_1 = test_multiphenicsx(block=False)
t1 = time()
v_2 = test_multiphenicsx(block=True)
t2 = time()

print(f"dolfinx time: {t1-t}")
print(f"multiphenicsx time: {t2-t1}")
print(f"Error: {(v_1-v_2).max()}")