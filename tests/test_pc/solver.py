import dolfinx as dfx
import multiphenicsx.fem
from mpi4py import MPI
import petsc4py
import numpy as np
from tabulate import tabulate
import typing
from pcs import *

class NonlinearBlockProblem(object):
    """Define a nonlinear problem, interfacing with SNES."""

    def __init__(self, F, u, bcs, J, restriction = None, P = None, debug=True):
        self._F = dfx.fem.form(F)
        self._J = dfx.fem.form(J)
        self._obj_vec = multiphenicsx.fem.petsc.create_vector_block(self._F, restriction)
        self._solutions = u
        self._bcs = bcs
        self._restriction = restriction
        self._P = dfx.fem.form(P)
        self.debug=debug
        if debug:
            self.residuals = []
            self.steps = []

    def create_snes_solution(self) -> petsc4py.PETSc.Vec:
        """Create SNES solution vector"""
        x = multiphenicsx.fem.petsc.create_vector_block(self._F, restriction=self._restriction)
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(x, [c.function_space.dofmap for c in self._solutions], self._restriction) as x_wrapper:
            for x_wrapper_local, component in zip(x_wrapper, self._solutions):
                with component.vector.localForm() as component_local:
                    x_wrapper_local[:] = component_local
        return x

    def update_solutions(self, x: petsc4py.PETSc.Vec) -> None:
        """Update `self._solutions` with data in `x`."""
        x.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(x, [c.function_space.dofmap for c in self._solutions], self._restriction) as x_wrapper:
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
        multiphenicsx.fem.petsc.assemble_vector_block(b, self._F, self._J, self._bcs, x0=x, scale=-1.0,restriction=self._restriction, restriction_x0=self._restriction)

    def J(self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec, A: petsc4py.PETSc.Mat, P_mat: petsc4py.PETSc.Mat) -> None:
        """Assemble the jacobian."""
        A.zeroEntries()
        if self._restriction is None:
            restriction = None
        else:
            restriction = (self._restriction, self._restriction)
        multiphenicsx.fem.petsc.assemble_matrix_block(A, self._J, self._bcs, diagonal=1.0, restriction=restriction)
        A.assemble()
        if self._P is not None:
            P_mat.zeroEntries()
            multiphenicsx.fem.petsc.assemble_matrix_block(P_mat, self._P, self._bcs, diagonal=1.0, restriction=restriction)
            P_mat.assemble()
        if self.debug:
            self._view_residuals(snes, x)

    def _view_residuals(self, snes, x):
        residual = multiphenicsx.fem.petsc.assemble_vector_block(self._F,self._J,bcs=self._bcs,x0=x,scale=-1.0,restriction=self._restriction, restriction_x0=self._restriction)
        step = snes.ksp.getSolution().array if snes.ksp.getSolution().array.any() else np.zeros_like(residual.array)
        self.residuals.append([self.obj(snes,x)]+[np.linalg.norm(residual[dl]) for dl in self.dofs])
        self.steps.append([self.obj(snes,x)]+[np.linalg.norm(step[dl]) for dl in self.dofs])
        # self.aux.append([self.obj(snes,x)]+[dfx.fem.assemble_vector(F).norm() for F in self._F])

    def _print_residuals_norm(self):
        print('--------------- Problem Residuals ---------------')
        print(tabulate(self.residuals, headers=['objective']+[u.name for u in self._solutions],showindex=True))
        print('--------------------- End -----------------------')
        self.residuals = []
        print('\n--------------- Problem Steps ---------------')
        print(tabulate(self.steps, headers=['objective']+[u.name for u in self._solutions],showindex=True))
        print('-------------------- End --------------------')
        self.residuals = []


    def _prepare_variable_indices(self, x):
        self.dofs = []
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(
            x, [c.function_space.dofmap for c in self._solutions], self._restriction
        ) as x_wrapper:
            for index in range(x_wrapper._len):
                self.dofs.append(x_wrapper._restricted_index_sets[index].getIndices())

class NewtonBlockSolver:
    log = [("snes_monitor",":snes.log"),("ksp_monitor",":ksp.log"), ("snes_linesearch_monitor",":line_search.log"),
        ("options_view",True), ("options_left",True)]
    def __init__(self, comm: MPI.Intracomm, problem: NonlinearBlockProblem, pc=None, debug=True):
        """A Newton solver for non-linear block problems."""
        self.problem = problem
        self.snes = petsc4py.PETSc.SNES().create(comm)
        self.snes.setTolerances(max_it=50, atol=1e-10, rtol=1e-9)

        # Create matrix and vector to be used for assembly of the non-linear problem
        self._A = multiphenicsx.fem.petsc.create_matrix_block(problem._J, restriction=(problem._restriction, problem._restriction) if problem._restriction is not None else None)
        self._b = multiphenicsx.fem.petsc.create_vector_block(problem._F, restriction=problem._restriction)
        self.solution = self.problem.create_snes_solution()
        self.problem._prepare_variable_indices(self.solution)
        if pc=='none':
            self.snes.getKSP().setType("bcgs")
            self.snes.ksp.pc.setType("none")
        if pc=='mumps':
            self.snes.getKSP().setType("preonly")
            pc_mumps(self.snes.getKSP().getPC())
        if pc=='hypre' or pc is None:
            self.snes.getKSP().setType("bcgs")
            pc_boomerAMG(self.snes.getKSP().getPC())
            self._set_options([
                    # ("ksp_max_it", int(5e3)),
                    # ('pc_hypre_boomeramg_print_statistics', 1),
                    # ('snes_line_search_type', 'l2'),
                    # ("pc_hypre_boomeramg_strong_threshold", 0.9),
                    # ("pc_hypre_boomeramg_numfunctions", 3),
                    # ("pc_hypre_boomeramg_grid_sweeps_all", 3)
                ])
        if pc=='fieldsplit':
            self.snes.getKSP().setType("bcgs")
            fnames = [u.name for u in self.problem._solutions]
            pc_ffs_jacobi(self.snes.getKSP().getPC(), fnames, self.problem.dofs)
        if pc=='mixed':
            self.snes.getKSP().setType("bcgs")
            fnames = [u.name for u in self.problem._solutions]
            pc_mixed(self.snes.getKSP().getPC(), fnames, self.problem.dofs)
        if debug:
            self._set_options(self.log[:3]+self.log[-2:])
        self.snes.setObjective(problem.obj)
        self.snes.setFunction(problem.F, self._b)
        self.snes.setJacobian(problem.J, J=self._A, P=None)
        self.snes.setFromOptions()
        # self.snes.setMonitor(lambda _, it, residual: print(it, residual))
        
    def solve(self):
        """Solve non-linear problem into function u. Returns the number
        of iterations and if the solver converged."""
        # self.problem.init_solution(self.solution)
        self.snes.solve(None, self.solution)
        self.problem.update_solutions(self.solution)
        self.problem._view_residuals(self.snes,self.solution)
        self.problem._print_residuals_norm()
        if self.snes.converged:
            return self.snes.its, self.snes.converged
        else:
            raise Exception(f'Solver not Converged: {self._snes_reason_message()}')

    def _set_options(self, options:typing.List[typing.Tuple[str, typing.Any]]) -> None:
        petsc_options = petsc4py.PETSc.Options()
        for (name, value) in options:
            petsc_options.setValue(name, value)

    def _clear_options(self):
        petsc_options = petsc4py.PETSc.Options()
        for k in petsc_options.getAll():
            petsc_options.delValue(k)

    def _snes_reason_message(self):
        reason = self.snes.reason
        reasons = {
            2: "||F|| < atol",
            3: "||F|| < rtol*||F_initial||",
            4: "Newton computed step size small; || delta x || < stol || x ||",
            5: "maximum iterations reached",
            6: "Flag to break out of inner loop after checking custom convergence",
            -1: "the new x location passed the function is not in the domain of F",
            -2: "maximum function count reached",
            -3: "the linear solve failed",
            -4: "Fnorm NAN",
            -5: "maximum iterations reached",
            -6: "the line search failed",
            -7: "inner solve failed",
            -8: "|| J^T b || is small, implies converged to local minimum of F()",
            -9: "|| F || > divtol*||F_initial||",
            -10: "Jacobian calculation does not make sense",
            -11: "Trust Region delta",
            0: "Iterating"
        }
        assert reason in reasons.keys()
        message = reasons[reason] 
        if reason == -3:
            message += f". KSP: {self._ksp_reason_message()}"    
        return message

    def _ksp_reason_message(self):
        reasons = {
            2: "residual 2-norm decreased by a factor of rtol, from 2-norm of right hand side",
            3: "residual 2-norm less than abstol",
            4: "maximum PC iterations reached",
            5: "NewtonTR specific",
            6: "NewtonTR specific",
            -2: "KSP_DIVERGED_NULL",
            -3: "required more than its to reach convergence",
            -4: "residual norm increased by a factor of divtol",
            -5: "KSP_DIVERGED_BREAKDOWN",
            -6: "Initial residual is orthogonal to preconditioned initial residual. Try a different preconditioner, or a different initial Level",
            -7: "KSP_DIVERGED_NONSYMMETRIC",
            -8: "KSP_DIVERGED_INDEFINITE_PC",
            -9: "residual norm became NAN or Inf likely due to 0/0",
            -10: "KSP_DIVERGED_INDEFINITE_MAT",
            -11: "It was not possible to build or use the requested preconditioner. This is usually due to a zero pivot in a factorization.",
        }
        reason = self.snes.ksp.reason
        assert reason in reasons.keys()
        return reasons[reason]