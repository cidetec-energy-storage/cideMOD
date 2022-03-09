import dolfinx as dfx
import multiphenicsx.fem
import typing
import ufl
import petsc4py
import numpy as np
from mpi4py import MPI


class NonlinearBlockProblem(object):
    """Define a nonlinear problem, interfacing with SNES."""

    def __init__(
        self,
        F: typing.List[ufl.Form],
        u: typing.Tuple[dfx.fem.Function],
        bcs: typing.List[dfx.fem.DirichletBCMetaClass],
        J: typing.List[typing.List[ufl.Form]],
        restriction: typing.Optional[
            typing.List[multiphenicsx.fem.DofMapRestriction]
        ] = None,
        P: typing.Optional[ufl.Form] = None,
    ) -> None:
        self._F = dfx.fem.form(F)
        self._J = dfx.fem.form(J)
        self._obj_vec = multiphenicsx.fem.petsc.create_vector_block(
            self._F, restriction
        )
        self._solutions = u
        self._bcs = bcs
        self._restriction = restriction
        self._P = dfx.fem.form(P)

    def create_snes_solution(self) -> petsc4py.PETSc.Vec:
        """Create SNES solution vector"""
        x = multiphenicsx.fem.petsc.create_vector_block(
            self._F, restriction=self._restriction
        )
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(
            x, [c.function_space.dofmap for c in self._solutions], self._restriction
        ) as x_wrapper:
            for x_wrapper_local, component in zip(x_wrapper, self._solutions):
                with component.vector.localForm() as component_local:
                    x_wrapper_local[:] = component_local
        return x

    def update_solutions(self, x: petsc4py.PETSc.Vec) -> None:
        """Update `self._solutions` with data in `x`."""
        x.ghostUpdate(
            addv=petsc4py.PETSc.InsertMode.INSERT,
            mode=petsc4py.PETSc.ScatterMode.FORWARD,
        )
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(
            x, [c.function_space.dofmap for c in self._solutions], self._restriction
        ) as x_wrapper:
            for x_wrapper_local, component in zip(x_wrapper, self._solutions):
                with component.vector.localForm() as component_local:
                    component_local[:] = x_wrapper_local

    def obj(self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec) -> np.float64:
        """Compute the norm of the residual."""
        self.F(snes, x, self._obj_vec)
        return self._obj_vec.norm()

    def F(
        self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec, b: petsc4py.PETSc.Vec
    ) -> None:
        """Assemble the residual."""
        self.update_solutions(x)
        with b.localForm() as b_local:
            b_local.set(0.0)
        multiphenicsx.fem.petsc.assemble_vector_block(
            b,
            self._F,
            self._J,
            self._bcs,
            x0=x,
            scale=-1.0,
            restriction=self._restriction,
            restriction_x0=self._restriction,
        )

    def J(
        self,
        snes: petsc4py.PETSc.SNES,
        x: petsc4py.PETSc.Vec,
        A: petsc4py.PETSc.Mat,
        P_mat: petsc4py.PETSc.Mat,
    ) -> None:
        """Assemble the jacobian."""
        A.zeroEntries()
        if self._restriction is None:
            restriction = None
        else:
            restriction = (self._restriction, self._restriction)
        multiphenicsx.fem.petsc.assemble_matrix_block(
            A, self._J, self._bcs, diagonal=1.0, restriction=restriction
        )
        A.assemble()
        if self._P is not None:
            P_mat.zeroEntries()
            multiphenicsx.fem.petsc.assemble_matrix_block(
                P_mat, self._P, self._bcs, diagonal=1.0, restriction=restriction
            )
            P_mat.assemble()


class NewtonBlockSolver:
    def __init__(self, comm: MPI.Intracomm, problem: NonlinearBlockProblem, conf='hypre'):
        """A Newton solver for non-linear block problems."""
        self.problem = problem
        self.snes = petsc4py.PETSc.SNES().create(comm)
        self.snes.setType('newtontr')
        self.snes.setTolerances(max_it=20, atol=1e-6, rtol=1e-9)

        # Create matrix and vector to be used for assembly
        # of the non-linear problem
        self._A = multiphenicsx.fem.petsc.create_matrix_block(
            problem._J,
            restriction=(problem._restriction, problem._restriction)
            if problem._restriction is not None
            else None,
        )
        self._b = multiphenicsx.fem.petsc.create_vector_block(
            problem._F, restriction=problem._restriction
        )
        self.solution = self.problem.create_snes_solution()
        if conf == 'mumps':
            self.snes.getKSP().setType("preonly")
            self.snes.getKSP().getPC().setType("lu")
            self.snes.getKSP().getPC().setFactorSolverType("mumps")
        elif conf == 'hypre':
            self.snes.getKSP().setType("bcgs")
            self.snes.getKSP().getPC().setType("hypre")
            self.snes.getKSP().getPC().setHYPREType("boomeramg")


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
        if self.snes.converged:
            return self.snes.its, self.snes.converged
        else:
            raise Exception('Solver not Converged')

    def reset_snes_solution(self):
        with self.solution.localForm() as b_local:
            b_local.set(0.0)
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(
            self.solution, [c.function_space.dofmap for c in self.problem._solutions], self.problem._restriction
        ) as x_wrapper:
            for x_wrapper_local, component in zip(x_wrapper, self.problem._solutions):
                with component.vector.localForm() as component_local:
                    x_wrapper_local[:] = component_local