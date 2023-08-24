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
import ufl
import petsc4py
import numpy as np
import dolfinx as dfx
import multiphenicsx.fem
import multiphenicsx.fem.petsc

from dolfinx.common import timed
from mpi4py import MPI
from tabulate import tabulate
from matplotlib import pyplot as plt
from typing import Union, Optional, Tuple, List, Any

from cideMOD.numerics.helper import analyze_jacobian, estimate_condition_number
from cideMOD.helpers.miscellaneous import plot_jacobian, init_results_folder, save_plot


class NonlinearBlockProblem(object):
    """Define a nonlinear problem, interfacing with SNES."""

    def __init__(
        self,
        F: List[ufl.Form],
        u: Tuple[dfx.fem.Function],
        bcs: List[dfx.fem.DirichletBC],
        J: List[List[ufl.Form]],
        restriction: Optional[
            List[multiphenicsx.fem.DofMapRestriction]
        ] = None,
        P: Optional[ufl.Form] = None,
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

        self._set_options()
        self._clear_history()

    def _set_options(self, inspect_residuals=False, inspect_jacobian=False,
                     print_residuals=False, plot_jacobian=False):
        self._inspect_residuals = inspect_residuals
        self._inspect_jacobian = inspect_jacobian
        self._print_residuals = print_residuals
        self._plot_jacobian = plot_jacobian

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

    @timed('Update solutions')
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

    @timed('Eval Objective')
    def obj(self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec) -> np.float64:
        """Compute the norm of the residual."""
        self.F(snes, x, self._obj_vec)
        return self._obj_vec.norm()

    @timed('Build Residual')
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

    @timed('Build Jacobian')
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
        # FIXME: Move this to the solver's interface
        self._on_nonlinear_iteration_begin(snes, x, A)

    def _on_nonlinear_iteration_begin(self, snes, x, A=None):
        """
        Callback to be called at the beginning of every nonlinear
        solver's iteration
        """
        if self._inspect_residuals:
            self._view_residuals(snes, x)
        if self._inspect_jacobian and A is not None:
            self._view_jacobian(A)
        # NOTE: To see the jacobian each non linear solver iteration
        # if self._plot_jacobian and A is not None:
        #     plot_jacobian(self, x, A)

    def _on_solve_end(self, snes, solution, A=None, plot=False, clean=True,
                      save_path='', save_fig=False):
        """
        Callback to be called at the end of the problem resolution
        """
        self.update_solutions(solution)
        if self._inspect_residuals:
            self._view_residuals(snes, solution)
        if self._print_residuals:
            self._print_residuals_norm(plot, save_path=save_path, save_fig=save_fig)
        if self._plot_jacobian and A is not None:
            plot_jacobian(self, solution, A, save_path=save_path, save_fig=save_fig)
        if clean:
            self._clear_history()

    def _view_jacobian(self, A):
        self.conditions.append(estimate_condition_number(A))
        self.distances.append(A.norm())

    def _view_residuals(self, snes, x):
        residual = multiphenicsx.fem.petsc.assemble_vector_block(
            self._F, self._J, bcs=self._bcs, x0=x, scale=-1.0, restriction=self._restriction,
            restriction_x0=self._restriction)
        step = (snes.ksp.getSolution().array
                if snes.ksp.getSolution().array.any() else np.zeros_like(residual.array))
        self.residuals.append(
            [self.obj(snes, x)] + [np.linalg.norm(residual[dl]) for dl in self.dofs])
        self.steps.append([self.obj(snes, x)] + [np.linalg.norm(step[dl]) for dl in self.dofs])

    def _print_residuals_norm(self, plot=True, save_path='', save_fig=False):
        headers = ['objective'] + [u.name for u in self._solutions]
        if plot:
            cmap = plt.get_cmap('jet')
            plt.figure()
            plt.title("Non-linear solver residuals")
            for i, label in enumerate(headers):
                color = cmap(i / len(headers)) if len(headers) > 10 else None
                plt.plot([r[i] for r in self.residuals], label=label, color=color)
            plt.yscale('log')
            plt.ylabel('residual')
            plt.xlabel('Newton iteration')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            if save_fig:
                save_plot(filename='res', isuffix='_{i}', foldername=save_path)
            plt.show()
            plt.figure()
            plt.title("Linear solver solutions (x-steps)")
            for i, label in enumerate(headers[1:], 1):
                color = cmap(i / len(headers)) if len(headers) > 10 else None
                plt.plot([r[i] for r in self.steps], label=label, color=color)
            plt.yscale('log')
            plt.ylabel('step size')
            plt.xlabel('Newton iteration')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            if save_fig:
                save_plot(filename='steps', isuffix='_{i}', foldername=save_path)
            plt.show()

        else:
            print('--------------- Problem Residuals ---------------')
            print(tabulate(
                self.residuals, headers=['objective'] + [u.name for u in self._solutions],
                showindex=True))
            print('--------------------- End -----------------------')
            print('\n--------------- Problem Steps ---------------')
            print(tabulate(
                self.steps, headers=['objective'] + [u.name for u in self._solutions],
                showindex=True))
            print('-------------------- End --------------------')

    def _clear_history(self):
        self.steps = []
        self.residuals = []
        self.conditions = []
        self.distances = []

    def _prepare_variable_indices(self, x):
        self.dofs = []
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(
            x, [c.function_space.dofmap for c in self._solutions], self._restriction
        ) as x_wrapper:
            for index in range(x_wrapper._len):
                self.dofs.append(x_wrapper._restricted_index_sets[index].getIndices())


class NewtonBlockSolver:
    log = [
        ("snes_monitor", ":snes_log.txt"),
        ("ksp_monitor", ":ksp_log.txt"),
        ("snes_linesearch_monitor", ":line_search_log.txt"),
        ("options_view", True),
        ("options_left", True)
    ]

    def __init__(self, comm: MPI.Intracomm, problem: NonlinearBlockProblem,
                 conf='mumps', monitor=False, save_path=None):
        """A Newton solver for non-linear block problems."""
        self.problem = problem
        self._comm = comm
        # Clear existing snes option settings
        self._clear_options()
        # Add save_path to logging files
        self.monitor = monitor
        self.save_path = save_path
        self._set_save_path(save_path)
        # Setup snes object
        self.snes = petsc4py.PETSc.SNES().create(comm)
        # Setup monitor
        if monitor:
            self._set_monitor()
        self.snes.setType('newtonls')
        # self.snes.setType('newtontrdc')
        # self._set_options([
        #     ('trdc_auto_scale_multiphase',True),
        #     ('trdc_use_cauchy',True)
        #     ], update=False)
        self.snes.setTolerances(max_it=50, atol=1e-6, rtol=1e-9)

        # Create matrix and vector to be used for assembly
        # of the non-linear problem
        self._A = multiphenicsx.fem.petsc.create_matrix_block(
            problem._J,
            restriction=((problem._restriction, problem._restriction)
                         if problem._restriction is not None else None),
        )
        self._b = multiphenicsx.fem.petsc.create_vector_block(
            problem._F, restriction=problem._restriction
        )
        self.solution = self.problem.create_snes_solution()
        self.problem._prepare_variable_indices(self.solution)
        if conf == 'mumps':
            self.snes.getKSP().setType("preonly")
            self.snes.getKSP().getPC().setType("lu")
            self.snes.getKSP().getPC().setFactorSolverType("mumps")
            # self._set_options([
            # # ("mat_mumps_use_omp_threads",4),
            # # ('snes_line_search_type', 'basic')
            # ], update=False)
        elif conf == 'hypre':
            self.snes.getKSP().setType("minres")
            self.snes.getKSP().getPC().setType("ilu")
            self._set_options([
                ("ksp_diagonal_scale", True), ("ksp_diagonal_scale_fix", True)
            ], update=False)
            # self.snes.getKSP().getPC().setHYPREType("boomeramg")
            # self._set_options([
            #     ("ksp_max_it", int(5e3)),
            #     ("pc_hypre_boomeramg_numfunctions", len(self.problem._F)),
            #     # ('pc_hypre_boomeramg_print_statistics', 1),
            #     ("pc_hypre_boomeramg_strong_threshold", 0.7),
            #     # ("pc_hypre_boomeramg_grid_sweeps_all", 3)
            # ], update=False)
        self.snes.setObjective(problem.obj)
        self.snes.setFunction(problem.F, self._b)
        self.snes.setJacobian(problem.J, J=self._A, P=None)
        self.snes.setFromOptions()

    def solve(self, plot=False, clean=True):
        """Solve non-linear problem into function u. Returns the number
        of iterations and if the solver converged."""
        # self.problem.init_solution(self.solution)
        self.snes.solve(None, self.solution)
        self.problem._on_solve_end(self.snes, self.solution, self._A, plot, clean, self.save_path,
                                   save_fig=self.save_path is not None and self.monitor)
        if self.snes.converged:
            return self.snes.its, self.snes.converged
        else:
            raise RuntimeError(f"Solver not Converged: {self._snes_reason_message()}")

    def reset_snes_solution(self):
        with self.solution.localForm() as b_local:
            b_local.set(0.0)
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(
                self.solution, [c.function_space.dofmap for c in self.problem._solutions],
                self.problem._restriction) as x_wrapper:
            for x_wrapper_local, component in zip(x_wrapper, self.problem._solutions):
                with component.vector.localForm() as component_local:
                    x_wrapper_local[:] = component_local

    def reset(self):
        """
        This method resets the solver in order to be ready for running
        another simulation with the same initial configuration.
        """
        # with self._b.localForm() as b_local:
        #     b_local.set(0.0)
        # self._A.zeroEntries()
        # TODO: reset self._P
        self.reset_snes_solution()
        self.problem._clear_history()

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
            -6: ("Initial residual is orthogonal to preconditioned initial residual. "
                 + "Try a different preconditioner, or a different initial Level"),
            -7: "KSP_DIVERGED_NONSYMMETRIC",
            -8: "KSP_DIVERGED_INDEFINITE_PC",
            -9: "residual norm became NAN or Inf likely due to 0/0",
            -10: "KSP_DIVERGED_INDEFINITE_MAT",
            -11: ("It was not possible to build or use the requested preconditioner. "
                  + "This is usually due to a zero pivot in a factorization."),
        }
        reason = self.snes.ksp.reason
        assert reason in reasons.keys()
        return reasons[reason]

    def _view(self, filename) -> None:
        viewer = petsc4py.PETSc.Viewer().createASCII(filename, 'w')
        self.snes.view(viewer)

    def _set_options(self, options: Union[List[Tuple[str, Any]], dict], update=True) -> None:
        petsc_options = petsc4py.PETSc.Options()
        _options = options.items() if isinstance(options, dict) else options
        for name, value in _options:
            petsc_options.setValue(name, value)
        if update:
            self.snes.setFromOptions()

    def _set_monitor(self, clear=True):
        petsc4py.PETSc.Log.begin()
        options = self.log
        self._set_options(options)
        if clear:
            self._clear_options([option for option, value in options])
        self.monitor = True

    def _clear_options(self, options: List[str] = None):
        petsc_options = petsc4py.PETSc.Options()
        for k in options or petsc_options.getAll():
            petsc_options.delValue(k)

    def _profile(self, filename) -> None:
        viewer = petsc4py.PETSc.Viewer().createASCII(filename, 'w')
        petsc4py.PETSc.Log.view(viewer)

    def _set_save_path(self, save_path=None):
        if save_path is None:
            save_path = self.save_path  # Assume it is not None
        else:
            self.save_path = save_path
        if self.monitor:
            self.save_path = init_results_folder(
                save_path, prefix="log_", overwrite=False, comm=self._comm, verbose=False)
            for i in range(len(self.log)):
                if self.log[i][0] in ("snes_monitor", "ksp_monitor", "snes_linesearch_monitor"):
                    self.log[i] = (
                        self.log[i][0],
                        ":" + os.path.join(self.save_path, os.path.basename(self.log[i][1][1:]))
                    )
