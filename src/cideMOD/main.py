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
"""
This module is the core of cideMOD. Contain the class for handling
battery cell simulations
"""
import os
import ufl
import dolfinx as dfx
import warnings
from dolfinx.common import Timer, timed
from petsc4py.PETSc import ScalarType
from typing import Union, List, overload

from cideMOD.helpers.logging import VerbosityLevel, _print
from cideMOD.helpers.miscellaneous import add_to_results_folder, format_time
from cideMOD.numerics.fem_handler import (BlockFunctionSpace, assign, isinstance_dolfinx,
                                          assemble_scalar as assemble, block_derivative)
from cideMOD.numerics.time_scheme import TimeScheme
from cideMOD.numerics.time_stepper import ConstantTimeStepper, AdaptiveTimeStepper
from cideMOD.numerics.solver import NonlinearBlockProblem, NewtonBlockSolver
from cideMOD.numerics.triggers import Trigger, TriggerDetected, TriggerSurpassed, SolverCrashed
from cideMOD.cell.components import BatteryCell
from cideMOD.cell.parser import CellParser
from cideMOD.cell.dimensional_analysis import DimensionalAnalysis
from cideMOD.cell.warehouse import Warehouse
from cideMOD.cell.variables import ProblemVariables
from cideMOD.mesh.base_mesher import DolfinMesher, SubdomainMapper
from cideMOD.mesh.gmsh_adapter import GmshMesher
from cideMOD.models import BaseModelOptions


class Problem:
    """
    Class for handling battery cell simulations.

    Parameters
    ----------
    cell : CellParser
        Parser of the cell dictionary.
    model_options : ModelOptions
        Model options already configured by the user.
    """

    def __init__(self, cell: CellParser, model_options: BaseModelOptions):
        self.model_options = model_options
        self.cell_parser = cell
        self.verbose = model_options.verbose
        self.save_path = model_options.save_path
        self._comm = model_options.comm
        self._models = model_options._get_model_handler()
        self.user_bcs = {}
        self._ready = False

        # Build the mesh
        self._build_mesh()

        # Initialize the cell state
        self.set_cell_state()

        # Initialize the warehouse
        self._WH = Warehouse(self.save_path, self)

        # Setup the CellParser object
        self.cell_parser._set_problem(self)

        # Add files to the results folder
        self.save_path = self.model_options.save_path
        if self.save_path is not None:
            add_to_results_folder(
                self.save_path, files=[cell.get_dict(), self.model_options.dict(exclude={'comm'})],
                filenames=['params', 'simulation_options'])

    def set_cell_state(self, **kwargs):
        """
        This method set the current state of the cell.

        Parameters
        ----------
        kwargs: dict
            Dictionary containing the parameters that describe the cell
            state. To know more type :meth:`cideMOD.info(
            'set_cell_state', model_options=model_options)`
        """
        self._models.set_cell_state(self, **kwargs)

    def set_boundary_condition(self, bc):
        """
        This method set the given boundary condition.

        Parameters
        ----------
        bc: BaseBoundaryCondition
            Object that contain the required information to set the new
            boundary condition.
        """
        # TODO: Implement BaseBoundaryCondition and Heater. Think about including the default
        #       boundary conditions to a new dictionary self._bcs.
        raise NotImplementedError
        self._models.parse_boundary_condition(bc)
        self.user_bcs[bc._tag_] = bc

    def add_global_variables(self, name: Union[str, List[str]]):
        """
        This method add the requested variable to the list of requested
        global variables to be compute throughout the simulation.

        name: Union[str, List[str]]
            Name/s of the requested global variable/s. It should be
            provided by the active models.
        """
        # TODO: Provide the user a method like print_available_global_variables
        if not isinstance(name, list):
            name = [name]
        for var in name:
            self._WH.add_global_variable(var)

    def add_internal_variables(self, name: Union[str, List[str]]):
        """
        This method add the requested variable to the list of requested
        internal variables to be compute throughout the simulation.

        name: Union[str, List[str]]
            Name/s of the requested global variable/s. It should be
            provided by the active models.
        """
        # TODO: Provide the user a method like print_available_internal_variables
        if not isinstance(name, list):
            name = [name]
        for var in name:
            self._WH.add_internal_variable(var)

    def _build_mesh(self):
        """Builds the mesh of the cell"""
        # TODO: Mesh engine should be given by the model type, add it as a private attribute to
        #       ModelHandler and ask the models for the right mesher class. Ideally, there should
        #       be only one mesher class.
        mesh_engine = DolfinMesher if self.model_options.model == "P2D" else GmshMesher
        self.mesher = mesh_engine(options=self.model_options, cell=self.cell_parser)
        self.mesher.build_mesh(dimless_model=self.model_options.dimensionless)

    # ******************************************************************************************* #
    # ***                                     Setup Stage                                     *** #
    # ******************************************************************************************* #

    def setup(self):
        """
        Set-up the Problem object:
        - Build the FEM function spaces
        - Build the cell properties
        - Set-up internal state variables
        - Set-up Warehouse object
        - Build the variational formulation
        """

        basic_info = VerbosityLevel.BASIC_PROBLEM_INFO
        detailed_info = VerbosityLevel.DETAILED_PROGRESS_INFO
        self._print('Building problem setup', verbosity=basic_info)

        # Initialize the time scheme and the time
        self.time = 0.
        self._time = dfx.fem.Constant(self.mesher.mesh, ScalarType(0))
        self._DT = TimeScheme(self.model_options.time_scheme, self.mesher.mesh)
        # TODO: Initialize here also the time stepper if implemented

        # Setup independent variables and control variables
        self._vars = ProblemVariables(self)

        # Build the dimensional analysis
        self._print('- Building cell parameters ...', end='\r', verbosity=basic_info, only=True)
        if self.model_options.dimensionless:
            self._print(
                '- Performing the dimensional analysis ...', end='\r', verbosity=detailed_info)

        self._DA = DimensionalAnalysis(self.cell_parser, self.model_options)  # , self._vars)

        if self.model_options.dimensionless:
            self._print('- Performing the dimensional analysis - Done', verbosity=detailed_info)

        # Build FEM function spaces
        self._print('- Building FEM function spaces ...', end='\r', verbosity=detailed_info)
        self._build_function_spaces()
        self._print('- Building FEM function spaces - Done', verbosity=detailed_info)

        # Build control and state variables
        self._vars.setup(self)

        # Build cell parameters
        if self.verbose >= detailed_info:
            self._print('- Building cell parameters:', verbosity=basic_info)
        else:
            self._print('- Building cell parameters ...', end='\r', verbosity=basic_info)
        self.cell = BatteryCell(self)

        # Build dependent variables
        self._models.set_dependent_variables(self._vars, self.cell, self._DT, self)
        self._print('- Building cell parameters - Done', verbosity=basic_info)

        # Initial guess
        self._print('- Initializing state ... ', end='\r', verbosity=basic_info)
        self._models.initial_guess(self._f_0, self._vars, self.cell, self)
        assign(self.u_2, self.u_1)
        assign(self.u_0, self.u_1)
        self._print('- Initializing state - Done ', verbosity=basic_info)

        # Build weak formulation
        self._print('- Build variational formulation ... ', end='\r', verbosity=basic_info)
        self._build_weak_formulation()
        self._print('- Build variational formulation - Done ', verbosity=basic_info)

        # Build solver
        self._print('- Building solvers ... ', end='\r', verbosity=detailed_info)
        self._build_solvers()
        self._print('- Building solvers - Done', verbosity=detailed_info)

        # Warehouse setup
        self._WH.setup()

        # Setup active models if required
        self._models.setup(self)

        self._ready = True

    def _build_function_spaces(self):
        """Builds the function space"""
        # Get model function spaces
        P1 = dfx.fem.FunctionSpace(self.mesher.mesh, ('Lagrange', 1))
        P1_vec = dfx.fem.VectorFunctionSpace(self.mesher.mesh, ('Lagrange', 1))
        elements = self._models.set_state_variables(self.mesher, P1, P1_vec, self)

        # Define mixed function space
        self.W = BlockFunctionSpace(*zip(*elements))
        self.du = self.W.create_trial_function()
        self.u_2 = self.W.create_block_function()
        self.u_1 = self.W.create_block_function(suffix='_prev')
        self.u_0 = self.W.create_block_function(suffix='_2prev')
        self.test = self.W.create_test_function()

        self._f_1 = self.u_2
        self._f_0 = self.u_1

        # Additional fem functions
        self.V = dfx.fem.FunctionSpace(self.mesher.mesh, ('Lagrange', 1))
        self.V_vec = dfx.fem.VectorFunctionSpace(self.mesher.mesh, ('Lagrange', 1))
        self.V_0 = dfx.fem.FunctionSpace(self.mesher.mesh, ('DG', 0))
        self.V_0_vec = dfx.fem.VectorFunctionSpace(self.mesher.mesh, ('DG', 0))

        self.P1_map = SubdomainMapper(self.mesher.field_restrictions, self.V)
        self.P1_vec_map = SubdomainMapper(self.mesher.field_restrictions, self.V_vec)
        self.P0_map = SubdomainMapper(self.mesher.field_restrictions, self.V_0)
        self.P0_vec_map = SubdomainMapper(self.mesher.field_restrictions, self.V_0_vec)

    def _build_weak_formulation(self):
        self._solvers_info = self._models.get_solvers_info(self)

        eq = self._models.build_weak_formulation(
            self._solvers_info, self._vars, self.cell, self.mesher, self._DT, self.W, self)

        eq_transitory = self._models.build_weak_formulation_transitory(
            self._solvers_info, self._vars, self.cell, self.mesher, self.W, self)

        # TODO: Implement the stationary solver
        # eq_stationary = self._models.build_weak_formulation_stationary(
        #     self._solvers_info, self._vars, self.cell, self.mesher, self.W, self)

        self._solvers_info['solver']['equations'] = eq
        self._solvers_info['solver_transitory']['equations'] = eq_transitory
        # self._solvers_info['solver_stationary']['equations'] = eq_stationary

    def _build_solvers(self):
        monitor = self.verbose >= VerbosityLevel.DETAILED_SOLVER_INFO
        for solver_name, solver_info in self._solvers_info.items():
            state_vars = solver_info['state_variables']
            u = [self.u_2(state_var) for state_var in state_vars]
            du = [self.du(state_var) for state_var in state_vars]
            F_var = [solver_info['equations'][state_var] for state_var in state_vars]
            J_var = block_derivative(F_var, u, du)

            restrictions = [self.W.get_restriction(state_var) for state_var in state_vars]
            bcs = solver_info['equations'].get_boundary_conditions()

            if self.save_path is not None:
                save_path = os.path.join(self.save_path, solver_name)
            else:
                save_path = None

            block_problem = NonlinearBlockProblem(F_var, u, bcs, J_var, restrictions)
            block_solver = NewtonBlockSolver(
                self._comm, block_problem, monitor=monitor, save_path=save_path)

            if solver_info['options']:
                block_solver._set_options(solver_info['options'])

            solver_info['block_problem'] = block_problem
            solver_info['block_solver'] = block_solver

        self._solver = self._solvers_info['solver']['block_solver']
        self._solver_transitory = self._solvers_info['solver_transitory']['block_solver']
        # self._solver_stationary = self._solvers_info['solver_stationary']['block_solver']

    # ******************************************************************************************* #
    # ***                                  Simulation Stage                                   *** #
    # ******************************************************************************************* #

    @overload
    def solve(self, t_f=3600, store_delay=1, min_step=0.01, triggers: List[Trigger] = [],
              adaptive: bool = False, **kwargs):
        ...

    @overload
    def solve(self, t_f=3600, store_delay=1, initial_step=None, max_step=3600, min_step=0.01,
              triggers: List[Trigger] = [], adaptive: bool = True, time_adaptive_tol=1e-2,
              **kwargs):
        ...

    def solve(self, t_f=3600, store_delay=1, initial_step=None, max_step=3600, min_step=0.01,
              triggers: List[Trigger] = [], adaptive: bool = True, time_adaptive_tol=1e-2,
              **kwargs):
        """
        Perform a simulation step. For more complex inputs it is
        recommended to use several calls to this method.

        Parameters
        ----------
        t_f : float, optional
            The maximum duration of the simulation. Defaults to 3600.
        store_delay : int, optional
            The delay to apply between consecutive saves of the internal
            variables, in number of timesteps. Defaults to 1.
        initial_step : float, optional
            Initial timestep length. If not given, the timestep chose is
            the minimum. Default to None.
        max_step : float, optional
            Maximum timestep length for adaptive solver in seconds.
            Default to 3600.
        min_step : float, optional
            Minimum timestep length for adaptive solver in seconds.
            Default to 0.01.
        triggers : list, optional
            List of Triggers to check during runtime. Default to [].
        adaptive : bool, optional
            Whether to use adaptive timestepping or not. Default to
            True.
        time_adaptive_tol : Union[float,int]
            Tolerance of the time-adaptive scheme. Defaults to 1e-2.
        kwargs : dict
            Control variables of the problem. Could be constant or a
            time-dependent expression. To know the required control
            variables type `problem.print_control_variables_info()`.

        Returns
        -------
        Union[int, Exception]
            The status of the simulation. If there is an error, the
            Exception object is returned. Otherwise return 0.
        """
        if not self._ready:
            self.setup()

        if initial_step is None:
            initial_step = min_step
        initial_step = max(initial_step, min_step)
        self._WH.set_delay(store_delay)

        if adaptive:
            time_stepper = AdaptiveTimeStepper(self, dt=initial_step, min_step=min_step,
                                               max_step=max_step, t_max=t_f, tol=time_adaptive_tol,
                                               triggers=triggers, **kwargs)
        else:
            time_stepper = ConstantTimeStepper(self, dt=initial_step, triggers=triggers, **kwargs)

        # Initialize the cell state attribute
        self.state = {'time': self.time}

        self._print('Solving ...', only=True)

        timer = Timer('Simulation time')
        timer.start()

        self._solver._set_options([('snes_lag_jacobian', 1), ('snes_max_it', 50)])
        _pad = 0
        while self.time < t_f:
            # if it==1:
            #     self._solver._set_options([('snes_lag_jacobian', 5), ('snes_max_it', 30)])

            errorcode = time_stepper.timestep()
            errorcode_ex = self._explicit_processing()

            if self.verbose >= VerbosityLevel.BASIC_PROGRESS_INFO:
                log_msg = f"Time: {format_time(self.state['time'])}  "
                if _pad < len(log_msg):
                    _pad = len(log_msg)
                log_msg = (log_msg.ljust(_pad)
                           + '  '.join([f"{k.capitalize()}: {v:.4g}"
                                        for k, v in self.state.items() if k != 'time']))
                _print(log_msg, end='\r', comm=self._comm)

            if errorcode != 0 or errorcode_ex != 0:
                if (errorcode_ex == 0
                        and isinstance(errorcode, (TriggerDetected, TriggerSurpassed))):
                    self._advance_problem()
                break
            else:
                self._advance_problem()

        if self.time >= t_f:
            self._print(f"Reached max time {self.time:.2f}", end="\n\n")
        timer.stop()
        return self.exit(errorcode if errorcode != 0 else errorcode_ex)

    @timed('Explicit Processing')
    def _explicit_processing(self):
        # TODO: Allow explicit models to return an status variable to stop the solve.
        self._models.explicit_update(self)
        return 0

    def _advance_problem(self):
        self.time += self.get_timestep()
        self._time.value = self.time
        self._WH.store(self.time)
        assign(self.u_0, self.u_1)
        assign(self.u_1, self.u_2)

    def set_timestep(self, timestep):
        timestep_ = self._DA.scale_variable('time', timestep)
        self._DT.set_timestep(timestep_)

    def get_timestep(self):
        timestep_ = self._DT.get_timestep()
        return self._DA.unscale_variable('time', timestep_)

    def exit(self, errorcode):
        if self.model_options.save_on_exit:
            self._WH.write_globals(self.model_options.clean_on_exit,
                                   individual_txt=self.model_options.globals_txts)
        if self.model_options.raise_errors_on_exit and isinstance(errorcode, SolverCrashed):
            raise errorcode
        return errorcode

    # ******************************************************************************************* #
    # ***                                    Problem Utils                                    *** #
    # ******************************************************************************************* #

    def update_dynamic_parameters(self, dynamic_parameters):
        """
        This method updates the values of the dynamic parameters of the
        cell and the components it has.

        Parameters
        ----------
        dynamic_parameters: dict
            Dictionary containing the dynamic parameter names and values
            to be updated.

        Notes
        -----
        To update a dynamic parameter of the cell:

        >>> problem.update_dynamic_parameters({'area': 0.1})

        or

        >>> problem.update_dynamic_parameters({'cell.area': 0.1})

        To update a dynamic parameter of a cell component:

        >>> problem.update_dynamic_parameters({'anode.thickness': 1e-4})
        """
        if not dynamic_parameters:
            return

        _dynamic_parameters = dict()
        for name, value in dynamic_parameters.items():
            parameter = self.cell_parser.get_parameter(name)
            if self._ready and isinstance_dolfinx(value):
                raise RuntimeError(parameter._get_error_msg(
                    reason=("Unable to update the dynamic parameter as a "
                            + "new dolfinx object after setting up the problem"),
                    action='handling'
                ))
            _dynamic_parameters[str(parameter)] = value

        self.cell_parser.update_parameters(_dynamic_parameters)

    def reset(self, new_parameters: dict = None, triggers: List[Trigger] = None,
              save_path: str = None, save_config=True, prefix='results_'):
        """
        This method resets the problem in order to be ready for running
        another simulation with the same initial conditions, and maybe
        using different parameters.

        Parameters
        ----------
        new_parameters: Dict[str, float], optional
            Dictionary containing the cell parameters to be updated.
        triggers: List[Triggers]
            List containing the triggers that should be reset.
        save_path: str, optional
            Path to the new results folder.
        save_config
            Whether to save the parameter and simulation options to the
            results folder. Defaults to True.

        Notes
        -----
        This method is used to avoid the generation of multiple Problem
        objects when running multiple simulations, for example when
        performing optimizations.
        """
        timer = Timer('Problem Reset')

        basic_info = VerbosityLevel.BASIC_PROBLEM_INFO
        self._print('Reseting the problem ...', end='\r', verbosity=basic_info)

        # Reset triggers
        if triggers is not None:
            for t in triggers:
                t.reset()

        # Update the dynamic parameters
        last_dims = self.mesher.get_dims()
        self.update_dynamic_parameters(new_parameters)

        # Check if a deep reset is needed (repeat the setup stage)
        deep_reset = False
        if new_parameters and self._ready:
            # TODO: Just check if the mesh is dimensional
            if isinstance(self.mesher, GmshMesher) and self.model_options.dimensionless:
                deep_reset = not all([old == new
                                      for old, new in zip(last_dims, self.mesher.get_dims())])
            else:
                self.mesher.scale = self.mesher.get_dims()[0]

        # New results folder
        if save_path:
            self.save_path = self.update_save_path(
                save_path, save_config=save_config, prefix=prefix)

        # Build the new mesh if required
        # TODO: Check if the mesh is dimensional
        if deep_reset and self.model_options.dimensionless:
            # timer.stop()
            self._build_mesh()
            # timer.resume()

        # Reset internal classes and attributes
        if self._ready:
            self._WH.reset(self.save_path, deep_reset=deep_reset)
        self._models.reset(self, deep_reset=deep_reset)

        # Reset internal classes and attributes
        if not self._ready:
            timer.stop()
        elif deep_reset:
            timer.stop()
            self.setup()
        else:
            self.time = 0.
            self._time.value = 0.

            # Reset block functions
            self.u_1.clear()

            # Reset mesh dependent parameters
            # TODO: It refers to parameters defined with dolfinx.fem.Function, see CellParameter.

            # Initial state
            self._models.initial_guess(self._f_0, self._vars, self.cell, self)
            assign(self.u_2, self.u_1)
            assign(self.u_0, self.u_1)
            timer.stop()

            # Reset solvers
            for solver_info in self._solvers_info.values():
                block_solver = solver_info['block_solver']
                block_solver.reset()

            self._print('Reseting the problem - Done', verbosity=basic_info)

    def update_save_path(self, save_path, save_config=True, files=[], filenames=[],
                         prefix='results_'):
        save_path = self.model_options._update_save_path(save_path, prefix=prefix)
        if save_path == self.save_path and not self.model_options.overwrite:
            warnings.warn(
                f"The given save path has already been set: '{save_path}'", RuntimeWarning)
            return
        else:
            self.save_path = save_path
        if save_config:
            files.extend([self.cell_parser.get_dict(), self.model_options.dict(exclude={'comm'})])
            filenames.extend(['params', 'simulation_options'])
        if files:
            add_to_results_folder(self.save_path, files, filenames, self._comm, overwrite=False)
        self._WH._update_save_path(self.save_path)
        if self._ready:
            for solver_name, solver_info in self._solvers_info.items():
                block_solver = solver_info['block_solver']
                if self.save_path is not None:
                    solver_save_path = os.path.join(self.save_path, solver_name)
                    block_solver._set_save_path(solver_save_path)
                    if block_solver.monitor:
                        # TODO: Deactivate previous monitors
                        block_solver._set_monitor()
                else:
                    # TODO: Deactivate previous monitors
                    block_solver.save_path = None
        return self.save_path

    def get_global_variable(self, name: str):
        """
        Get the values of a global variable over the timesteps

        Parameters
        ----------
        name : str
            Name of the global variable

        Returns
        -------
        list
            List of values
        """
        return self._WH.get_global_variable(name)

    def get_global_variable_fnc(self, name: str):
        return self._WH.get_global_variable_fnc(name)

    def get_global_variable_value(self, name: str):
        return self._WH.get_global_variable_value(name)

    def get_avg(self, variable, domain: Union[ufl.Measure, str], integral_type='x'):
        """
        Get the average of the variable over the given subdomain or
        surface

        Parameters
        ----------
        variable : Union[ufl.Integral, ufl.Operator, dolfinx.Function]
            Expression of the variable to be averaged
        domain : Union[Measure, str]
            Measure or tag of the domain to integrate over
        integral_type : str
            Integral type. Only used if `domain` is a string.
            Available options: 'x', 's' and 'S'

        Returns
        -------
        float
            Averaged variable over the given subdomain or surface
        """
        if isinstance(variable, (float, int)):
            return variable
        elif isinstance(domain, ufl.Measure):
            dx = domain
            volume = self.mesher.volumes[self.mesher.get_measures().index(dx)]
        else:
            dx = self.mesher.get_measures()._asdict()[f'{integral_type}_{domain}']
            volume = self.mesher.volumes._asdict()[f'{integral_type}_{domain}']
        if isinstance(variable, (ufl.Form, dfx.fem.Form)):
            return assemble(variable) / volume
        else:
            return assemble(variable * dx) / volume

    def print_control_variables_info(self):
        # TODO: Implement this method, maybe using the models_info method
        raise NotImplementedError("Not implemented yet.")

    def print_available_outputs(self):
        # TODO: Use model_info for this method.
        # TODO: Different colors if the are already selected.
        info = ("Available global variables: '"
                + "' '".join(self._WH._outputs_info['globals'].keys()) + "'\n"
                + "Available internal variables: '"
                + "' '".join(self._WH._outputs_info['internals'].keys()) + "'")
        self._print(info, verbosity=-1)

    def _monitor_last_iteration(self, solvers='solver', set_monitor=True, plot=True, clean=True):
        """
        This method monitors the last iteration of the specified solver.

        Parameters
        ----------
        solver: str
            Name of the solver. Default to `solver`.

        Notes
        -----
        Normally used to debug a iteration where the solver crashed.
        """
        # Reset solution to the previous timestep
        assign(self._f_1, self._f_0)

        if isinstance(solvers, str):
            solvers = [solvers]
        for solver in solvers:
            # Update solver configuration
            block_problem = self._solvers_info[solver]['block_problem']
            block_solver = self._solvers_info[solver]['block_solver']
            block_problem._set_options(inspect_residuals=True, inspect_jacobian=True,
                                       print_residuals=True, plot_jacobian=True)

            if set_monitor and block_solver.save_path is not None and not block_solver.monitor:
                block_solver.monitor = True
                block_solver._set_save_path()
                block_solver._set_monitor()

            # Solve
            try:
                out = block_solver.solve(plot=plot, clean=clean)
            except RuntimeError as e:
                out = e
                break

            # Clean
            if clean:
                # TODO: Deactivate block_solver monitor
                block_problem._set_options(inspect_residuals=False, inspect_jacobian=False,
                                           print_residuals=False, plot_jacobian=False)
                block_solver.monitor = False

        return out

    def _print(self, *args, verbosity: int = VerbosityLevel.BASIC_PROGRESS_INFO,
               only=False, **kwargs):
        if not only and self.verbose >= verbosity or self.verbose == verbosity:
            return _print(*args, comm=self._comm, **kwargs)
