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

from abc import ABC, abstractmethod
from dolfinx.common import Timer, timed

from cideMOD.helpers.logging import VerbosityLevel, _print
from cideMOD.helpers.miscellaneous import format_time
from cideMOD.numerics.fem_handler import assign
from cideMOD.numerics.triggers import SolverCrashed, TriggerDetected, TriggerSurpassed


class BaseTimeStepper(ABC):
    """
    Abstract base class for TimeStepper classes
    """

    def __init__(self, problem, dt=1, triggers=[], initialize=True, **kwargs):
        self.problem = problem
        self.dt = dt
        self.triggers = triggers
        self.initialize = initialize
        self._models = problem._models
        self._comm = problem._comm
        self.verbose = problem.verbose

        self.problem.set_timestep(self.dt)
        self._models.update_control_variables(problem._vars, problem, **kwargs)

    def accept_timestep(self, timer, errorcode):
        problem = self.problem
        if isinstance(errorcode, SolverCrashed):
            timer.stop()
            return errorcode
        problem.state = self._models.get_cell_state(problem)
        try:
            for t in self.triggers:
                t.check(problem.state)
        except TriggerSurpassed as e:
            timer.stop()
            new_tstep = e.new_tstep(problem.get_timestep())
            if new_tstep > 3e-16:
                assign(problem.u_2, problem.u_1)  # Reset solution to avoid possible NaN values
                self.dt = new_tstep
                problem.set_timestep(new_tstep)
                errorcode = self.linear_timestep()
            else:
                errorcode = e
                self._print(f"{str(e)} at {format_time(problem.state['time'])}", end="\n\n")
            return errorcode
        except TriggerDetected as e:
            errorcode = e
            self._print(f"{str(e)} at {format_time(problem.state['time'])}", end="\n\n")
        timer.stop()
        return errorcode

    def basic_timestep(self):
        timer = Timer('Basic TS')
        timer.start()
        try:
            self.tstep_implicit()
            timer.stop()
            return 0
        except Exception as e:
            timer.stop()
            return SolverCrashed(e)

    def tstep_implicit(self):
        problem = self.problem
        if self.initialize:
            self._print("Initializing solution", verbosity=VerbosityLevel.DETAILED_PROGRESS_INFO)
            problem._solver_transitory.solve()
            if problem.time == 0:
                problem._WH.store(problem.time)
            self.initialize = False
            self._print("Solving ...", verbosity=VerbosityLevel.DETAILED_PROGRESS_INFO)
        problem._solver.solve()

    @abstractmethod
    def linear_timestep(self):
        """Solve a linear time step"""

    @abstractmethod
    def timestep(self):
        """Solve the current time step"""

    def _print(self, *args, verbosity: int = VerbosityLevel.BASIC_PROGRESS_INFO,
               only=False, **kwargs):
        if not only and self.verbose >= verbosity or self.verbose == verbosity:
            return _print(*args, comm=self._comm, **kwargs)


class ConstantTimeStepper(BaseTimeStepper):

    def timestep(self):
        timer = Timer('Constant TS')
        timer.start()
        errorcode = self.basic_timestep()
        errorcode = self.accept_timestep(timer, errorcode)
        return errorcode

    def linear_timestep(self):
        return self.timestep()


class AdaptiveTimeStepper(ConstantTimeStepper):
    def __init__(self, problem, dt=1, triggers=[], max_step=3600, min_step=1, t_max=None, tol=1e-2,
                 initialize=True, **kwargs):
        super().__init__(problem, dt, triggers, initialize, **kwargs)
        self.max_step = max_step
        self.min_step = min_step
        self.t_max = t_max
        self.tol = tol
        self.tau = 1
        self.nu = self._calc_nu(self.tau)

    def linear_timestep(self):
        return super().timestep()

    def timestep(self):
        # Decide which timestep to use
        problem = self.problem
        timer = Timer('Adaptive TS')
        timer.start()
        self.dt = max(min(self.dt * self.tau, self.max_step), self.min_step)
        if self.t_max is not None:
            if problem.time + self.dt > self.t_max or problem.time + self.dt + self.min_step > self.t_max:
                self.dt = self.t_max - problem.time
                self.min_step = self.dt
                self.max_step = self.dt
        problem.set_timestep(self.dt)
        errorcode = self.basic_timestep()
        if errorcode != 0:
            timer.stop()
            if self.dt == self.min_step:
                return errorcode
            else:
                self.tau = 0.5
                self.nu = self._calc_nu(self.tau)
                errorcode = self.timestep()
                return errorcode
        error = self.get_time_filter_error()
        self.tau = problem._DT.update_time_step(
            max(error), self.dt, tol=self.tol, max_step=self.max_step, min_step=self.min_step)
        self.nu = self._calc_nu(self.tau)
        if self.tau < 1 - 3e-16:
            # This means the result is not accurate, need to recompute
            timer.stop()
            errorcode = self.timestep()
            return errorcode
        else:  # self.tau >= 1
            # This means the result is acepted, advance
            errorcode = self.accept_timestep(timer, errorcode)
            return errorcode

    @timed('TF Error')
    def get_time_filter_error(self):
        problem = self.problem
        error = []
        for index in range(len(problem.u_2.functions)):
            var_err = self.nu / 2 * (
                (2 / (1 + self.tau) * problem.u_2[index].x.array
                 - 2 * problem.u_1[index].x.array
                 + 2 * self.tau / (1 + self.tau) * problem.u_0[index].x.array)
            )
            error.append(np.linalg.norm(var_err))
        return error

    def _calc_nu(self, tau):
        return tau * (1 + tau) / (1 + 2 * tau)
