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
from typing import List

from cideMOD.helpers.logging import VerbosityLevel, _print
from cideMOD.helpers.miscellaneous import format_time
from cideMOD.numerics.triggers import SolverCrashed, Trigger, TriggerDetected, TriggerSurpassed
from cideMOD.main import Problem


def execute_step(step, problem: Problem):
    status = step.execute(problem)
    if isinstance(status, (TriggerDetected, TriggerSurpassed)):
        if status.action().upper() == 'NEXT':
            return 0
        elif status.action().upper() == 'CV':
            capacity = problem.cell_parser.ref_capacity
            trig = Trigger(value=capacity / 20, variable='i', atol=capacity / 200)
            inpt = VoltageInput(name='CV', v_app=status.trigger.trigger_value,
                                min_step=step.min_step, t_max=step.t_max,
                                store_delay=step.store_delay)
            inpt.add_trigger(trig)
            status = execute_step(inpt, problem)
            return status
        elif status.action().upper() in ('END', 'END CYCLE'):
            return status
        else:
            available_actions = ['Next', 'CV', 'End', 'End Cycle']
            raise ValueError(f"Unrecognized trigger action '{status.action()}'. "
                             + "Available options: '" + "' '".join(available_actions) + "'")
    return status


class Input:
    def __init__(self):
        self.i_app = None
        self.v_app = None

        self.triggers: List[Trigger] = []
        self.t_max = None

        self.max_step = None
        self.min_step = None
        self.initial_step = None

        self.store_delay = None
        self.adaptive = None
        self.time_adaptive_tol = None

    def execute(self, problem: Problem):
        if problem.verbose >= VerbosityLevel.BASIC_PROGRESS_INFO:
            _print(str(self), comm=problem._comm)
        max_t = problem.time + self.t_max
        status = problem.solve(
            i_app=self.i_app, v_app=self.v_app, t_f=max_t, initial_step=self.initial_step,
            max_step=self.max_step, min_step=self.min_step, triggers=self.triggers,
            store_delay=self.store_delay, adaptive=self.adaptive,
            time_adaptive_tol=self.time_adaptive_tol)
        # status can be 0, TriggerDetected or SolverCrashed
        #  - 0 means final time reached
        #  - TriggerDetected has the trigger attribute
        #  - SolverCrashed means it didn't run well
        return status

    def add_trigger(self, new_trigger: Trigger):
        # TODO: Add a check to ensure there are no overlapping triggers
        self.triggers.append(new_trigger)

    def restrictions(self, problem: Problem):
        pass


class Cycle:
    def __init__(self, name: str, count: int):
        self.name = name
        self.count = count
        self.steps = []
        self.triggers = []

    def add_step(self, step: Input):
        self.steps.append(step)

    def add_trigger(self, trigger: Trigger):
        self.triggers.append(trigger)
        for step in self.steps:
            step.add_trigger(trigger)

    def execute(self, problem: Problem):
        for i in range(self.count):
            if problem.verbose >= VerbosityLevel.BASIC_PROGRESS_INFO:
                _print(f"-- Cycle '{self.name}', iteration number {i} --", comm=problem._comm)
            for step in self.steps:
                status = execute_step(step, problem)
                if isinstance(status, TriggerDetected):
                    if status.action() == 'End Cycle':
                        return 0
                elif isinstance(status, SolverCrashed):
                    return status

    def __str__(self) -> str:
        string = f"""Cycle '{self.name}' repeats {self.count} times:\n"""
        for i, step in enumerate(self.steps):
            string += f"\t {i} - {str(step)}\n"
        return string[:-1]


class CurrentInput(Input):
    def __init__(self, name, i_app, t_max, store_delay=10, initial_step=None,
                 max_step=3600, min_step=5, adaptive=True, time_adaptive_tol=1e-2):
        super().__init__()
        self.i_app = i_app
        self._i_app = i_app
        self.t_max = t_max
        self.name = f'CC_{name}'
        self.store_delay = store_delay
        self.initial_step = initial_step
        self.max_step = max_step
        self.min_step = min_step
        self.adaptive = adaptive
        self.time_adaptive_tol = time_adaptive_tol

    def execute(self, problem: Problem):
        if self._i_app is None:
            self.i_app = problem._WH.get_global_variable_value('current')
        return super().execute(problem)

    def __repr__(self):
        return (f"CurrentInput(name={self.name}, i_app={self.i_app}, "
                + f"t_max={self.t_max}, triggers={self.triggers})")

    def __str__(self):
        return (f"{self.name}: Apply {self.i_app} A during "
                + f"{format_time(self.t_max, 0)} until {self.triggers}")


class VoltageInput(Input):
    def __init__(self, name, v_app, t_max, store_delay=10, initial_step=None,
                 max_step=3600, min_step=5, adaptive=True, time_adaptive_tol=1e-2):
        super().__init__()
        self.v_app = v_app
        self._v_app = v_app
        self.t_max = t_max
        self.name = f'CV_{name}'
        self.store_delay = store_delay
        self.initial_step = initial_step
        self.max_step = max_step
        self.min_step = min_step
        self.adaptive = adaptive
        self.time_adaptive_tol = time_adaptive_tol

    def execute(self, problem: Problem):
        if self._v_app is None:
            self.v_app = problem._WH.get_global_variable_value('voltage')
        return super().execute(problem)

    def __repr__(self):
        return (f"VoltageInput(name={self.name}, v_app={self.v_app}, "
                + f"t_max={self.t_max}, triggers={self.triggers})")

    def __str__(self):
        return (f"{self.name}: Apply {self.v_app} V during "
                + f"{format_time(self.t_max, 0)} until {self.triggers}")


class Rest(Input):
    def __init__(self, name, t_max, store_delay=100, initial_step=None,
                 max_step=3600, min_step=5, adaptive=True, time_adaptive_tol=1e-2):
        super().__init__()
        self.i_app = 0
        self.t_max = t_max
        self.name = str(name)
        self.store_delay = store_delay
        self.initial_step = initial_step
        self.max_step = max_step
        self.min_step = min_step
        self.adaptive = adaptive
        self.time_adaptive_tol = time_adaptive_tol

    def __repr__(self):
        return f"Rest(name={self.name}, t_max={self.t_max})"

    def __str__(self):
        return f"{self.name}: Rest during {format_time(self.t_max, 0)}"
