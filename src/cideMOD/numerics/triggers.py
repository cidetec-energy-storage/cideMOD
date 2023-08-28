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
import functools


class Trigger:
    """
    An event that fires a change in the cell input. It is evaluated
    during cell cycling at each timestep.

    Parameters
    ----------
    value: Union[float, int]
        Value that fires the trigger.
    variable: str
        Variable of the state of the cell to compare. To know the
        available ones type `Trigger.print_available_variables`.
    atol: float, optional
        Absolute tolerance for the trigger to fire. Default to 1e-3.
    rtol: float, optional
        Relative tolerance for the trigger to fire. Default to 1e-3.
    action: str, optional
        Event to trigger. Default to 'Next'.
    mode: str, optional
        Can be 'min', 'max' or None to specify total triggers or
        relative to current state. Default to None.
    """

    variables: dict = {'time': {'label': 't', 'units': 's', 'abs': False, 'atol': 1e-6}}
    available_variables: list = ['time', 't']

    def __init__(self, value, variable: str, atol=None, rtol=None,
                 action="Next", mode=None):
        self.trigger_value = value
        if variable not in self.available_variables:
            grouped_vars = zip(*[self.available_variables[i::2] for i in range(2)])
            raise ValueError(
                f"Unrecognized trigger variable '{variable}'. Available options: "
                + ", ".join([f"'{var}'('{alias}')" for var, alias in grouped_vars]))
        self.variable = self._get_variable(variable)
        self._var_info = self.variables[self.variable]
        self.atol = abs(atol) if atol else self._var_info.get('atol', 1e-6)
        self.rtol = abs(rtol) if rtol else self._var_info.get('rtol', 1e-3)

        self.old_value = None
        self.action = action
        modes = [None, 'min', 'max']
        if mode not in modes:
            raise ValueError(f"Unrecognized trigger mode '{mode}'. Available options: '"
                             + "' '".join(modes) + "'")
        self.mode = mode

    def __repr__(self):
        return (f"{self.variable.capitalize()} Trigger at {self.trigger_value:.2g} "
                + self._var_info['units'])

    def __str__(self):
        symbol_mod = [">", "<", ""][["max", "min", None].index(self.mode)]
        return (f"{self.variable.capitalize()} is {symbol_mod}{self.trigger_value:.2g} "
                + self._var_info['units'])

    @classmethod
    def register_variable(cls, name, label, units, need_abs=False, atol=1e-6):
        """
        Register a new trigger variable.

        Parameters
        ----------
        name: str
            Name of the variable
        label: str
            Label of the variable. It is used as an alias.
        units: str
            Units of the trigger variable.
        need_abs: bool
            Whether or not to check the absolute value of the trigger
            variable. Default to False.
        """
        if name in cls.available_variables:
            raise ValueError(f"Trigger name '{name}' already registered!")
        elif label in cls.available_variables:
            raise ValueError(f"Trigger label '{label}' already registered!")

        cls.variables[name] = {'label': label, 'units': units, 'abs': need_abs, 'atol': atol}
        cls.available_variables.extend([name, label])

    @classmethod
    def print_available_variables(cls):
        # TODO: Consider including model_options as input to print only the available variables of
        #       the active models.
        grouped_vars = zip(*[cls.available_variables[i::2] for i in range(2)])
        print(f"Available trigger variables: "
              + ", ".join([f"'{var}'('{alias}')" for var, alias in grouped_vars]))

    @classmethod
    def _get_variable(cls, variable):
        if variable in cls.variables.keys():
            return variable
        else:
            return cls.available_variables[cls.available_variables.index(variable) - 1]

    def _measure(self, state_dict):
        value = state_dict[self.variable]
        return abs(value) if self._var_info['abs'] else value

    def _detect(self, value):
        comp = (self.trigger_value - value) / max(self.atol, self.rtol * abs(self.trigger_value))
        if self.mode == 'max':
            return comp < 0 and comp > -1
        elif self.mode == 'min':
            return comp > 0 and comp < 1
        else:
            return abs(comp) < 1

    def _check_surpass(self, value):
        comp = (self.trigger_value - value) / max(self.atol, self.rtol * abs(self.trigger_value))
        change = (value - self.trigger_value) * (self.old_value - self.trigger_value) < 0
        if self.mode == 'max':
            return comp < -1 or change
        elif self.mode == 'min':
            return comp > 1 or change
        else:
            return change

    def start_record(self, state):
        self.old_value = self._measure(state)

    def check(self, state):
        if self.old_value is None:
            self.start_record(state)
            if self._check_surpass(self.old_value):
                raise TriggerDetected(self)
        else:
            new_value = self._measure(state)
            if self._detect(new_value):
                # Trigger detected
                raise TriggerDetected(self)
            elif self._check_surpass(new_value):
                # Trigger surpassed
                raise TriggerSurpassed(
                    (self.trigger_value - self.old_value) / (new_value - self.old_value), self)
            else:
                # Trigger not reached
                self.old_value = new_value

    def reset(self):
        self.old_value = None


class TriggerDetected(Exception):
    def __init__(self, trigger):
        self.trigger = trigger

    def action(self):
        return self.trigger.action

    def __repr__(self) -> str:
        return self.trigger.__repr__().replace('Trigger', 'TriggerDetected')

    def __str__(self) -> str:
        return self.trigger.__str__()  # .replace('<', '').replace('>', '')


class TriggerSurpassed(Exception):
    def __init__(self, time_slope, trigger):
        self.time_slope = time_slope
        self.trigger = trigger

    def new_tstep(self, old_tstep):
        return self.time_slope * old_tstep

    def __repr__(self) -> str:
        return self.trigger.__repr__().replace('Trigger', 'TriggerSurpassed')

    def __str__(self) -> str:
        return functools.reduce(
            lambda string, old_new: string.replace(*old_new),
            [(" is ", " surpasses "), ('<', ''), ('>', '')],
            self.trigger.__str__())


class SolverCrashed(Exception):
    """
    Exception indicating that the solver has crashed due to
    non-convergence or NaN values
    """
    pass
