#
# Copyright (c) 2021 CIDETEC Energy Storage.
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
class Trigger:
    """An event that fires a change in the cell input. It is evaluated during cell cycling at each timestep

    Args:
        value (_type_): Value that fires the trigger.
        variable (str): Variable of the state of the cell to compare. One of:

         - 'v': Cell Voltage 
         - 'i': Cell Current
         - 't': Simulation time
        
        atol (float, optional): Absolute tolerance for the trigger to fire. Defaults to 1e-3.
        rtol (float, optional): relative tolerance for the trigger to fire. Defaults to 1e-3.
        action (str, optional): Event to trigger. Defaults to 'Next'.
        mode (str, optional): Can be 'min', 'max' or None to specify total triggers or relative to current state. Defaults to None.
    """
    def __init__(
        self,
        value,
        variable: str,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        action="Next",
        mode=None,
    ):
        self.trigger_value = value
        assert variable in ("v", "i", "t")
        self.variable = variable
        if not isinstance(atol, float):
            atol = 1e-6
        if not isinstance(rtol, float):
            rtol = 1e-3
        self.atol = abs(atol)
        self.rtol = abs(rtol)

        self.old_value = None
        self.action = action
        assert mode in [None, 'min', 'max']
        self.mode = mode

    def start_record(self, state):
        self.old_value = self._measure(state)

    def _measure(self, state_dict):
        value = state_dict[self.variable]
        return abs(value) if self.variable == "i" else value

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
                    time_slope=(self.trigger_value - self.old_value)
                    / (new_value - self.old_value) * 1.02
                )
            else:
                # Trigger not reached
                self.old_value = new_value

    def __repr__(self):
        text = {"v": ("Voltage", "V"), "i": ("Current", "A")}
        return f"{text[self.variable][0]} Trigger at {self.trigger_value:.2g} {text[self.variable][1]}"

    def __str__(self):
        text = {"v": ("Voltage", "V"), "i": ("Current", "A")}
        symbol_mod = [">","<",""][["max","min",None].index(self.mode)]

        return f"{text[self.variable][0]} is {symbol_mod}{self.trigger_value:.2g} {text[self.variable][1]}"


class TriggerDetected(Exception):
    def __init__(self, trigger):
        self.trigger = trigger

    def action(self):
        return self.trigger.action

    def __str__(self) -> str:
        return self.trigger.__str__()

    def __repr__(self):
        text = {'v': ('Voltage', 'V'), 'i': ('Current', 'A'), 'q': ('Capacity', 'As'), 'dod': ('DoD', '')} 
        return f'{text[self.trigger.variable][0]} TriggerDetected at {self.trigger.trigger_value:.2g} {text[self.trigger.variable][1]}'



class TriggerSurpassed(Exception):
    def __init__(self, time_slope, trigger = None):
        self.time_slope = time_slope
        self.trigger = trigger

    def new_tstep(self, old_tstep):
        return self.time_slope * old_tstep

    def __repr__(self):
        text = {'v': ('Voltage', 'V'), 'i': ('Current', 'A'), 'q': ('Capacity', 'As'), 'dod': ('DoD', '')} 
        return f'{text[self.trigger.variable][0]} TriggerSurpassed at {self.trigger.trigger_value:.2g} {text[self.trigger.variable][1]}'

    def __str__(self) -> str:
        text = {'v': ('Voltage', 'V'), 'i': ('Current', 'A'), 'q': ('Capacity', 'As'), 'dod': ('DoD', '')}
        return f'{text[self.trigger.variable][0]} surpasses {self.trigger.trigger_value:.2g} {text[self.trigger.variable][1]}'


class SolverCrashed(Exception):
    """Exception indicating that the solver has crashed due to non-convergence or NaN values"""
    pass
