class Trigger:
    def __init__(self, value, variable:str, atol:float=1e-3, rtol:float=1e-3, action='Next'):
        self.trigger_value = value
        assert variable in ('v', 'i', 't')
        self.variable = variable
        if not isinstance(atol,float):
            atol = 1e-3
        if not isinstance(rtol,float):
            rtol = 1e-3
        self.atol = atol
        self.rtol = rtol

        self.old_value = None
        self.action = action

    def start_record(self, state):
        self.old_value = self._measure(state)

    def _measure(self, state_dict):
        value = state_dict[self.variable]
        return abs(value) if self.variable == 'i' else value

    def check(self, state):
        if self.old_value is None:
            self.start_record(state)
        else:
            new_value = self._measure(state)
            if abs(self.trigger_value - new_value)/max(self.atol, self.rtol * self.trigger_value) < 1:
                # Trigger detected
                raise TriggerDetected(self)
            elif (new_value-self.trigger_value)*(self.old_value-self.trigger_value)<0:
                # Trigger surpassed
                raise TriggerSurpassed(time_slope=(self.trigger_value-self.old_value)/(new_value-self.old_value))
            else:
                # Trigger not reached
                self.old_value = new_value

    def __repr__(self):
        text = {'v': ('Voltage', 'V'), 'i': ('Current', 'A')} 
        return f'{text[self.variable][0]} Trigger at {self.trigger_value:.2g} {text[self.variable][1]}'

    def __str__(self):
        text = {'v': ('Voltage', 'V'), 'i': ('Current', 'A')} 
        return f'{text[self.variable][0]} is {self.trigger_value:.2g} {text[self.variable][1]}'


class TriggerDetected(Exception):
    def __init__(self, trigger):
        self.trigger = trigger

    def action(self):
        return self.trigger.action

    def __str__(self) -> str:
        return self.trigger.__str__()

class TriggerSurpassed(Exception):
    def __init__(self, time_slope):
        self.time_slope = time_slope

    def new_tstep(self, old_tstep):
        return self.time_slope*old_tstep

class SolverCrashed(Exception):
    pass