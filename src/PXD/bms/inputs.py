from PXD.bms.triggers import SolverCrashed, Trigger, TriggerDetected
from PXD.helpers.miscellaneous import format_time

def execute_step(step, problem):
    status = step.execute(problem)
    if isinstance( status, TriggerDetected ):
        if status.action() == 'Next':
            return 0
        elif status.action() == 'CV':
            trig = Trigger(value=problem.Q/20,variable='i',atol=problem.Q/200)
            inpt = VoltageInput(name='CV',v_app=status.trigger.trigger_value ,t_max=step.t_max,store_delay=step.store_delay)
            inpt.add_trigger(trig)
            status = execute_step(inpt, problem)
            return status
        elif status.action() in ('End','End Cycle'):
            return status
        else:
            print(f'Dont know what means {status.action()}')
    return status

class Input:
    def __init__(self):
        self.i_app = None
        self.v_app = None

        self.triggers = []
        self.t_max = None

        self.max_step = None
        self.min_step = None

        self.store_delay = None
        self.adaptive = None

    def execute(self, problem):
        max_t = problem.time + self.t_max
        print(self)
        status = problem.solve_ie(i_app=self.i_app, v_app=self.v_app, t_f=max_t, max_step=self.max_step, min_step=self.min_step, 
                                  triggers=self.triggers, store_delay=self.store_delay, adaptive=self.adaptive)
        # status can be 0, TriggerDetected or SolverCrashed
        #  - 0 means final time reached
        #  - TriggerDetected has the trigger attribute
        #  - SolverCrashed means it didn't run well 
        return status

    def add_trigger(self, new_trigger:Trigger):
        # TODO: Add a check to ensure there are no overlapping triggers 
        self.triggers.append(new_trigger)

    def restrictions(self, problem):
        pass

class Cycle:
    def __init__(self, name:str, count:int):
        self.name = name
        self.count = count
        self.steps = []
        self.triggers = []

    def add_step(self, step:Input):
        self.steps.append(step)
    
    def add_trigger(self, trigger:Trigger):
        self.triggers.append(trigger)
        for step in self.steps:
            step.add_trigger(trigger)

    def execute(self, problem):
        for i in range(self.count):
            print(f"-- Cycle '{self.name}', iteration number {i} --")
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
    def __init__(self, name, i_app, t_max, store_delay=10, max_step=3600, min_step=5, adaptive=True):
        super().__init__()
        self.i_app = i_app
        self._i_app = i_app
        self.t_max = t_max
        self.name = 'CC_{}'.format(name)
        self.store_delay = store_delay
        self.max_step = max_step
        self.min_step = min_step
        self.adaptive = adaptive

    def execute(self, problem):
        if self._i_app is None:
            self.i_app = problem.get_current()
        return super().execute(problem)

    def __repr__(self):
        return f'CurrentInput(name={self.name}, i_app={self.i_app}, t_max={self.t_max}, triggers={self.triggers})'

    def __str__(self):
        return f'{self.name}: Apply {self.i_app} A during {format_time(self.t_max, 0)} until {self.triggers}'

class VoltageInput(Input):
    def __init__(self, name, v_app, t_max, store_delay=10, max_step=3600, min_step=5, adaptive=True):
        super().__init__()
        self.v_app = v_app
        self._v_app = v_app
        self.t_max = t_max
        self.name = 'CV_{}'.format(name)
        self.store_delay = store_delay
        self.max_step = max_step
        self.min_step = min_step
        self.adaptive = adaptive

    def execute(self, problem):
        if self._v_app is None:
            self.v_app = problem.get_voltage()
        return super().execute(problem)

    def __repr__(self):
        return f'VoltageInput(name={self.name}, v_app={self.v_app}, t_max={self.t_max}, triggers={self.triggers})'

    def __str__(self):
        return f'{self.name}: Apply {self.v_app} V during {format_time(self.t_max, 0)} until {self.triggers}'

class Rest(Input):
    def __init__(self, name, t_max, store_delay=100, max_step=3600, min_step=5, adaptive=True):
        super().__init__()
        self.i_app = 0
        self.t_max = t_max
        self.name = '{}'.format(name)
        self.store_delay = store_delay
        self.max_step = max_step
        self.min_step = min_step
        self.adaptive = adaptive

    def __repr__(self):
        return f'Rest(name={self.name}, t_max={self.t_max})'

    def __str__(self):
        return f'{self.name}: Rest during {format_time(self.t_max, 0)}'

