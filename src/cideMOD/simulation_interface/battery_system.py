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
import json
import os
from pathlib import Path
from typing import IO, Union

import numpy as np

from cideMOD.simulation_interface.inputs import CurrentInput, Cycle, Rest, VoltageInput, execute_step
from cideMOD.simulation_interface.triggers import Trigger, TriggerDetected, TriggerSurpassed
from cideMOD.helpers.config_parser import CellParser
from cideMOD.helpers.error_check import ErrorCheck
from cideMOD.helpers.miscellaneous import init_results_folder
from cideMOD.models.model_options import ModelOptions
from cideMOD.pxD import NDProblem, Problem

current_dict = {'A':1, 'mA':1e-3}
voltage_dict = {'V':1, 'mV':1e-3}
time_dict = {'s':1, 'min':60, 'h':3600, 'day':24*3600}

DEFAULT_SIMULATION_OPTIONS = ModelOptions()

DEFAULT_EVENT = {
    'type': 'Voltage', # Voltage, Current, Ah, Wh
    'value': 2.8, # Number 
    'unit': 'V', # 
    'atol': 1e-3, # Absolute tolerance
    'rtol': 1e-3, # Relative tolerance
    'goto': 'End' # Next or End or End Cycle or CV
}

DEFAULT_INPUT = {
    'name': 'Discharge',
    'type': 'Current', # 'Current' or 'Voltage' or 'CC' or 'CV' or Rest
    'value': '-1', # Must be float, int or string
    'unit': 'C', # One of 'A', 'V', 'mA', 'mV', C
    't_max': {'value': 60, 'unit': 'min'},
    'store_delay': 2,
    'events': [DEFAULT_EVENT]
}

DEFAULT_PROFILE = {
    'type': 'Profile',
    'source': 'data_DEFACTO/test-profile.csv',
    'unit': 'mA',
    'delimiter': ';',
    'current_column': 3,
    'events': [DEFAULT_EVENT]
}

DEFAULT_TEST_PLAN = {
    'initial_state': {
        'SOC': 1,
        'exterior_temperature': 298.15
    },
    'steps': [DEFAULT_INPUT]
}

class CSI:
    def __init__(self, cell_data:Union[dict,str], simulation_type:dict={}, data_path:str=None, name:str=None, overwrite=False, save_path=None):
        # Load simulation options
        self.simulation_options = DEFAULT_SIMULATION_OPTIONS
        if isinstance(simulation_type,ModelOptions):
            self.simulation_options = simulation_type
        elif isinstance(simulation_type, dict):
            for key in simulation_type:
                assert hasattr(self.simulation_options, key) , "Option '{}' not recognized".format(key)
                setattr(self.simulation_options,key,simulation_type[key])
        else:
            raise Exception(f"Argument simulation_type must be of type Union[dict,ModelOptions], is {type(simulation_type)}")
        # Load cell data
        self.cell = self.load_cell_data(cell_data, data_path)
        self.reference_capacity = self.cell.capacity

        if name is None and save_path is None:
            self.save_path = None
        else:
            self.save_path = init_results_folder(os.path.join(save_path or '', name or ''), overwrite=overwrite, copy_files=[cell_data, self.simulation_options])

        # Create the problem
        self.create_problem()

        # Create the solver

        # Init test plan
        self.initial_state = None
        self.test_plan = None

    def create_problem(self):
        # Create the problem
        if self.simulation_options.mode == 'P2D':
            self.problem = Problem(self.cell, self.simulation_options, save_path = self.save_path)
        else:
            self.problem = NDProblem(self.cell, self.simulation_options, save_path = self.save_path) 
        
    def load_cell_data(self, cell_data:Union[dict,str], data_path=None):
        if isinstance(cell_data,str):
            assert os.path.exists(cell_data), "Path to cell data '{}' doesn't exists".format(cell_data)
            if data_path is None:
                data_path = Path(cell_data).parent
            cell_data = Path(cell_data).name
        return CellParser(cell_data, data_path)

    def read_test_plan(self, test_plan:Union[dict,str]=None):
        if test_plan is None:
            plan = DEFAULT_TEST_PLAN
        if isinstance(test_plan, str):
            assert os.path.exists(test_plan), "Path to test plan doesn't exists"
            with open(test_plan, 'r') as f:
                plan = json.load(f)
        if isinstance(test_plan, dict):
            plan = test_plan
        self.test_plan = plan
        # Process Test Plan
        self.initial_state = plan['initial_state']
        
        self.steps = []
        for step in plan['steps']:
            step = self._parse_step(step)
            if isinstance(step, list):
                self.steps = self.steps + step
            else:
                self.steps.append(step)
        self.print_test_plan()
        if self.save_path:
            with open(os.path.join(self.problem.save_path,'test_plan.json'),'w') as fout:
                json.dump(self.test_plan,fout,indent=4,sort_keys=True)

    def print_test_plan(self):
        print('Initial state:')
        print(self.initial_state)
        print('Steps:')
        for i, step in enumerate(self.steps):
            print('{} - '.format(i), str(step))

    def _parse_step(self, step):
        # Parse Input
        wrong_format_message = "Wrong format in '{key}'. Here you have an example: \n"+ str(DEFAULT_INPUT)
        assert step['type'] in ['Voltage', 'Current', 'CC', 'CV', 'Rest', 'Profile', 'Cycle'], wrong_format_message.format(key=type)
        if step['type'] == 'Rest':
            t_max = self._unit_conversor(step['t_max']['value'], step['t_max'].get('unit','s'), time_dict)
            input_event = Rest(step['name'], t_max, step.get('store_delay',10), 
                step.get('max_step', 100), step.get('min_step', 5), step.get('adaptive', True))
        elif step['type'] in ('Current', 'CC'):
            value = self._unit_conversor(step['value'], step.get('unit','A'), current_dict)
            t_max = self._unit_conversor(step['t_max']['value'], step['t_max'].get('unit','s'), time_dict)
            input_event = CurrentInput(step['name'], value , t_max, step.get('store_delay',10), 
                step.get('max_step', 100), step.get('min_step', 5), step.get('adaptive', True))
        elif step['type'] in ('Voltage', 'CV'):
            value = self._unit_conversor(step['value'], step.get('unit','V'), voltage_dict)
            t_max = self._unit_conversor(step['t_max']['value'], step['t_max'].get('unit','s'), time_dict)
            input_event = VoltageInput(step['name'], value , t_max, step.get('store_delay',10), 
                step.get('max_step', 100), step.get('min_step', 5), step.get('adaptive', True))
        elif step['type'] in ('Profile',):
            fin = step['source']
            delimiter = step['delimiter']
            time_column = step.get('time_column', 0)
            current_column = step.get('current_column', 1)
            unit = step.get('unit', 'A')
            delay = step.get('store_delay', 10)
            time, current = self.read_current_profile(fin, time_column, current_column)
            if time[0] != 0:
                time = time-time[0]
            input_event = []
            for i, cur in enumerate(current):
                profile_step = CurrentInput(i,self._unit_conversor(cur, unit, current_dict),time[i+1],delay, 
                    step.get('max_step', 100), step.get('min_step', 5), step.get('adaptive', True))
                input_event.append(profile_step)
        elif step['type'] in ('Cycle', ):
            assert all([key in step for key in ('count', 'steps')])
            input_event = Cycle(step.get('name'), int(step['count']))
            for substep in step.get('steps',[]):
                input_event.add_step(self._parse_step(substep))

        # Parse Triggers/Events
        for event in step.get('events',[]):
            for key in ['type']:
                assert key in event, 'Wrong format in event. Here you have an example: \n {default}'.format(default=DEFAULT_EVENT)
            if event['type'] == 'Voltage':
                assert all([key in event for key in ['value','unit']])
                variable = 'v'
                value = self._unit_conversor(event['value'], event['unit'], voltage_dict)
            elif event['type'] == 'Current':
                variable = 'i'
                value = self._unit_conversor(event['value'], event['unit'], current_dict)
            action = event.get('goto','Next')
            trigger = Trigger(value,variable, event['atol'], event['rtol'], action=action)
            if isinstance(input_event, list):
                for event in input_event:
                    event.add_trigger(trigger)
            else:
                input_event.add_trigger(trigger)
        return input_event

    def _unit_conversor(self, value, unit, units_dict={}):
        if not units_dict:
            units_dict = {**current_dict, **voltage_dict, **time_dict}
        if unit is 'C':
            units_dict['C'] = self.reference_capacity
        assert unit in units_dict, 'Unit not found'
        if isinstance(value, str):
            return '(' + value + ')*' + str(units_dict[unit])
        elif value is None:
            return value
        else:
            return float(value)*units_dict[unit]

    def run_test_plan(self):
        # assert self.test_plan is not None, "Need to load a test plan"
        # setup problem
        self.set_initial_state()
        self.problem.setup()
        # Do necessary steps
        for step in self.steps:
            status = execute_step(step, self.problem)
            if status != 0:
                return status
        return 0

    def read_current_profile(self, profile:Union[np.ndarray,list,str], time_index = 0, current_index = 1, optimize =True ):
        # Reads series of input voltage or current as a function of time. 
        if isinstance(profile,str):
            assert os.path.exists(profile), "'{}' is not a valid path"
            data=np.genfromtxt(profile, delimiter=';',skip_header=True)
        if isinstance(profile, list):
            data = np.array(profile)
        time = data[:,time_index]
        current = data[:,current_index]

        if optimize:
            # Group similar data to get lower count of timesteps
            capacity_threshold = self.reference_capacity/100
            max_usage = max(abs(current[1:] - current[:-1]).max(), abs(current).max())
            threshold = max( capacity_threshold , max_usage/1000 )
            clean_data = self._clean_profile(time, current, threshold)
        else:
            clean_data = (time, current)
        return clean_data
        
            
        
    def _clean_profile(self, time:np.ndarray, data:np.ndarray, threshold:float):
        """
        Clean the time series grouping similar values together up to a threshold in order to lower the count of timesteps

        Parameters
        ----------
        time : np.ndarray
            Time measurements array
        data : np.ndarray
            Data measured at a time
        threshold : float
            Threshold of 

        Returns
        -------
        [type]
            [description]
        """
        assert time.shape == data.shape
        assert len(time.shape) == 1
        clean_data = []
        clean_time = []
        temp_index = 0
        for index, I in enumerate(data[1:]):
            temp_I = data[temp_index:index+1].mean()
            if abs(temp_I-I)>threshold:
                clean_data.append(round( temp_I, int(-np.floor(np.log10(threshold))) ))
                clean_time.append(time[temp_index])
                temp_index = index+1
        clean_time.append(time[-1])
        return clean_time, clean_data            

    def read_voltage_profile(self, *args, **kwargs):
        pass

    def set_initial_state(self):
        soc = self.initial_state.get('SOC')
        T = self.initial_state.get('exterior_temperature')
        self.problem.set_cell_state(soc, T, T)
