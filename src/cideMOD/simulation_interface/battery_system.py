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
import json
import os
import numpy as np
from pathlib import Path
from typing import Union

from cideMOD.helpers.logging import VerbosityLevel, _print
from cideMOD.helpers.plotview import PlotView
from cideMOD.helpers.miscellaneous import add_to_results_folder
from cideMOD.numerics.triggers import Trigger
from cideMOD.cell.parser import CellParser
from cideMOD.models import get_model_options, BaseModelOptions
from cideMOD.main import Problem
from cideMOD.simulation_interface.inputs import (CurrentInput, Cycle, Rest,
                                                 VoltageInput, execute_step)

current_dict = {'A': 1, 'mA': 1e-3}
voltage_dict = {'V': 1, 'mV': 1e-3}
time_dict = {'s': 1, 'min': 60, 'h': 3600, 'day': 24 * 3600}

DEFAULT_SIMULATION_OPTIONS = get_model_options()

DEFAULT_EVENT = {
    'type': 'Voltage',  # Voltage, Current, Ah, Wh
    'value': 2.8,  # Number
    'unit': 'V',
    'atol': 1e-3,  # Absolute tolerance
    'rtol': 1e-3,  # Relative tolerance
    'goto': 'End'  # Next or End or End Cycle or CV
}

DEFAULT_INPUT = {
    'name': 'Discharge',
    'type': 'Current',  # 'Current' or 'Voltage' or 'CC' or 'CV' or Rest
    'value': '-1',  # Must be float, int or string
    'unit': 'C',  # One of 'A', 'V', 'mA', 'mV', C
    't_max': {'value': 60, 'unit': 'min'},
    'store_delay': 2,
    'events': [DEFAULT_EVENT]
}

DEFAULT_PROFILE = {
    'type': 'CurrentProfile',
    'source': 'examples/notebooks/UDDS.txt',
    'unit': 'A',
    'delimiter': ';',
    'var_column': 3,
    'events': [DEFAULT_EVENT]
}

DEFAULT_TEST_PLAN = {
    'initial_state': {
        'SOC': 1,
        'exterior_temperature': 298.15
    },
    'steps': [DEFAULT_INPUT]
}

# TODO: Adapt CSI to any root model, not only the electrochemical.


class CSI:
    """
    Interface class to deal with the Problem class at high level.

    Parameters
    ----------
        cell_data : Union[dict,str]
            dictionary or path to the json file with the cell parameters
        simulation_options : Union[dict, BaseModelOptions], optional
            dicionary with simulation options, normally provided from
            the get_model_options method.
        test_plan : Union[dict,str], optional
            The dictionary with the test plan or a path to a JSON file
            with the test plan. Defaults to None.
        data_path : str, optional
            path to the folder with the additional data specified in
            cell_data. Defaults to None.
    """

    def __init__(self, cell_data: Union[CellParser, dict, str],
                 simulation_options: Union[dict, BaseModelOptions] = None,
                 test_plan: Union[str, dict] = None, data_path: str = None):
        # Load simulation options
        if simulation_options is None:
            self.simulation_options = DEFAULT_SIMULATION_OPTIONS
        if isinstance(simulation_options, BaseModelOptions):
            self.simulation_options = simulation_options
        elif isinstance(simulation_options, dict):
            self.simulation_options = get_model_options(**simulation_options)
        else:
            raise TypeError("Argument simulation_options must be of type"
                            + f"Union[dict, BaseModelOptions], is {type(simulation_options)}")
        self._comm = simulation_options.comm

        # Load cell data
        self.cell = self._load_cell_data(cell_data, data_path)
        self.ref_capacity = self.cell.ref_capacity

        # Create the problem
        self._create_problem()

        # Init test plan
        self._read_test_plan(test_plan)

    def _create_problem(self):
        # Create the problem
        self.problem = Problem(self.cell, self.simulation_options)

    def _load_cell_data(self, cell_data: Union[CellParser, dict, str], data_path=None):
        if isinstance(cell_data, CellParser):
            return cell_data
        elif isinstance(cell_data, str):
            if not os.path.exists(cell_data):
                raise FileNotFoundError(f"Path to cell data '{cell_data}' does not exists")
            if data_path is None:
                data_path = Path(cell_data).parent
            cell_data = Path(cell_data).name
        else:
            assert data_path is not None, 'data_path must be provided'
        return CellParser(cell_data, data_path, self.simulation_options)

    def _read_test_plan(self, test_plan: Union[dict, str] = None):
        """
        Read the specified test_plan. For examples about how to build a
        test_plan, see the DEFAULT_TEST_PLAN variable

        Parameters
        ----------
        test_plan : Union[dict,str], optional
            The dictionary with the test plan or a path to a JSON file
            with the test plan. Defaults to None.
        """
        if test_plan is None:
            plan = DEFAULT_TEST_PLAN
        elif isinstance(test_plan, str):
            assert os.path.exists(test_plan), "Path to test plan doesn't exists"
            with open(test_plan, 'r') as f:
                plan = json.load(f)
        elif isinstance(test_plan, dict):
            plan = test_plan
        self.test_plan = plan

        # Process Test Plan
        self.initial_state = plan['initial_state']
        self.steps = []
        self._C_rate_steps_ = []
        self._C_rate_triggers_ = []
        for step in plan['steps']:
            step = self._parse_step(step)
            if isinstance(step, list):
                self.steps.extend(step)
            else:
                self.steps.append(step)

        if len(self.steps) <= 0:
            raise RuntimeError("The test plan is empty")

        if self.simulation_options.verbose >= VerbosityLevel.BASIC_PROBLEM_INFO:
            self.print_test_plan()
        if self.simulation_options.save_path is not None:
            add_to_results_folder(
                self.simulation_options.save_path, files=[self.test_plan], filenames=['test_plan'])

    def print_test_plan(self):
        # TODO: print current in terms of C_rate if given
        _print('Initial state:', comm=self._comm)
        _print(self.initial_state, comm=self._comm, print_dic_kwargs={'tab_ini': '\t'})
        _print('Steps:', comm=self._comm)
        for i, step in enumerate(self.steps):
            _print(f"{i} - ", str(step), comm=self._comm)

    def _parse_step(self, step):
        # Parse Input
        wrong_format_message = ("Wrong format in '{key}'. Here you have an example: \n"
                                + str(DEFAULT_INPUT))
        step_types = ['Voltage', 'Current', 'CC', 'CV', 'Rest',
                      'Profile', 'VoltageProfile', 'CurrentProfile', 'Cycle']
        if step['type'] not in step_types:
            raise ValueError(wrong_format_message.format(key='type'))
        if step['type'] == 'Rest':
            t_max = self._unit_conversor(
                step['t_max']['value'], step['t_max'].get('unit', 's'), time_dict)
            input_event = Rest(
                step['name'], t_max, step.get('store_delay', 10), step.get('initial_step', None),
                step.get('max_step', 100), step.get('min_step', 5), step.get('adaptive', True),
                step.get('time_adaptive_tol', 1e-2))
        elif step['type'] in ('Current', 'CC'):
            value = self._unit_conversor(step['value'], step.get('unit', 'A'), current_dict)
            t_max = self._unit_conversor(
                step['t_max']['value'], step['t_max'].get('unit', 's'), time_dict)
            input_event = CurrentInput(
                step['name'], value, t_max, step.get('store_delay', 10),
                step.get('initial_step', None), step.get('max_step', 100), step.get('min_step', 5),
                step.get('adaptive', True), step.get('time_adaptive_tol', 1e-2))

            if step.get('unit', 'A') == 'C':
                self._C_rate_steps_.append((input_event, step['value']))  # Save (step, C-rate)

        elif step['type'] in ('Voltage', 'CV'):
            value = self._unit_conversor(step['value'], step.get('unit', 'V'), voltage_dict)
            t_max = self._unit_conversor(
                step['t_max']['value'], step['t_max'].get('unit', 's'), time_dict)
            input_event = VoltageInput(
                step['name'], value, t_max, step.get('store_delay', 10),
                step.get('initial_step', None), step.get('max_step', 100), step.get('min_step', 5),
                step.get('adaptive', True), step.get('time_adaptive_tol', 1e-2))
        elif step['type'] in ('Profile', 'CurrentProfile', 'VoltageProfile'):
            fin = step['source']
            delimiter = step.get('delimiter', ';')
            optimize = step.get('optimize', True)
            time_column = step.get('time_column', 0)
            var_column = step.get('var_column', 1)
            unit = step.get('unit', 'A')
            delay = step.get('store_delay', 10)
            skip_header = step.get('skip_header', True)
            if step['type'] == 'CurrentProfile':
                profile_type = 'current'
                unit = step.get('unit', 'A')
            elif step['type'] == 'VoltageProfile':
                profile_type = 'voltage'
                unit = step.get('unit', 'V')
            else:
                profile_type = step.get('profile_type', 'current')
                unit = step.get('unit', 'A')
                if profile_type not in ('voltage', 'current'):
                    raise ValueError("Invalid profile type")
            time, var = self._read_profile(fin, profile_type=profile_type, time_index=time_column,
                                           var_index=var_column, optimize=optimize,
                                           delimiter=delimiter, skip_header=skip_header)
            if time[0] != 0:
                time = time - time[0]
            ProfileInput = CurrentInput if profile_type == 'current' else VoltageInput
            profile_dict = current_dict if profile_type == 'current' else voltage_dict
            input_event = []
            for i, v in enumerate(var[:-1]):
                profile_step = ProfileInput(
                    f"{step['name']}_{i}", self._unit_conversor(v, unit, profile_dict),
                    time[i + 1] - time[i], store_delay=delay,
                    max_step=step.get('max_step', 100), min_step=step.get('min_step', 5),
                    adaptive=step.get('adaptive', True),
                    time_adaptive_tol=step.get('time_adaptive_tol', 1e-2),
                )

                if profile_type == 'current' and unit == 'C':
                    self._C_rate_steps_.append((profile_step, step['value']))

                input_event.append(profile_step)

        elif step['type'] in ('Cycle', ):
            assert all([key in step for key in ('count', 'steps')])
            input_event = Cycle(step.get('name'), int(step['count']))
            for substep in step.get('steps', []):
                input_event.add_step(self._parse_step(substep))

        # Parse Triggers/Events
        available_trigger_actions = ['NEXT', 'CV', 'END', 'END CYCLE']
        for event in step.get('events', []):
            for key in ['type']:
                if key not in event:
                    raise ValueError(
                        f"Wrong format in event. Here you have an example: \n {DEFAULT_EVENT}")
            if event['type'] == 'Voltage':
                assert all([key in event for key in ['value', 'unit']])
                variable = 'v'
                value = self._unit_conversor(event['value'], event['unit'], voltage_dict)
            elif event['type'] == 'Current':
                variable = 'i'
                value = self._unit_conversor(event['value'], event['unit'], current_dict)
            action = event.get('goto', 'Next')
            if action.upper() not in available_trigger_actions:
                raise ValueError(f"Unrecognized trigger action '{action}'. Available options: '"
                                 + "' '".join(available_trigger_actions) + "'")

            trigger = Trigger(value, variable, event['atol'], event['rtol'], action=action)
            if isinstance(input_event, list):
                for profile_event in input_event:
                    profile_event.add_trigger(trigger)
            else:
                input_event.add_trigger(trigger)

            if event['type'] == 'Current' and event['unit'] == 'C':
                self._C_rate_triggers_.append((trigger, event['value']))

        return input_event

    def _unit_conversor(self, value, unit, units_dict={}):
        if not units_dict:
            units_dict = {**current_dict, **voltage_dict, **time_dict}
        if unit == 'C':
            units_dict['C'] = self.ref_capacity
        assert unit in units_dict, 'Unit not found'
        if isinstance(value, str):
            return '(' + value + ')*' + str(units_dict[unit])
        elif value is None:
            return value
        else:
            return float(value) * units_dict[unit]

    def setup(self):
        """
        Initialize the initial state and set up the `cideMOD.Problem` object.
        For further information type `help(Problem.setup)`
        """
        self._set_initial_state()
        self.problem.setup()

    def run_test_plan(self):
        """
        Do the simulation specified in the test plan with the loaded
        cell

        Returns
        -------
        int or Exception:
            Status of the simulation, 0 means no problem, otherwise an
            Exception with information on the issues is returned.
        """
        if not self.problem._ready:
            self.setup()
        # Do necessary steps
        _save_on_exit = self.simulation_options.save_on_exit
        self.simulation_options.save_on_exit = False
        for step in self.steps:
            status = execute_step(step, self.problem)
            if status != 0:
                break
        if _save_on_exit:
            self.simulation_options.save_on_exit = True
            self.problem._WH.write_globals(self.simulation_options.clean_on_exit,
                                           individual_txt=self.simulation_options.globals_txts)
        return status

    def _read_profile(self, profile: Union[np.ndarray, list, str], profile_type='current',
                      time_0=0, time_index=0, var_index=1, optimize=True, skip_header=True,
                      delimiter=';'):
        # Reads series of input voltage or current as a function of time.
        # TODO: check if the profile is valid
        # FIXME: it skips some of the last steps, probably related to clean profile
        # if profile_type not in ['current', 'voltage']:
        #     raise ValueError("Unrecognized profile type, options are 'current' or 'voltage'")
        if isinstance(profile, str):
            if not os.path.exists(profile):
                raise FileNotFoundError(f"'{profile}' is not a valid path")
            data = np.genfromtxt(profile, delimiter=delimiter, skip_header=skip_header)

        if isinstance(profile, list):
            data = np.array(profile)

        time = data[:, time_index]
        var = data[:, var_index]

        # if time_0 == 0 and time[0] != 0:
        #     time = np.insert(time, 0, 0)
        # elif time_0 != 0:
        #     # TODO: Make it possible to start at a time > 0
        #     pass

        if not optimize:
            pass
        elif profile_type == 'current':
            capacity_threshold = self.ref_capacity / 100
            max_usage = max(abs(var[1:] - var[:-1]).max(), abs(var).max())
            threshold = max(capacity_threshold, max_usage / 1000)
            time, var = self._clean_profile(time, var, threshold)
        elif profile_type == 'voltage':
            profile_type
            # Group similar data to get lower count of timesteps
            capacity_threshold = self.ref_capacity / 100
            max_usage = max(abs(var[1:] - var[:-1]).max(), abs(var).max())
            threshold = max(capacity_threshold, max_usage / 1000)
            time, var = self._clean_profile(time, var, threshold)
        else:
            raise ValueError("Unrecognized profile type, options are 'current' or 'voltage'")

        # Convert to delta time
        # delta_time = np.diff(time)
        # var = var[-len(delta_time):]
        # clean_data = (delta_time, var)
        clean_data = (time, var)

        return clean_data

    def _clean_profile(self, time: np.ndarray, var: np.ndarray, threshold: float):
        """
        Clean the time series grouping similar values together up to a
        threshold in order to lower the count of timesteps

        Parameters
        ----------
        time : np.ndarray
            Time measurements array
        var : np.ndarray
            Variable measured at a time (current or voltage)
        threshold : float
            Threshold for grouping values
        """
        if time.shape != var.shape or time.size == 1:
            raise RuntimeError("Invalid profile input.")
        precision = int(-np.floor(np.log10(threshold)))
        clean_var = []
        clean_time = []
        temp_index = 0
        for index, v in enumerate(var[1:]):
            temp_v = var[temp_index:index + 1].mean()
            if abs(temp_v - v) > threshold or index == len(var[1:]) - 1:
                # Check if there has been a jump
                if index - temp_index > 1 and abs(var[index - 1] - v) > threshold:
                    # The jump will apply in the next step
                    clean_var.append(round(var[temp_index:index - 1].mean(), precision))
                else:
                    clean_var.append(round(temp_v, precision))
                clean_time.append(time[temp_index])
                temp_index = index + 1
        clean_time.append(time[-1])
        return clean_time, clean_var

    def _set_initial_state(self):
        SoC = self.initial_state.get('SOC')
        T = self.initial_state.get('exterior_temperature')
        self.problem.set_cell_state(SoC=SoC, T_ini=T, T_ext=T)

    def plot_global_results(self, results_path=None):
        """
        Generates an interactive plot with all the global variables.

        Parameters
        ----------
        results_path : StrPath, optional
            Path to the results folder, if None, the save_path of the
            problem object will be used. Defaults to None.
        """
        PlotView(self.problem, results_path)

    def reset(self, new_parameters: dict = None, save_path: str = None, save_config=True):
        """
        This method resets CSI in order to be ready for running another
        simulation with the same initial conditions, and maybe using
        different parameters.

        Parameters
        ----------
        new_parameters: Dict[str, float], optional
            Dictionary containing the cell parameters to be updated.
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
        self.problem.reset(
            new_parameters=new_parameters, save_path=save_path, save_config=save_config)

        # Update test plan if needed
        self.update_test_plan()

    def update_test_plan(self, new_test_plan=None):
        """
        This method updates the test plan.

        Parameters
        ----------
        new_test_plan: dict, optional
            New test plan to be simulated. Default to None.

        Notes
        -----
        If no new test plan is provided, this method checks if the
        reference capacity has changed and if so, updates the current
        steps and triggers based on C-rate input.
        """
        if self.simulation_options.verbose >= VerbosityLevel.BASIC_PROBLEM_INFO:
            _print("Updating test plan - ...", end='\r', comm=self._comm)

        has_C_rate_changed = not np.isclose(self.ref_capacity, self.cell.ref_capacity)
        if has_C_rate_changed:
            # Update the reference capacity
            self.ref_capacity = self.cell.ref_capacity

        if new_test_plan is not None:
            # Set the new test plan
            self._read_test_plan(test_plan=new_test_plan)
        else:
            # Reset triggers
            for step in self.steps:
                for trigger in step.triggers:
                    trigger.reset()

            if has_C_rate_changed:
                # Update the current steps that depends on the C-rate
                if self._C_rate_steps_:
                    for step, C_rate_value in self._C_rate_steps_:
                        step.i_app = C_rate_value * self.ref_capacity

                # Update the current triggers that depends on the C-rate
                if self._C_rate_triggers_:
                    for trigger, C_rate_value in self._C_rate_triggers_:
                        trigger.trigger_value = C_rate_value * self.ref_capacity

        if self.simulation_options.verbose >= VerbosityLevel.BASIC_PROBLEM_INFO:
            _print("Updating test plan - Done", comm=self._comm)
