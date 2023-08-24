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
import dolfinx as dfx
from mpi4py import MPI
from dolfinx.common import timed

import os
import numpy as np
from itertools import chain
from sys import getsizeof

from cideMOD.numerics.fem_handler import interpolate, BlockFunction


class Warehouse:
    def __init__(self, save_path, problem, delay=1):
        self.save_path = save_path
        self._comm = problem._comm
        self.problem = problem
        self._models = problem._models
        self.delay = delay
        self._counter = 0
        self._prev_time = None

        # Prepare outputs information and add the time
        self._requested_outputs = {'globals': [], 'internals': []}
        self._outputs_info = {'globals': {}, 'internals': {}}

        self.add_global_variable_info(name='time', fnc=None, header="Time [s]",
                                      default=True, fmt="%2.2f")

        # Complete outputs information from the models
        self._models.get_outputs_info(self)

        # Initialize requested outputs
        for key in self._requested_outputs.keys():
            for out_name, out_info in self._outputs_info[key].items():
                if out_info['default']:
                    self._requested_outputs[key].append(out_name)

        self._global_vars = None
        self._storing_functions = None
        self._postprocessing_functions = None
        self._storing_times = []

    def add_global_variable(self, name):
        if name not in self._outputs_info['globals']:
            raise ValueError(f"Unrecognized global variable '{name}'. Available options: '"
                             + "' '".join(self._outputs_info['globals'].keys()) + "'")
        elif name not in self._requested_outputs['globals']:
            self._requested_outputs['globals'].append(name)

    def add_internal_variable(self, name):
        if name not in self._outputs_info['internals']:
            raise ValueError(f"Unrecognized internal variable '{name}'. Available options: '"
                             + "' '".join(self._outputs_info['internals'].keys()) + "'")
        elif name not in self._requested_outputs['internals']:
            self._requested_outputs['internals'].append(name)

    def add_global_variable_info(self, name: str, fnc, header=None, default=False, dtype='scalar',
                                 postprocessing_fnc=None, fmt="%1.8e") -> None:
        # TODO: Perform additional checks
        if name in self._outputs_info['globals']:
            raise ValueError(f"Global variable '{name}' already added")
        dtypes = ['scalar', 'list_of_scalar']
        if dtype not in dtypes:
            raise ValueError(f"Invalid dtype '{dtype}' for the global variable '{name}'. "
                             + "Available options '" + "' '".join(dtypes) + "'")

        if dtype == 'list_of_scalar' and isinstance(header, str) and '{' not in header:
            header += '[{i}]'

        self._outputs_info['globals'][name] = {
            'fnc': fnc,
            'dtype': dtype,
            'header': header,
            'default': default,
            'postprocessing_fnc': postprocessing_fnc,
            'fmt': fmt
        }

    def add_internal_variable_info(self, name: str, subdomains: str, dtype: str = 'scalar',
                                   function_space: str = 'P1', default=False,
                                   postprocessing_fnc=None):
        # TODO: Perform additional checks
        if name in self._outputs_info['internals']:
            raise ValueError(f"Internal variable '{name}' already added")

        fs_options = ['P0', 'P1']
        if function_space not in fs_options:
            raise ValueError(f"Unrecognized function space '{function_space}'. "
                             + "Available options: '" + "' '".join(fs_options) + "'")

        self._outputs_info['internals'][name] = {
            'subdomains': subdomains,
            'dtype': dtype,
            'function_space_tag': function_space,
            'default': default,
            'postprocessing_fnc': postprocessing_fnc,
            'source': None,
            'length': None
        }

    def set_delay(self, delay):
        """
        This method set the delay between timestep where the internal
        variables are stored. If delay <= 0 they are not stored. If
        delay is 0, it is posible to force the storage.
        """
        self.delay = delay

    def set_storing_time(self, times, deactivate_delay=True):
        """
        This method specify the points in time where the internal
        variables are stored.
        """
        self._storing_times = np.array(times, dtype=float).reshape((len(times),))
        if deactivate_delay:
            self.delay = 0

    def setup(self):
        """This method prepare the outputs to be stored"""

        # Preprocess output information
        self._setup_outputs_info()

        # Compute storing functions
        if self.save_path is not None:
            self._compute_storing_functions()

        # Prepare global variable warehouse
        self._global_vars = {name: [] for name in self._requested_outputs['globals']}

        # Add the required postprocessing functions
        self._postprocessing_functions = {'globals': [], 'internals': []}
        for key in self._postprocessing_functions.keys():
            for out_info in self._outputs_info[key].values():
                fnc = out_info['postprocessing_fnc']
                if fnc is not None and fnc not in self._postprocessing_functions[key]:
                    # Additional check, as globals will be stored every step
                    if key == 'globals' or fnc not in self._postprocessing_functions['globals']:
                        self._postprocessing_functions[key].append(fnc)

    def setup_internal_variable(self, name, source, length=None):
        internals_info = self._outputs_info['internals']
        if name not in internals_info.keys():
            raise ValueError(f"Unrecognized internal variable '{name}'. Available options: '"
                             + f"' '".join(internals_info.keys()) + "'")
        elif internals_info[name]['source'] is not None:
            raise RuntimeError(f"Internal variable '{name}' source already set")
        if 'list' in internals_info[name]['dtype'] and length is None:
            raise ValueError(
                f"The length of the internal variable '{name}' must be specified")
        else:
            internals_info[name]['source'] = source
            internals_info[name]['length'] = length

    def _setup_outputs_info(self):
        problem = self.problem
        # Get function space and subdomain mappers
        for var_info in self._outputs_info['internals'].values():
            is_vector = 'vector' in var_info['dtype']
            fs_tag = var_info['function_space_tag']
            if fs_tag == 'P0':
                if is_vector:
                    var_info['function_space'] = problem.V_0_vec
                    var_info['subdomain_mapper'] = problem.P0_vec_map
                else:
                    var_info['function_space'] = problem.V_0
                    var_info['subdomain_mapper'] = problem.P0_map
            elif fs_tag == 'P1':
                if is_vector:
                    var_info['function_space'] = problem.V_vec
                    var_info['subdomain_mapper'] = problem.P1_vec_map
                else:
                    var_info['function_space'] = problem.V
                    var_info['subdomain_mapper'] = problem.P1_map
            else:
                raise ValueError(f"Unrecognized function space '{fs_tag}'")

        # Compute internal variable sources
        self._models.prepare_outputs(self, problem._vars, problem.cell,
                                     problem.mesher, problem._DA, problem)

    def _compute_storing_functions(self):
        self.file = self._create_storing_file('results')
        var_names, var_functions = [], []
        for var_name in self._requested_outputs['internals']:
            var_info = self._outputs_info['internals'][var_name]
            is_vector = 'vector' in var_info['dtype']
            is_list = 'list' in var_info['dtype']

            if not is_list:
                storing_fnc = self._create_storing_function(
                    var_name, is_vector=is_vector, function_space=var_info['function_space'])
            elif var_info['subdomains'] not in ['anode', 'a', 'cathode', 'c']:
                raise ValueError(f"Internal variable '{var_name}' cannot be a list")
            else:
                storing_fnc = []
                for k in range(var_info['length']):
                    fnc = self._create_storing_function(f'{var_name}_{k}', is_vector=is_vector,
                                                        function_space=var_info['function_space'])
                    storing_fnc.append(fnc)

            var_names.append(var_name)
            var_functions.append(storing_fnc)

        self._storing_functions = BlockFunction(var_names, var_functions)

    def _create_storing_file(self, filename, folder=''):
        filepath = os.path.join(self.save_path, folder, f'{filename}.xdmf')
        with dfx.io.XDMFFile(self._comm, filepath, 'w') as file:
            file.write_mesh(self.problem.mesher.mesh)
        return filepath

    def _create_storing_function(self, name, is_vector=False, function_space=None):
        if function_space in [None, 'P1']:
            function_space = self.problem.V if not is_vector else self.problem.V_vec
        elif function_space == 'P0':
            function_space = self.problem.V_0 if not is_vector else self.problem.V_vec_0
        return dfx.fem.Function(function_space, name=name)

    def store(self, time, force=False):
        self._postprocess('globals')
        self._store_globals(time)
        if self.save_path is not None and self._need_to_store(time, force=force):
            self._postprocess('internals')
            self._update_internals()
            self._store_internals(time)

    def _need_to_store(self, time, force=False):
        # Check if is near a storing time
        if self._storing_times:
            idx = np.abs(self._storing_times - time).argmin()
            if self._storing_times[idx] < time or np.isclose(self._storing_times[idx], time):
                nearest_time = self._storing_times[idx]
            else:
                nearest_time = self._storing_times[idx - 1]
            if nearest_time > self._prev_time:
                self._prev_time = time
                return True
            else:
                self._prev_time = time

        # Check if the storage should be delayed
        if self.delay >= 0:
            self._counter += 1
            if self._counter >= self.delay or force:
                self._counter = 0
                return True

        return False

    @timed('Post-processing')
    def _postprocess(self, tag):
        for fnc in self._postprocessing_functions[tag]:
            fnc()

    @timed('Store Globals')
    def _store_globals(self, time):
        for var_name, var_list in self._global_vars.items():
            if var_name == 'time':
                var_value = time
            else:
                var_value = self._outputs_info['globals'][var_name]['fnc']()
            var_list.append(var_value)

    @timed('Store Internals')
    def _store_internals(self, time):
        with dfx.io.XDMFFile(self._comm, self.file, 'a') as file:
            for storing_fnc in self._storing_functions.functions:
                if isinstance(storing_fnc, (list, tuple)):
                    for fnc in storing_fnc:
                        file.write_function(fnc, time)
                else:
                    file.write_function(storing_fnc, time)

    @timed('Update Internals')
    def _update_internals(self):
        # Update internals
        for var_name, storing_fnc in self._storing_functions.items():
            var_info = self._outputs_info['internals'][var_name]
            mapper = var_info['subdomain_mapper']
            source = var_info['source']
            if isinstance(storing_fnc, (list, tuple)):
                for am_idx, am_fnc in enumerate(storing_fnc):
                    if isinstance(source, dict):
                        am_source = {subdomain: ex[am_idx] for subdomain, ex in source.items()}
                    elif isinstance(source, list):
                        am_source = source[am_idx]
                    else:
                        raise TypeError(f"Unrecognized type of source '{type(source)}' "
                                        + f"of the variable '{var_name}'")
                    self._update_storing_function(am_source, am_fnc, mapper)
            else:
                self._update_storing_function(source, storing_fnc, mapper)

    def _update_storing_function(self, source, function: dfx.fem.Function, mapper):
        if isinstance(source, dict):
            return mapper.interpolate(source, function)
        else:
            return interpolate(source, function)

    def write_globals(self, clean=True, timings=False, individual_txt=True, sep='\t'):
        if self._comm.rank != 0 or self.save_path is None:
            if clean:
                self.clean()
            return

        time = np.array(self._global_vars['time'], ndmin=2).T
        data = []
        for var_name, var_list in self._global_vars.items():
            if not var_list:
                continue
            var_info = self._outputs_info['globals'][var_name]
            if var_info['dtype'] == 'list_of_scalar':
                var_info['length'] = len(var_list[0])
                var_array = np.array(var_list, copy=False, ndmin=2)
            else:
                var_array = np.array(var_list, copy=False, ndmin=2).T  # column vector

            data.append(var_array)

            if individual_txt and var_name != 'time':
                fname = os.path.join(self.save_path, f'{var_name}.txt')
                var_data = np.concatenate((time, var_array), axis=1)
                time_info = self._outputs_info['globals']['time']
                if var_info['dtype'] == 'list_of_scalar':
                    n = var_info['length']
                    fmt = sep.join([time_info['fmt']] + [var_info['fmt']] * n)
                    var_header = var_info['header']
                    if not isinstance(var_header, list):
                        var_header = [var_header.format(i=i) for i in range(n)]
                    headers = sep.join([time_info['header']] + var_header)
                else:
                    fmt = sep.join([time_info['fmt'], var_info['fmt']])
                    headers = sep.join(([time_info['header'], var_info['header']]))
                self._save_txt_file(fname, var_data, fmt, headers, clean)

        # Write condensated
        if data:
            data = np.concatenate(tuple(data), axis=1)
            self._write_compiled_output(data, clean=clean, sep=sep)

        # Reset globals container
        if clean:
            self.clean()

        # Write timings table
        # TODO: Write timings table if the DEBUG mode is active
        if timings:
            timing_types = [dfx.TimingType.wall, dfx.TimingType.user, dfx.TimingType.system]
            dfx.list_timings(self._comm, timing_types)

    def _save_txt_file(self, fname, data, fmt, headers, clean=True):
        if os.path.exists(fname):
            fmode = 'ab' if clean else 'wb'
            with open(fname, fmode) as f:
                np.savetxt(fname=f, X=data, fmt=fmt)
        else:
            np.savetxt(fname=fname, X=data, header=headers, fmt=fmt)

    def _write_compiled_output(self, data, clean=True, sep='\t'):
        fname = os.path.join(self.save_path, 'condensated.txt')
        headers, fmt = [], []
        for var_name in self._requested_outputs['globals']:
            if self._global_vars[var_name]:
                var_info = self._outputs_info['globals'][var_name]
                if var_info['dtype'] == 'list_of_scalar':
                    n = var_info['length']
                    var_header = var_info['header']
                    if not isinstance(var_header, list):
                        var_header = [var_header.format(i=i) for i in range(n)]
                    headers.extend(var_header)
                    fmt.extend([var_info['fmt']] * n)
                else:
                    headers.append(var_info['header'])
                    fmt.append(var_info['fmt'])
        fmt = sep.join(fmt)
        headers = sep.join(headers)
        self._save_txt_file(fname, data, fmt, headers, clean)

    def clean(self):
        for var_list in self._global_vars.values():
            var_list.clear()

    def reset(self, save_path=None, deep_reset=False):
        """
        This method resets the warehouse in order to be ready for
        running another simulation with the same configuration.

        Parameters
        ----------
        save_path: str, optional
            Path to the new results folder.
        deep_reset: bool
            Whether or not a deep reset will be performed. It means
            that the Problem setup stage will be run again as the mesh
            has been changed. Default to False.
        """
        # Reset internal counters
        self._counter = 0
        self._prev_time = None

        # Clear global and internal variables' values
        if not deep_reset:
            self.clean()
            if self._storing_functions is not None:
                for fnc in self._storing_functions.functions:
                    if isinstance(fnc, list):
                        for f in fnc:
                            interpolate(0., f)
                    else:
                        interpolate(0., fnc)

        # Update the save path
        self.save_path = save_path
        if save_path is not None and not deep_reset:
            self.file = self._create_storing_file('results')

    def get_global_variable(self, name: str):
        if not self._global_vars:
            raise RuntimeError(f"Global variables dictionary is empty")
        elif name not in self._global_vars.keys():
            raise ValueError(f"Unrecognized global variable '{name}'. Available options: '"
                             + "' '".join(self._global_vars.keys()) + "'")
        else:
            return self._global_vars[name]

    def get_global_variable_fnc(self, name: str):
        if name not in self._outputs_info['globals'].keys():
            raise ValueError(f"Unrecognized global variable '{name}'. Available options: '"
                             + "' '".join(self._global_vars.keys()) + "'")
        else:
            return self._outputs_info['globals'][name]['fnc']

    def get_global_variable_value(self, name: str):
        fnc = self.get_global_variable_fnc(name)
        return fnc()

    def _update_save_path(self, save_path):
        self.save_path = save_path
        if save_path is not None:
            self.file = self._create_storing_file('results')

    def check_globals_memory_size(self, estimate=True):
        """
        Returns memory size of stored global variables in MB

        Parameters
        ----------
        estimate : bool, optional
            Wether to do a fast estimate or the slow exact measure, by default True

        Returns
        -------
        float
            memory size in MB
        """
        if estimate:
            lower_estimate = total_size([arr[0] for arr in self.global_var_arrays])
            higher_estimate = total_size([arr[-1] for arr in self.global_var_arrays])
            size = (lower_estimate + higher_estimate) * len(self.global_var_arrays) / 2
            return size / 2**20
        else:
            return total_size(self.global_var_arrays)


def total_size(o, handlers={}):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)
