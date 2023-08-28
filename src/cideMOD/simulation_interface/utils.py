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
import os
from pathlib import Path
from typing import Union, List

from cideMOD.helpers.plotview import PlotView
from cideMOD.numerics.triggers import Trigger
from cideMOD.cell.parser import CellParser
from cideMOD.models.model_options import get_model_options
from cideMOD.main import Problem
from cideMOD.simulation_interface.battery_system import CSI
from cideMOD.simulation_interface.error_check import ErrorCheck


def run_case(options_dict: dict, cell_data: Union[dict, str], data_path: str = None,
             test_plan: Union[dict, str] = None, i_app: float = None, C_rate: float = None,
             v_app: float = None, min_step: float = 10, max_step: float = 60, t_f: float = None,
             adaptive: bool = False, time_adaptive_tol: float = 1e-2, triggers: List[Trigger] = [],
             v_min: float = None, v_max: float = None, store_delay: int = 10,
             SoC: float = 1., T_ini: float = 298.15, T_ext: float = 298.15,
             plot_globals: bool = False):
    """
    Configure and run a battery cell simulation using cideMOD.

    Parameters
    ----------
    options_dict: dict
        Dictionary containing the simulation options. For more details
        type `print(cideMOD.get_model_options().__doc__)`
    cell_data : Union[dict,str]
        Dictionary of the cell parameters or path to a JSON file
        containing them.
    data_path : str, optional
        Path to the folder where *cell_data* is together with extra data
        like materials OCVs. Required if `cell_data` is a dictionary.
    test_plan : Union[dict,str], optional
        The dictionary with the test plan or a path to a JSON file
        with the test plan. Defaults to None.
    i_app : float, optional
        Applied current. If CV use None. Defaults to None.
    C_rate : float, optional
        Discharge C-rate. Used if CC and i_app is not given.
        Defaults to None.
    v_app : Union[float,str], optional
        Applied voltage in Volts. If CC use None. Default to None.
    min_step : float, optional
        Minimum timestep length for adaptive solver in seconds.
        Defaults to 10.
    max_step: int, optional
        Maximum timestep length for adaptive solver in seconds.
        Defaults to 60.
    t_f : float, optional
        The maximum duration of the simulation in seconds.
        Defaults to None.
    adaptive : bool, optional
        Whether to use adaptive timestepping or not. Defaults to False.
    time_adaptive_tol : Union[float,int]
        Tolerance of the time-adaptive scheme. Defaults to 1e-2.
    triggers : List[Trigger], optional
        List of Triggers to check during runtime. Default to [].
    v_min: float, optional
        Minimum voltage of the simulation in Volts.
    v_max: float, optional
        Maximum voltage of the simulation in Volts.
    store_delay : int, optional
        The delay to apply between consecutive saves of the internal
        variables, in number of timesteps. Defaults to 10.
    SoC : float, optional
        Current State of Charge of the battery cell. Defaults to 1.
    T_ini : float, optional
        Uniform value of the internal temperature. Defaults to 298.15 K.
    T_ext : float, optional
        External temperature. Defaults to 298.15 K.
    plot_globals: bool, optional
        Whether or not to plot the global variables. Defaults to False.
    """
    if 'save_path' not in options_dict.keys():
        raise KeyError("A save path should be provided when using 'run_case'")
    model_options = get_model_options(**options_dict)

    if all([var is None for var in [test_plan, i_app, C_rate, v_app]]):
        raise RuntimeError("Need to provide one of 'i_app', 'C_rate', 'v_app' or 'test_plan")
    elif test_plan is not None:
        problem_cls = CSI
    else:
        problem_cls = Problem

    if problem_cls is CSI:
        csi = problem_cls(cell_data, model_options, test_plan)
        problem = csi.problem
        csi.setup()
        status = csi.run_test_plan()

    elif problem_cls is Problem:

        if not os.path.exists(cell_data):
            raise FileNotFoundError(f"Path to cell data '{cell_data}' does not exists")
        if data_path is None:
            data_path = Path(cell_data).parent
        cell_data = Path(cell_data).name

        cell = CellParser(cell_data, data_path, model_options)
        problem = problem_cls(cell, model_options)
        problem.set_cell_state(SoC=SoC, T_ini=T_ini, T_ext=T_ext)
        problem.setup()
        i_app = i_app if i_app is not None else -C_rate * cell.ref_capacity
        t_f = t_f or 3600 / abs(cell.ref_capacity / i_app) * 1.25

        if v_min is not None:
            triggers.append(Trigger(v_min, "v", mode="min"))
        if v_max is not None:
            triggers.append(Trigger(v_max, "v", mode="max"))

        status = problem.solve(i_app=i_app, v_app=v_app, min_step=min_step, max_step=max_step,
                               t_f=t_f, store_delay=store_delay, adaptive=adaptive,
                               time_adaptive_tol=time_adaptive_tol, triggers=triggers)

    if plot_globals:
        PlotView(problem)

    err = ErrorCheck(problem, status)
    return status
