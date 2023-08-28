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
import warnings
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad, IntegrationWarning
import matplotlib.pyplot as plt
try:
    from cideMOD import (CellParser, ErrorCheck, SolverCrashed, init_results_folder)
except BaseException:
    pass


def LAM_testplan(n_cycles, cycle_current, min_step=10, max_step=60, time_adaptive_tol=1e-2):
    return {
        'initial_state': {
            'SOC': 1,
            'exterior_temperature': 298
        },
        'steps': [{
            "name": "Discharge Cycle",
            "type": "Cycle",
            "count": n_cycles,
            "steps": [
                {
                    "name": "Discharge",
                    "type": "Current",
                    "value": -cycle_current,
                    "unit": "A",
                    "t_max": {"value": 2, "unit": "h"},
                    "store_delay": -1,
                    "min_step": min_step,
                    "max_step": max_step,
                    "adaptive": True,
                    "time_adaptive_tol": time_adaptive_tol,
                    "events": [
                        {
                            "type": "Voltage",
                            "value": 3,
                            "unit": "V",
                            "atol": 1e-4,
                            "rtol": 1e-3,
                            "goto": "Next"
                        }
                    ]
                },
                {
                    "name": "Pause",
                    "type": "Rest",
                    "t_max": {"value": 600, "unit": "s"},
                    "store_delay": -1,
                    "min_step": min_step,
                    "max_step": max_step,
                    "adaptive": True,
                    "time_adaptive_tol": time_adaptive_tol,
                },
                {
                    "name": "Charge-CC",
                    "type": "Current",
                    "value": cycle_current,
                    "unit": "A",
                    "t_max": {"value": 2, "unit": "h"},
                    "store_delay": -1,
                    "min_step": min_step,
                    "max_step": max_step,
                    "adaptive": True,
                    "time_adaptive_tol": time_adaptive_tol,
                    "events": [
                        {
                            "type": "Voltage",
                            "value": 4.2,
                            "unit": "V",
                            "atol": 1e-4,
                            "rtol": 1e-3,
                            "goto": "Next"
                        }
                    ]
                },
                {
                    "name": "Charge-CV",
                    "type": "Voltage",
                    "value": 4.199,
                    "unit": "V",
                    "t_max": {"value": 600, "unit": "s"},
                    "store_delay": -1,
                    "min_step": min_step,
                    "max_step": max_step,
                    "adaptive": True,
                    "time_adaptive_tol": time_adaptive_tol,
                    "events": [{
                        "type": "Current",
                        "value": 0.02,
                        "unit": "C",
                        "atol": 1e-4,
                        "rtol": 1e-3,
                        "goto": "Next"
                    }]
                },
                {
                    "name": "Pause",
                    "type": "Rest",
                    "t_max": {"value": 60, "unit": "s"},
                    "store_delay": -1,
                    "min_step": min_step,
                    "max_step": max_step,
                    "adaptive": True,
                    "time_adaptive_tol": time_adaptive_tol,
                },
            ]}]
    }


def SEI_testplan(n_cycles, cycle_current, min_step=10, max_step=60, time_adaptive_tol=1e-2):
    return {
        "initial_state": {"SOC": 1, "exterior_temperature": 298.15},
        "steps": [
            {"name": "Cycling",
             "type": "Cycle",
             "count": n_cycles,
             "steps": [{
                 "name": "Discharge",
                 "type": "Current",
                 "value": -cycle_current,
                 "unit": "A",
                 "t_max": {"value": 150, "unit": "min"},
                 "store_delay": -1,
                 "min_step": min_step,
                 "max_step": max_step,
                 "initial_step": min_step,
                 "time_adaptive_tol": time_adaptive_tol,
                 "adaptive": True,
                 "events": [{
                     "type": "Voltage",
                     "value": 2,
                     "unit": "V",
                     "atol": 1e-4,
                     "rtol": 1e-3,
                     "goto": "Next",
                 }],
             }, {
                 "name": "Charge",
                 "type": "Current",
                 "value": cycle_current,
                 "unit": "A",
                 "t_max": {"value": 150, "unit": "min"},
                 "store_delay": -1,
                 "min_step": min_step,
                 "max_step": max_step,
                 "initial_step": min_step,
                 "time_adaptive_tol": time_adaptive_tol,
                 "adaptive": True,
                 "events": [{
                     "type": "Voltage",
                     "value": 4.2,
                     "unit": "V",
                     "atol": 1e-4,
                     "rtol": 1e-3,
                     "goto": "Next",
                 }],
             }, {
                 "name": "Charge",
                 "type": "Voltage",
                 "value": 4.2,
                 "unit": "V",
                 "t_max": {"value": 120, "unit": "min"},
                 "store_delay": -1,
                 "min_step": min_step,
                 "max_step": max_step,
                 "initial_step": min_step,
                 "time_adaptive_tol": time_adaptive_tol,
                 "adaptive": True,
                 "events": [{
                     "type": "Current",
                     "value": 0.05,
                     "unit": "C",
                     "atol": 1e-6,
                     "rtol": 1e-4,
                     "goto": "Next",
                 }],
             }
             ]
             }
        ],
    }
