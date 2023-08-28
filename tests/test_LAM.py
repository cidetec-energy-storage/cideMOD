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
import json
import pytest
import numpy as np
from cideMOD import (
    __path__,
    Problem,
    get_model_options,
    CSI,
    Trigger
)
from helpers_testplans import LAM_testplan
from helpers_pytest import run_case

pytestmark = [pytest.mark.degradation, pytest.mark.Ai]


@pytest.mark.p2d
class TestP2D_LAM:
    case = 'Ai_2020'
    params = "params_LAM.json"
    data_path = os.path.abspath(os.path.join(__path__[0], "../..", f"data/data_{case}"))
    with open(os.path.join(data_path, params), "r+") as jsonFile:
        dataCell = json.load(jsonFile)

    simulation_options = get_model_options(
        model='P2D', clean_on_exit=False, solve_LAM=True, dimensionless=False)
    problem_class = CSI
    cycle_current = 2.28 / 34  # [A]
    testplan_kwargs = {'min_step': 10,
                       'max_step': 60,
                       'time_adaptive_tol': 1e-1,
                       }
    case_kwargs = {'params_json': params,
                   'problem_class': CSI,
                   'save': False,
                   'no_comparison': False
                   }

    @pytest.mark.degradation
    def test_LAM_2cycles(self):
        n_cycles = 2
        crate = 1
        betas = [1e-4 / 3600]  # [1e-4/3600, 1e-3/3600, 1e-2/3600]
        for beta in betas:
            self.dataCell["positive_electrode"]["LAM"]["beta"]["value"] = beta
            self.dataCell["negative_electrode"]["LAM"]["beta"]["value"] = beta

            test_plan = LAM_testplan(n_cycles=n_cycles, cycle_current=crate * self.cycle_current,
                                     **self.testplan_kwargs)

            ref_results = {'voltage-pybamm': [f"LAM/voltage_PyBaMM_{crate}C.txt", [0.05, 0.03]],
                           'voltage-sim': [f"LAM/voltage_{crate}C.txt", [0.01, 0.02]]}
            return run_case(
                self.case, self.simulation_options,
                test_plan=test_plan,
                ref_results=ref_results, tols=[0.05, 0.03],
                **self.case_kwargs)

    @pytest.mark.degradation_quick
    @pytest.mark.quicktest
    def test_LAM_1cycles(self):
        n_cycles = 1
        crate = 1
        beta = 1e-4 / 3600
        self.dataCell["positive_electrode"]["LAM"]["beta"]["value"] = beta
        self.dataCell["negative_electrode"]["LAM"]["beta"]["value"] = beta

        test_plan = LAM_testplan(n_cycles=n_cycles, cycle_current=crate * self.cycle_current,
                                 **self.testplan_kwargs)

        ref_results = {'voltage-pybamm': [f"LAM/voltage_PyBaMM_{crate}C.txt", [0.05, 0.03]],
                       'voltage-sim': [f"LAM/voltage_{crate}C.txt", [0.01, 0.03]]}
        run_case(self.case, self.simulation_options,
                 test_plan=test_plan,
                 ref_results=ref_results, time_ratio_ref=2., tols=[0.05, 0.1],
                 **self.case_kwargs)

    def new_reference(self):
        return self.test_LAM_2cycles()


if __name__ == '__main__':
    # TestP2D_LAM().test_LAM_2cycles()
    TestP2D_LAM().test_LAM_1cycles()
    pass
