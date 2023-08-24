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

import pytest
import numpy as np
from cideMOD import __path__, Problem, CSI, get_model_options, Trigger
from helpers_testplans import SEI_testplan
from helpers_pytest import run_case

pytestmark = [pytest.mark.degradation, pytest.mark.Safari]


@pytest.mark.p2d
class TestP2D_SEI:
    case = 'Safari_2009'
    model_options_dict = {'model': 'P2D',
                          'solve_SEI': True,
                          'clean_on_exit': False,
                          'particle_order': 2,
                          'dimensionless': False}
    I_app = 0.5 * 1.8
    testplan_kwargs = {'cycle_current': I_app,
                       'min_step': 2,
                       'max_step': 60,
                       'time_adaptive_tol': 1e2,
                       }
    case_kwargs = {'params_json': "params_cycling_lumped.json",
                   'problem_class': CSI,
                   'save': False,
                   'no_comparison': False,
                   'tols': [1e-3, 0.01],
                   'ref_results': {'voltage-sim': f"{case}/voltage_5cycles.txt",
                                   'delta_sei_a-sim': f"{case}/delta_sei_a_5cycles.txt",
                                   'Q_sei_a-sim': f"{case}/Q_sei_a_5cycles.txt", }
                   }

    @pytest.mark.degradation
    def test_SEI_5cycles(self):
        n_cycles = 5
        test_plan = SEI_testplan(n_cycles=n_cycles, **self.testplan_kwargs)
        return run_case(
            self.case, self.model_options_dict, test_plan=test_plan,
            **self.case_kwargs)

    @pytest.mark.degradation_quick
    def test_SEI_1cycle(self):
        n_cycles = 1
        test_plan = SEI_testplan(n_cycles=n_cycles, **self.testplan_kwargs)
        return run_case(
            self.case, self.model_options_dict, test_plan=test_plan,
            **self.case_kwargs, time_ratio_ref=5.)

    @pytest.mark.quicktest
    def test_SEI_1discharge(self):
        case_kwargs = {
            **self.case_kwargs,
            'problem_class': Problem,
            'I_app': -self.I_app,
            't_f': 3600 * 2.2,
            'triggers': [Trigger(3, "v")],
            'ref_results': {'voltage-exp': [f"{self.case}/voltage_dis_C2.txt", [0.01, 0.05]],
                            'voltage-sim': f"{self.case}/voltage_simC2.txt",
                            'delta_sei_a-sim': f"{self.case}/delta_sei_a_simC2.txt",
                            'Q_sei_a-sim': f"{self.case}/Q_sei_a_simC2.txt", }
        }

        return run_case(self.case, self.model_options_dict, **case_kwargs)

    def new_reference(self):
        return self.test_SEI_5cycles()
        # return self.test_SEI_1discharge()


if __name__ == '__main__':
    # TestP2D_SEI().test_SEI_1discharge()
    TestP2D_SEI().test_SEI_1cycle()
    # TestP2D_SEI().test_SEI_5cycles()
    pass
