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
from cideMOD import *
from cideMOD import Problem, CSI, get_model_options, Trigger
from helpers_pytest import run_case

pytestmark = [pytest.mark.literature, pytest.mark.Ai]


@pytest.mark.p2d
class TestP2D_Ai:
    case = 'Ai_2020'
    model_options_dict = {'model': 'P2D',
                          'clean_on_exit': False,
                          'dimensionless': False}
    case_kwargs = {'problem_class': Problem,
                   'triggers': [Trigger(3, 'v', mode='min')],
                   'save': False
                   }

    @pytest.mark.validation
    def test_Ai_implicit(self):
        crates = [0.5, 1, 2]
        for C_rate in crates:
            t_f = 3600 * 1.25 / abs(C_rate)
            ref_results = {'voltage-ref': f"{self.case}/V_{C_rate}C_Rieger.txt"}
            run_case(self.case, self.model_options_dict,
                     crate=C_rate, t_f=t_f,
                     ref_results=ref_results,
                     **self.case_kwargs)

    @pytest.mark.quicktest
    def test_Ai_1crate(self):
        crates = [1]
        for C_rate in crates:
            t_f = 3600 * 1.25 / abs(C_rate)
            ref_results = {'voltage-ref': f"{self.case}/V_{C_rate}C_Rieger.txt"}
            run_case(self.case, self.model_options_dict,
                     crate=C_rate, t_f=t_f,
                     ref_results=ref_results,
                     **self.case_kwargs)


if __name__ == '__main__':
    # TestP2D_Ai().test_Ai_implicit()
    TestP2D_Ai().test_Ai_1crate()
    pass
