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

from cideMOD import Problem, CSI, get_model_options, Trigger
from helpers_pytest import run_case

pytestmark = [pytest.mark.literature, pytest.mark.Xu, pytest.mark.thermal]


@pytest.mark.p2d
class TestP2D_thermal:
    case = 'Xu_2015'
    crate = 1
    case_kwargs = {'params_json': 'params.json',
                   'problem_class': Problem,
                   'triggers': [Trigger(2.1, "v", mode='min')],
                   'crate': crate,
                   'save': False,
                   'no_comparison': False,
                   'tols': [0.05, 0.05],
                   'ref_results': {
                       'voltage-exp': [f"{case}/V_{crate}C_Xu.txt", [0.05, 0.05]],
                       'voltage-sim': [f"{case}/voltage_{crate}C.txt", [0.01, 0.01]],
                       'T_max-sim': [f"{case}/T_max_{crate}C.txt", [0.01, 0.01]]}
                   }
    model_options_dict = {'model': 'P2D',
                          'solve_thermal': True,
                          'clean_on_exit': False,
                          'dimensionless': True}

    # @pytest.mark.quicktest
    # @pytest.mark.validation
    def test_implicit(self):
        return run_case(self.case, self.model_options_dict, **self.case_kwargs)

    @pytest.mark.quicktest
    @pytest.mark.validation
    def test_implicit_dimensional(self):
        self.model_options_dict['dimensionless'] = False
        return self.test_implicit()

    def new_reference(self):
        return self.test_implicit_dimensional()


if __name__ == '__main__':
    # TestP2D_thermal().test_implicit()
    TestP2D_thermal().test_implicit_dimensional()
    pass
