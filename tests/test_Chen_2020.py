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

pytestmark = [pytest.mark.literature, pytest.mark.Chen]


@pytest.mark.p2d
class TestP2D_Chen:
    case = 'Chen_2020'
    model_options_dict = {'model': 'P2D',
                          'clean_on_exit': False,
                          'dimensionless': False}
    case_kwargs = {'params_json': 'params_tuned.json',
                   'problem_class': Problem,
                   'triggers': [Trigger(2.5, 'v', mode='min')],
                   'I_app': -5,  # 1C
                   'save': False,
                   'no_comparison': False,
                   'tols': [0.05, 0.02],
                   'ref_results': {'voltage-exp': [f"{case}/V_1C_Chen.txt", [0.05, 0.02]],
                                   'voltage-sim': [f"{case}/voltage_1C.txt", [0.001, 0.001]]}
                   }

    # @pytest.mark.quicktest
    # @pytest.mark.validation
    def test_implicit(self):
        self.case_kwargs['ref_results']['voltage-sim'][1] = [1e-5, 0.001]  # Finer tolerance
        run_case(self.case, self.model_options_dict, **self.case_kwargs)

    @pytest.mark.quicktest
    def test_dimensional(self):
        model_options = {'model': 'P2D',
                         'clean_on_exit': False,
                         'dimensionless': False}
        return run_case(self.case, model_options, **self.case_kwargs)

    # @pytest.mark.quicktest
    def test_adaptive(self):
        run_case(self.case, self.model_options_dict, **self.case_kwargs)

    @pytest.mark.exact
    def test_exact(self):
        self.case_kwargs['ref_results'] = {
            'voltage-sim': [f"{self.case}/voltage_1C.txt", [1e-8, 1e-6]]}
        return run_case(self.case, self.model_options_dict, **self.case_kwargs)

    def new_reference(self):
        # return self.test_exact()
        return self.test_dimensional()


if __name__ == '__main__':
    # TestP2D_Chen().test_implicit()
    TestP2D_Chen().test_dimensional()
    # TestP2D_Chen().test_adaptive()
    # TestP2D_Chen().test_exact()
    pass
