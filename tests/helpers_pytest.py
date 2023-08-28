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
from scipy.interpolate import Akima1DInterpolator
from scipy.integrate import quad, IntegrationWarning
import matplotlib.pyplot as plt
from cideMOD import CellParser, ErrorCheck, SolverCrashed, Problem, CSI, get_model_options

# TODO: Adapt the docstring


def run_case(
    case, options_dict, problem_class=Problem,
    I_app=None, crate=None,
    data_path=None, params_json="params.json", cell_dict=None, test_plan=None,
    min_step=10, max_step=60, adaptive=False, t_f=None, overwrite=False, triggers=[],
    SOC_init=1., T_cell=298.15, T_env=298.15,
    ref_results: dict = None, delimiter=None, results_path=None,
    tols=[0.3, 0.05], time_ratio_ref=1.,
    save=False, exact_comparison=False, no_comparison=False,
):
    """
    Run the test case and compare the results to the specified reference.

    Args:
        case (str): Case name
        options (ModelOptions): Model options
        problem_class (class): Either Problem or NDProblem
        I_app (float, optional): Applied current. Defaults to None.
        crate (float, optional): Discharge C-rate. Defaults to None.
        data_path (str, optional): Path to input data. Defaults to None.
        params_json (str, optional): Params JSON filename. Defaults to "params.json".
        min_step (float, optional): min_step. Defaults to 10.
        max_step (int, optional): min_step. Defaults to 60.
        t_f (float, optional): t_f. Defaults to None.
        adaptive (bool, optional): adaptive. Defaults to False.
        overwrite (bool, optional): overwrite. Defaults to False
        triggers (list, optional): triggers. Defaults to [].
        SOC_init (float, optional): SOC_init. Defaults to 1.
        T_cell (float, optional): T_cell. Defaults to 298.
        T_env (float, optional): _description_. Defaults to 298.
        ref_results (dict, optional): Dictionary of filenames of reference results,
        e.g.,   {'voltage': 'voltage_file.txt',
                'thickness': 'thickness_file.txt',
                'temperature': 'temperature_file.txt'}. Defaults to None.
        results_path (str, optional): Path to reference results.
            Defaults to None.
        tols (list, optional): Tolerances for comparing RMSE and time_ratio.
            Defaults to [0.3, 0.05].
        save (bool, optional): Whether to save the results for Problem or NDProblem.
            Defaults to False.
        exact_comparison (bool, option): Whether to compare exactly the output to
            the reference, only works under the exact same conditions (to avoid
            interpolation). Defaults to False.
        no_comparison (bool, option): does not compare to the reference results, used to define a
        new reference. Defaults to False.
    Raises:
        RuntimeError: If the RSME is not within the tolerance
        RuntimeError: If the time_ratio is not within the tolerance
    """

    if data_path is None:
        data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", f"data/data_{case}"))
    if results_path is None:
        results_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", f"tests/ref_results"))

    if save:
        options_dict['save_path'] = case

    options = get_model_options(**options_dict)
    cell = CellParser(params_json if cell_dict is None else cell_dict, data_path, options)

    if problem_class is CSI:
        if test_plan is None:
            raise RuntimeError("Test plan was not selected for CSI")

        csi = problem_class(cell, options, test_plan)
        problem = csi.problem
        csi.setup()
        csi.run_test_plan()

    elif problem_class is Problem:
        problem = problem_class(cell, options)

        if I_app is None and crate is None:
            raise RuntimeError("Test ill defined, need to input either 'I_app' or 'crate'")

        problem.set_cell_state(SoC=SOC_init, T_ini=T_cell, T_ext=T_env)
        problem.setup()
        I_app = I_app if I_app else -crate * cell.ref_capacity

        t_f = t_f if t_f else 3600  # * cell.ref_capacity / I_app * 1.25

        status = problem.solve(
            min_step=min_step,
            max_step=max_step,
            i_app=I_app,
            t_f=t_f,
            store_delay=-1,
            adaptive=adaptive,
            triggers=triggers,
        )
        err = ErrorCheck(problem, status)
        assert not isinstance(err, SolverCrashed)

    if no_comparison:
        print("'no_comparison' flag is True: "
              "not comparing with references, USE ONLY TO DEFINE NEW REF.")
        return problem.save_path

    if not ref_results:
        raise RuntimeError("No reference results selected for comparison")

    _check_results(
        problem,
        results_path,
        ref_results,
        tols,
        delimiter,
        time_ratio_ref,
        exact_comparison,
    )


def _check_results(
    problem, results_path, ref_results: dict, tols, delimiter, time_ratio_ref,
    exact_comparison
):

    time = problem.get_global_variable('time')
    for global_var in ref_results.keys():
        global_var_ = global_var.split("-")[0]
        result_file = ref_results[global_var]

        if isinstance(result_file, list):
            tols = result_file[1] if len(result_file) == 2 else tols
            result_file = result_file[0]
        elif not isinstance(result_file, str):
            raise TypeError("Ill defined reference results, must be a str or a list.")

        model_result = np.column_stack([time, problem.get_global_variable(global_var_)])

        reference = np.genfromtxt(os.path.join(results_path, result_file), delimiter=delimiter)

        error, time_ratio = _check_results_near(model_result, reference, tols, time_ratio_ref,
                                                global_var, exact_comparison)


def _check_results_near(v1, v2, tols, time_ratio_ref, global_var, exact_comparison):
    t_ini = max(v1[0, 0], v2[0, 0])
    t_end = min(v1[-1, 0], v2[-1, 0])
    time_ratio = abs(v1[-1, 0] - v2[-1, 0] / time_ratio_ref) / t_end

    if not exact_comparison:
        c1 = Akima1DInterpolator(v1[:, 0], v1[:, 1])
        c2 = Akima1DInterpolator(v2[:, 0], v2[:, 1])
        error = lambda t: (c1(t) - c2(t)) ** 2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=IntegrationWarning)
            rmse = (quad(error, t_ini, t_end)[0] / t_end) ** 0.5
    else:
        rmse = np.sqrt(sum((v1[:, 1] - v2[:, 1]) ** 2) / v1.shape[0])

    print("\n" + "_" * 40 + "\n" + global_var)

    if tols[0] is None:
        warnings.warn("RMSE not checked against tolerance.")
    if rmse < tols[0]:
        print(f"RMSE: {rmse:.3g}\t\tOK!")
    else:
        raise RuntimeError(f"RMSE too high - RMSE: {rmse:.3g} (> {tols[0]})")

    if tols[1] is None:
        warnings.warn(f"time_ratio ({time_ratio:.3g}) not checked against tolerance.")
    if time_ratio < tols[1]:
        print(f"Time ratio: {time_ratio:.3g}\tOK!")
    else:
        raise RuntimeError(
            f"Model stopped too early or too late - Time ratio: {time_ratio:.3g} (> {tols[1]})")
    return rmse, time_ratio


def _create_comparison(simulation, reference, x_label, y_label, title, save_path):
    plt.figure()
    plt.plot(simulation[0], simulation[1], label="simulation")
    plt.plot(reference[0], reference[1], "x", label="reference")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(save_path, "validation.png"))
