import os

import numpy
import pytest
from cideMOD import (
    CellParser,
    ErrorCheck,
    Problem,
    ModelOptions,
    SolverCrashed,
    Trigger,
    init_results_folder,
)
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

pytestmark = [pytest.mark.literature, pytest.mark.Chen]


def _check_results_near(v1, v2):
    t_end = min(v1[-1, 0], v2[-1, 0])
    time_ratio = abs(v1[-1, 0] - v2[-1, 0]) / t_end
    c1 = CubicSpline(v1[:, 0], v1[:, 0])
    c2 = CubicSpline(v2[:, 0], v2[:, 0])
    error = lambda t: (c1(t) - c2(t)) ** 2
    rmse = (quad(error, 0, t_end)[0] / t_end) ** 0.5
    print(rmse, time_ratio)
    assert rmse < 0.3, f"RMSE too high - {rmse}"
    assert time_ratio < 0.1, f"Model stopped too early or too late - {time_ratio}"
    return rmse, time_ratio


def _run_case(
    options,
    problem_class,
    check_voltage=True,
    check_thickness=False,
    check_temperature=False,
):
    case = "Chen_2020"
    overwrite = False
    data_path = "data/data_{}".format(case)
    cell = CellParser("params_tuned.json", data_path=data_path)
    for C_rate in [1]:
        save_path = init_results_folder(
            case, overwrite=overwrite, copy_files=[f"{data_path}/params.json"]
        )
        problem = problem_class(cell, options, save_path=save_path)
        problem.set_cell_state(1, 298, 298)
        problem.setup()
        I_app = -C_rate * problem.Q
        t_f = 3600 * 1.25 / abs(C_rate)

        v_min = Trigger(3, "v")
        status = problem.solve_ie(
            min_step=10,
            i_app=I_app,
            t_f=t_f,
            store_delay=10,
            adaptive=False,
            triggers=[v_min],
        )
        err = ErrorCheck(problem, status)
        assert not isinstance(err, SolverCrashed)
        _check_results(
            problem, data_path, check_voltage, check_thickness, check_temperature
        )


def _check_results(
    problem, data_path, C_rate, voltage=True, thickness=False, temperature=False
):
    if voltage:
        model_result = numpy.genfromtxt(os.path.join(problem.save_path, "voltage.txt"))
        reference = numpy.genfromtxt(os.path.join(data_path, f"V_{C_rate}C_Chen.txt"))
        error, time_ratio = _check_results_near(model_result, reference)


@pytest.mark.p2d
class TestP2D:
    @pytest.mark.validation
    def test_implicit(self):
        simulation_options = ModelOptions(
            mode="P2D", particle_coupling="implicit", particle_order=2
        )
        problem_class = Problem
        _run_case(simulation_options, problem_class)
