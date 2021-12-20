import os

import numpy
import pytest
from cideMOD import *
from cideMOD import (
    CellParser,
    ErrorCheck,
    NDProblem,
    Problem,
    ModelOptions,
    SolverCrashed,
    StressProblem,
    Trigger,
    init_results_folder,
)
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

pytestmark = [pytest.mark.literature, pytest.mark.Ai]


def _check_results_near(v1, v2):
    t_end = min(v1[-1, 0], v2[-1, 0])
    time_ratio = abs(v1[-1, 0] - v2[-1, 0]) / t_end
    c1 = CubicSpline(v1[:, 0], v1[:, 0])
    c2 = CubicSpline(v2[:, 0], v2[:, 0])
    error = lambda t: (c1(t) - c2(t)) ** 2
    rmse = (quad(error, 0, t_end)[0] / t_end) ** 0.5
    print(rmse, time_ratio)
    assert rmse < 0.3, f"RMSE too high - {rmse}"
    assert time_ratio < 0.05, f"Model stopped too early or too late - {time_ratio}"
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


def _run_case(
    options,
    problem_class,
    check_voltage=True,
    check_thickness=False,
    check_temperature=False,
):
    case = "Ai_2020"
    overwrite = False
    data_path = "data/data_{}".format(case)
    cell = CellParser("params.json", data_path=data_path)
    for C_rate in [0.5, 1, 2]:
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
            problem,
            data_path,
            C_rate,
            check_voltage,
            check_thickness,
            check_temperature,
        )


def _check_results(
    problem, data_path, C_rate, voltage=True, thickness=False, temperature=False
):
    if voltage:
        model_result = numpy.genfromtxt(os.path.join(problem.save_path, "voltage.txt"))
        reference = numpy.genfromtxt(os.path.join(data_path, f"V_{C_rate}C_Rieger.txt"))
        error, time_ratio = _check_results_near(model_result, reference)
        _create_comparison(
            model_result.T,
            reference.T,
            "Time [s]",
            "Voltage [V]",
            f"Ai_2020 @ {C_rate} C",
            problem.save_path,
        )
    if thickness:
        model_result = numpy.genfromtxt(
            os.path.join(problem.save_path, "thickness.txt")
        )
        reference = numpy.genfromtxt(
            os.path.join(data_path, f"thick_{C_rate}C_Rieger.txt")
        )
        reference[:, 1] *= 34e6
        error, time_ratio = _check_results_near(model_result, reference)
    if temperature:
        model_result = numpy.genfromtxt(
            os.path.join(problem.save_path, "temperature.txt")
        )
        reference = numpy.genfromtxt(os.path.join(data_path, f"T_{C_rate}C_Rieger.txt"))
        reference[:, 1] += model_result[0, 1]
        error, time_ratio = _check_results_near(model_result, reference)


@pytest.mark.p2d
class TestP2DAi:
    @pytest.mark.validation
    def test_Ai_implicit(self):
        simulation_options = ModelOptions(
            mode="P2D", particle_coupling="implicit", particle_order=2
        )
        problem_class = Problem
        _run_case(simulation_options, problem_class)

    @pytest.mark.validation
    def test_Ai_temperature_implicit(self):
        simulation_options = ModelOptions(
            mode="P2D",
            particle_coupling="implicit",
            particle_order=2,
            solve_mechanic=True,
            solve_thermal=True,
        )
        problem_class = StressProblem
        _run_case(simulation_options, problem_class, check_temperature=True)
