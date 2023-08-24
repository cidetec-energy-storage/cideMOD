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
r"""The cideMOD library contains a suite of modules needed to simulate a battery"""

from cideMOD.helpers import (VerbosityLevel, LogLevel, PlotView,
                             init_results_folder, plot_list_variable)
from cideMOD.numerics import SolverCrashed, Trigger, TriggerDetected
from cideMOD.cell import CellParser
from cideMOD.models import get_model_options, models_info
from cideMOD.main import Problem
from cideMOD.simulation_interface import CSI, DEFAULTS, ErrorCheck, run_case


__all__ = [
    "CSI",
    "DEFAULTS",
    "run_case",
    "SolverCrashed",
    "Trigger",
    "TriggerDetected",
    "Problem",
    "CellParser",
    "ErrorCheck",
    "get_model_options",
    "models_info",
    "VerbosityLevel",
    "LogLevel",
    "init_results_folder",
    "plot_list_variable",
    "PlotView"
]
