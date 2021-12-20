#
# Copyright (c) 2021 CIDETEC Energy Storage.
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

from cideMOD.bms import BMS, DEFAULTS, SolverCrashed, Trigger, TriggerDetected
from cideMOD.helpers import CellParser, ErrorCheck, init_results_folder
from cideMOD.pxD import NDProblem, Problem, StressProblem
from cideMOD.models.model_options import ModelOptions

__all__ = [
    "BMS",
    "DEFAULTS",
    "SolverCrashed",
    "Trigger",
    "TriggerDetected",
    "Problem",
    "StressProblem",
    "NDProblem",
    "CellParser",
    "ErrorCheck",
    "ModelOptions",
    "init_results_folder",
]
