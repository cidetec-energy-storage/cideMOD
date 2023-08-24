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

"""
This module provides useful classes and functions to setup/handle the
battery cell simulation.
"""

from cideMOD.numerics.fem_handler import (
    BlockFunction,
    BlockFunctionSpace,
    interpolate,
    assign,
    assemble_scalar,
    block_derivative
)
from cideMOD.numerics.time_scheme import TimeScheme
from cideMOD.numerics.helper import (
    analyze_jacobian,
    plot_jacobian,
    print_diagonal_statistics,
    estimate_condition_number
)
from cideMOD.numerics.solver import NonlinearBlockProblem, NewtonBlockSolver
from cideMOD.numerics.polynomials import Lagrange
from cideMOD.numerics.triggers import Trigger, TriggerDetected, TriggerSurpassed, SolverCrashed

__all__ = [
    "BlockFunction",
    "BlockFunctionSpace",
    "interpolate",
    "assign",
    "assemble_scalar",
    "block_derivative",
    "TimeScheme",
    "analyze_jacobian",
    "plot_jacobian",
    "print_diagonal_statistics",
    "estimate_condition_number",
    "NonlinearBlockProblem",
    "NewtonBlockSolver",
    "Lagrange",
    "Trigger",
    "TriggerDetected",
    "TriggerSurpassed",
    "SolverCrashed"
]
