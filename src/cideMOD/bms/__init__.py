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
r"""This module contains the logic of the intermediary between the user and a battery."""

from enum import Enum

from cideMOD.bms.battery_system import (
    BMS,
    DEFAULT_EVENT,
    DEFAULT_INPUT,
    DEFAULT_PROFILE,
    DEFAULT_SIMULATION_OPTIONS,
    DEFAULT_TEST_PLAN,
)
from cideMOD.bms.triggers import SolverCrashed, Trigger, TriggerDetected


class DEFAULTS(Enum):
    EVENT = DEFAULT_EVENT
    INPUT = DEFAULT_INPUT
    PROFILE = DEFAULT_PROFILE
    SIMULATION_OPTIONS = DEFAULT_SIMULATION_OPTIONS
    TEST_PLAN = DEFAULT_TEST_PLAN


__all__ = ["BMS", "Trigger", "TriggerDetected", "SolverCrashed", "DEFAULTS"]
