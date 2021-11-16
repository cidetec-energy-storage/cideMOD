r"""This module contains the logic of the intermediary between the user and a battery."""

from enum import Enum

from PXD.bms.battery_system import (
    BMS,
    DEFAULT_EVENT,
    DEFAULT_INPUT,
    DEFAULT_PROFILE,
    DEFAULT_SIMULATION_OPTIONS,
    DEFAULT_TEST_PLAN,
)
from PXD.bms.triggers import SolverCrashed, Trigger, TriggerDetected


class DEFAULTS(Enum):
    EVENT = DEFAULT_EVENT
    INPUT = DEFAULT_INPUT
    PROFILE = DEFAULT_PROFILE
    SIMULATION_OPTIONS = DEFAULT_SIMULATION_OPTIONS
    TEST_PLAN = DEFAULT_TEST_PLAN


__all__ = ["BMS", "Trigger", "TriggerDetected", "SolverCrashed", "DEFAULTS"]
