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
from abc import ABC, abstractmethod

from cideMOD.models.base.base_models.inputs import BaseCellModelInputs
from cideMOD.models.base.base_models.outputs import BaseCellModelOutputs
from cideMOD.models.base.base_models.preprocessing import BaseCellModelPreprocessing
from cideMOD.models.base.base_models.equations import BaseCellModelEquations

__all__ = [
    "BaseCellModel",
    "BaseCellModelInputs",
    "BaseCellModelOutputs",
    "BaseCellModelPreprocessing",
    "BaseCellModelEquations"
]


class BaseCellModel(
    BaseCellModelInputs,
    BaseCellModelOutputs,
    BaseCellModelPreprocessing,
    BaseCellModelEquations,
    ABC
):
    """
    Abstract base class for testing models

    Parameters
    ----------
    name : str
        Name of the model. Defaults to 'Unnamed model'
    mtype : str
        Type of model. Should be one of 'PXD', 'SPM', 'P2D-2D', 'PM'. Defaults to 'Unknown type'
    time_scheme : str
        Type of time scheme. It must be either 'implicit' or 'explicit'
    root : bool
        Whether or not is a root model
    hierarchy : int
        Hierarchy level. The lower the hierarchy the greater its priority. Notice that
        `root model > implicit models > explicit models` is always true regardless of
        its hierarchy level

    .. note::

        Notice that some model methods may be overrided by other models with higher hierarchy level

    """

    _name_ = "Unnamed model"
    _mtype_ = "Unknown type"
    _time_scheme_ = "implicit"
    _root_ = False
    _hierarchy_ = -1

    @classmethod
    @property
    def name(cls):
        return cls._name_

    @classmethod
    @property
    def mtype(cls):
        return cls._mtype_

    @classmethod
    @property
    def time_scheme(cls):
        return cls._time_scheme_

    @classmethod
    @property
    def root(cls):
        return cls._root_

    @classmethod
    @property
    def hierarchy(cls):
        return cls._hierarchy_

    def __init__(self, options):
        self.options = options
