#
# Copyright (c) 2021 CIDETEC Energy Storage.
#
# This file is part of PXD.
#
# PXD is free software: you can redistribute it and/or modify
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

class StrongCoupledPM(ABC):

    def fields(self, *args):
        """
        List of fields that will be used in the mesoscale formulation, defaults to an empty list.

        Returns
        -------
        List
            Names of fields used
        """
        return []

    def initial_guess(self, *args, **kwargs):
        """
        Assign initial guess to the corresponding fields

        Parameters
        ----------
        Function : Function
            Dolfin Function to assign initial values
        Expression : Expression or Function
            Dolfin Expression or Function containing the initial values
        """

    def wf_0(self, *args, **kwargs):
        """
        Weak Formulation for the initialization of the problem dc_s/dt = 0
        """
        return []

    def wf_explicit_coupling(self, *args, **kwargs):
        """
        Weak Formulation for this implicit method
        """
        return []

    def wf_implicit_coupling(self, *args, **kwargs):
        """
        Weak formulation without using this implicit method
        """
        return []

    def update_functions(self, *args, **kwargs):
        """
        Update implicit model with results calculated without it
        """

    def c_s_surf(self, *args, **kwargs):
        """
        Update implicit model with results calculated without it
        """

class WeakCoupledPM(ABC):
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def initial_guess(self):
        pass

    @abstractmethod
    def microscale_update(self):
        pass

