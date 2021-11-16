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

