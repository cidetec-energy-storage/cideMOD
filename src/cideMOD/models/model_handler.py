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
import inspect
import itertools
from collections import OrderedDict
from typing import Union, Callable

from cideMOD.helpers.miscellaneous import ParsedList
from cideMOD.numerics.fem_handler import BlockFunction, BlockFunctionSpace
from cideMOD.numerics.time_scheme import TimeScheme
from cideMOD.cell.dimensional_analysis import DimensionalAnalysis
from cideMOD.cell.parser import CellParser
from cideMOD.cell.components import BatteryCell
from cideMOD.cell.variables import ProblemVariables
from cideMOD.cell.equations import ProblemEquations
from cideMOD.cell.warehouse import Warehouse
from cideMOD.mesh.base_mesher import BaseMesher
from cideMOD.models.base import BaseCellModel
from cideMOD.models import get_model_types, model_factory


class CellModelList(ParsedList):
    """
    Container abstracting a list of BaseCellModel. This class sorts by
    hierarchy the given list of models
    """

    _white_list_ = BaseCellModel

    def __init__(self, model_list, instances=True):
        self.instances = instances
        if len(set([model._mtype_ for model in model_list])) > 1:
            raise ValueError("model_list must contains models of the same type")
        for model in model_list:
            self._add_model(model)

        # Categorize models
        self.root = self[0]
        self._implicit = [model for model in self if model.time_scheme == 'implicit']
        self._explicit = [model for model in self if model.time_scheme == 'explicit']

        self._check_models_compatibility()

    def _add_model(self, new_model: BaseCellModel):
        """
        Add a new model following
        `root model > implicit models > explicit models`
        sorted by hierarchy
        """
        self._check_item(new_model)
        if new_model.root:
            if bool(self) and self[0].root:
                raise RuntimeError("There should be just one root model")
            super(ParsedList, self).insert(0, new_model)
            return
        for i, model in enumerate(self):
            has_priority = (not model.root and model.time_scheme == 'explicit'
                            and new_model.time_scheme == 'implicit')
            has_to_yield = model.root or (model.time_scheme == 'implicit'
                                          and new_model.time_scheme == 'explicit')
            if has_priority or (not has_to_yield and model.hierarchy > new_model.hierarchy):
                super(ParsedList, self).insert(i, new_model)
                return
        super(ParsedList, self).append(new_model)

    def _check_models_compatibility(self):
        """Check models compatibility"""
        # TODO: Right now just check that all models shares the same model type.
        #       Maybe in the future more checks will be needed
        if not bool(self):
            raise RuntimeError("No models have been selected")
        elif not self[0]._root_:
            raise RuntimeError("No root model has been selected")
        for model in self[1:]:
            if model.mtype != self.root.mtype:
                raise TypeError(f"Model '{model.name}' with type '{model.mtype}' is not compatible"
                                + f" with model '{self.root.name}' with type '{self.root.mtype}'")

    def _check_item(self, value):
        cls = self.__class__
        if (self.instances and isinstance(value, self._white_list_)
                or not self.instances and issubclass(value, self._white_list_)):
            return
        elif not isinstance(self._white_list_, tuple):
            raise TypeError(
                f"{cls.__name__} only admit elements of type '{self._white_list_.__name__}'")
        else:
            raise TypeError(f"{cls.__name__} only admit elements of type '"
                            + "' '".join([item.__name__ for item in self._white_list_]) + "'")


class ModelHandler(CellModelList):
    """
    Container abstracting a list of BaseCellModel. Enable to access
    the models attributes via a single call. It builds a bridge between
    models and cideMOD classes.
    """

    def __init__(self, options):
        self.options = options
        self._build_models()
        self._check_models_compatibility()

    def _build_models(self):
        """Initialize the requested models from the model options"""
        # Get the models corresponding to this model type
        mtype = get_model_types(self.options.model)
        models = model_factory(mtype)

        # Decide whether or not to include every model
        for model_cls in models:
            if model_cls.is_active_model(self.options):
                active_model = model_cls(self.options)
                self._add_model(active_model)

        # Categorize models
        self.root = self[0] if self[0].root else None
        self._implicit = [model for model in self if model.time_scheme == 'implicit']
        self._explicit = [model for model in self if model.time_scheme == 'explicit']

    def _model_iterator(self, name: str, return_action: Union[str, Callable] = 'append',
                        args: tuple = (), kwargs: dict = {}, not_exist_ok=False):
        """
        Iterates over the cell models

        Parameters
        ----------
        name : str
            Name of the model attribute to iterate over
        return_action: Union[str,Callable], optional
            Postprocessing action. Available options:
            - None : Return nothing
            - 'append': Append every model output. Used to gather single values
            - 'concatenate': Concatenate every model output. Used to concatenate lists
               or dictionaries
            - 'unzip': Unzip the model outputs. Used to gather multiple values
            - 'unzip-concatenate': Unzip and concatenate the model outputs. Used to
               concatenate multiple lists
            - Callable: Pass the list of outputs to the given postprocessing function
        args : tuple
            Arguments to be passed to the cell model method
        kwargs : dict
            Keyword arguments to be passed to the cell model method

        Returns
        -------
        Union[List, Any]
            Postprocessed output of the models
        """
        outs = None
        for model in self:
            if not hasattr(model, name):
                if not_exist_ok:
                    continue
                else:
                    raise AttributeError(f"Model '{model.name}' object has no attribute '{name}'")
            attr = getattr(model, name)
            out = attr(*args, **kwargs) if callable(attr) else attr
            if outs is None:
                outs = [] if not (return_action == 'concatenate' and isinstance(out, dict)) else {}
                out_type = type(out)  # TODO: just for debugging
            elif not isinstance(out, out_type):
                raise RuntimeError(f"Model {model.name} returned an output of type "
                                   + f"'{type(out)}' instead of '{out_type}'")
            if return_action != 'concatenate':
                outs.append(out)
            elif isinstance(outs, dict):
                outs.update(out)
            else:
                outs.extend(out)
        if return_action is None:
            return
        elif return_action in ['append', 'concatenate']:
            return outs
        elif return_action == 'unzip':
            return list(zip(*outs))
        elif return_action == 'unzip-concatenate':
            return [list(itertools.chain(*items)) for items in zip(*outs)]
        elif callable(return_action):
            return return_action(outs)
        else:
            raise ValueError(f"Unrecognized return_action '{return_action}'. Type "
                             + "help(ModelHandler._model_iterator) for more information")

    @classmethod
    def _check_item(cls, value):
        return super(CellModelList, cls)._check_item(value)

    def copy(self):
        raise NotImplementedError

    # ******************************************************************************************* #
    # ***                                   Inputs. Problem                                   *** #
    # ******************************************************************************************* #

    def set_cell_state(self, problem, **kwargs) -> None:
        """
        This method iterates over the active models to set the current
        state of the cell.

        Parameters
        ----------
        problem: Problem
            Object that handles the battery cell simulation.
        kwargs: dict
            Dictionary containing the parameters that describe the cell
            state. To know more type :meth:`cideMOD.info(
            'set_cell_state', model_options=model_options)`
        """
        valid_kwargs = set()
        for model in self:
            # Get valid kwargs for this model
            sig = inspect.signature(model.set_cell_state)
            model_kwargs = [key for key in list(sig.parameters.keys())[1:]]
            valid_kwargs.update(model_kwargs)

            # Set model cell state variables
            model.set_cell_state(problem, **{k: v for k, v in kwargs.items() if k in model_kwargs})

        invalid_kwargs = set(kwargs.keys()).difference(valid_kwargs)
        if invalid_kwargs:
            raise TypeError(f"ModelHandler.set_cell_state got unexpected keyword arguments: '"
                            + "' '".join(invalid_kwargs) + "'")

    # ******************************************************************************************* #
    # ***                                 Inputs. CellParser                                  *** #
    # ******************************************************************************************* #

    def build_cell_components(self, cell) -> None:
        """
        This method iterates over the active models to build the
        components of the cell that fit our model type.

        Parameters
        ----------
        cell: CellParser
            Parser of the cell dictionary.
        """
        return self._model_iterator('build_cell_components', return_action=None, args=(cell,))

    def parse_cell_structure(self, cell) -> None:
        """
        This method iterates over the active models to parse the cell
        structure.

        Parameters
        ----------
        cell: CellParser
            Parser of the cell dictionary.
        """
        unrecognized_components = set()
        for model in self:
            out = model.parse_cell_structure(cell)
            if not out:
                pass
            elif out is True:
                return
            else:
                unrecognized_components.add(out)
        raise ValueError("Unable to parse the cell structure. Unrecognized components: '"
                         + "' '".join(unrecognized_components) + "'")

    def parse_component_parameters(self, component) -> None:
        """
        This method iterates over the active models to parse the
        parameters of the given component.

        Parameters
        ----------
        component: BaseComponentParser
            Object that parses the component parameters.
        """
        self._model_iterator(f'parse_{component._name_}_parameters',
                             return_action=None, args=(component,), not_exist_ok=True)

    def compute_reference_cell_properties(self, cell: CellParser) -> None:
        """
        This method iterates over the active models to compute the
        general reference cell properties.

        Parameters
        ----------
        cell: CellParser
            Parser of the cell dictionary.

        Notes
        -----
        This method is called once the cell parameters has been parsed.
        """
        self._model_iterator(f'compute_reference_cell_properties',
                             return_action=None, args=(cell,))

    # ******************************************************************************************* #
    # ***                         Preprocessing. DimensionalAnalysis                          *** #
    # ******************************************************************************************* #

    def build_dimensional_analysis(self, DA: DimensionalAnalysis, cell: CellParser):
        """
        This method computes the dimensionless numbers that arise from
        the dimensional analysis.

        Parameters
        ----------
        DA: DimensionalAnalysis
            Object where the dimensional analysis is performed.
        cell: CellParser
            Parser of the cell dictionary.
        """
        self._model_iterator('build_reference_parameters',
                             return_action=None, args=(DA, cell,))
        self._model_iterator('build_dimensionless_numbers',
                             return_action=None, args=(DA, cell,))

    def scale_variable(self, name: str, value):
        """
        This method scales the given variable.

        Parameters
        ----------
        name: str
            Name of the variable to be scaled.
        value: Any
            Value to be scaled.

        Returns
        -------
        Any
            Scaled value of the variable.

        Examples
        --------
        >>> models.scale_variable('c_e', 1000)
        0
        """
        raise NotImplementedError

    def unscale_variable(self, name: str, value):
        """
        This method unscales the given variable.

        Parameters
        ----------
        name: str
            Name of the variable to be unscaled.
        value: Any
            Value to be unscaled.

        Returns
        -------
        Any
            Unscaled value of the variable.

        Examples
        --------
        >>> models.unscale_variable('c_e', 0)
        1000
        """
        raise NotImplementedError

    def scale_variables(self, variables: dict):
        """
        This method scales the given variables.

        Parameters
        ----------
        variables: Dict[str, Any]
            Dictionary containing the names and the values of the
            variables to be scaled.

        Returns
        -------
        dict
            Dictionary containing the scaled variables.

        Examples
        --------
        >>> variables = {'c_e': 1000, 'c_s_a': 28700}
        >>> models.scale_variables(variables)
        {'c_e': 0, 'c_s_a': 1}
        """
        raise NotImplementedError

    def unscale_variables(self, variables: dict):
        """
        This method unscales the given variables.

        Parameters
        ----------
        variables: Dict[str, Any]
            Dictionary containing the names and the values of the
            variables to be unscaled.

        Returns
        -------
        dict
            Dictionary containing the unscaled variables.

        Examples
        --------
        >>> variables = {'c_e': 0, 'c_s_a': 1}
        >>> models.unscale_variables(variables)
        {'c_e': 1000, 'c_s_a': 28700}
        """
        raise NotImplementedError

    # ******************************************************************************************* #
    # ***                             Preprocessing. BatteryCell                              *** #
    # ******************************************************************************************* #

    def set_component_parameters(self, component, problem):
        """
        This method iterates over the active models to preprocess the
        component parameters.

        Parameters
        ----------
        component: BaseCellComponent
            Object where component parameters are preprocessed and
            stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        self._model_iterator(f'set_{component._name_}_parameters',
                             return_action=None, args=(component, problem), not_exist_ok=True)

    def compute_cell_properties(self, cell: BatteryCell):
        """
        This method iterates over the active models to compute the
        general cell properties.

        Parameters
        ----------
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.

        Notes
        -----
        This method is called once the cell parameters has been
        preprocessed.
        """
        self._model_iterator('compute_cell_properties',
                             return_action=None, args=(cell,))

    # ******************************************************************************************* #
    # ***                               Preprocessing. Problem                                *** #
    # ******************************************************************************************* #

    def set_state_variables(self, mesher, V, V_vec, problem) -> list:
        """
        This method iterates over the active models to set the state
        variables.

        Parameters
        ----------
        mesher : BaseMesher
            Object that contains the mesh information.

        V : dolfinx.fem.FunctionSpace
            Common FunctionSpace to be used for each model.

        V_vec : dolfinx.fem.VectorFunctionSpace
            Common VectorFunctionSpace to be used for each model.
        problem: Problem
            Object that handles the battery cell simulation.

        Returns
        -------
        List(Tuple(str, numpy.ndarray, dolfinx.fem.FunctionSpace))
            Returns a list of tuples, each one containing the name, the
            subdomain and the function space of the state variable.
        """
        state_vars = []
        self._model_iterator('set_state_variables', return_action=None,
                             args=(state_vars, mesher, V, V_vec, problem))
        return state_vars

    def set_problem_variables(self, var: ProblemVariables, DT: TimeScheme, problem) -> None:
        """
        This method iterates over the active models to set the problem
        variables.

        Parameters
        ----------
        var: ProblemVariables
            Object that store the preprocessed problem variables.
        DT: TimeScheme
            Object that provide the temporal derivatives with the
            specified scheme.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        return self._model_iterator('set_problem_variables',
                                    return_action=None, args=(var, DT, problem))

    def set_dependent_variables(self, var: ProblemVariables,
                                cell: BatteryCell, DT: TimeScheme, problem) -> None:
        """
        This method iterates over the active models to set the dependent
        variables.

        Parameters
        ----------
        var: ProblemVariables
            Object that store the preprocessed problem variables.
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
        DT: TimeScheme
            Object that provide the temporal derivatives with the
            specified scheme.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        return self._model_iterator('set_dependent_variables',
                                    return_action=None, args=(var, cell, DT, problem))

    def initial_guess(self, f: BlockFunction, var: ProblemVariables, cell: BatteryCell, problem):
        """
        This method iterates over the active models to initialize the
        state variables based on the initial conditions and assuming
        that the simulation begins after a stationary state.

        Parameters
        ----------
        f: BlockFunction
            Block function that contain the state variables to be
            initialized.
        var: ProblemVariables
            Object that store the preprocessed problem variables.
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        # Initialize the BlockFuction to 0.
        f.clear()
        # Let the active models initialize the state variables
        self._model_iterator('initial_guess', return_action=None, args=(f, var, cell, problem))

    def get_solvers_info(self, problem) -> dict:
        """
        This method iterates over the active models to get the solvers
        information.

        Parameters
        ----------
        problem: Problem
            Object that handles the battery cell simulation.

        Returns
        -------
        dict
            Returns a dictionary containing solvers information
        """
        solvers_info = {solver: {'state_variables': [], 'options': {}}
                        for solver in ['solver', 'solver_transitory']}  # , 'solver_stationary']}
        self._model_iterator('get_solvers_info', return_action=None, args=(solvers_info, problem))
        return solvers_info

    def build_weak_formulation(self, solvers_info: dict, var: ProblemVariables, cell: BatteryCell,
                               mesher: BaseMesher, DT: TimeScheme, W: BlockFunctionSpace, problem):
        """
        This method iterates over the active models to build the weak
        formulation of the problem.

        Parameters
        ----------
        solvers_info: dict
            Dictionary containing solvers information
        var: ProblemVariables
            Object that store the preprocessed problem variables.
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
        mesher: BaseMesher
            Object that store the mesh information.
        DT: TimeScheme
            Object that provide the temporal derivatives with the
            specified scheme.
        W: BlockFunctionSpace
            Object that store the function space of each state variable.
        problem: Problem
            Object that handles the battery cell simulation.

        Returns
        -------
        ProblemEquations
            Object containing the integral forms of each equation, as
            well as the dirichlet boundary conditions, that corresponds
            to each state variable.
        """
        equations = ProblemEquations(solvers_info['solver']['state_variables'])
        self._model_iterator('build_weak_formulation', return_action=None,
                             args=(equations, var, cell, mesher, DT, W, problem))
        equations.check()
        return equations

    def build_weak_formulation_transitory(
        self, solvers_info, var: ProblemVariables, cell: BatteryCell,
        mesher: BaseMesher, W: BlockFunctionSpace, problem
    ):
        """
        This method iterates over the active models to build the weak
        formulation of the transitory problem.

        Parameters
        ----------
        solvers_info: dict
            Dictionary containing solvers information
        var: ProblemVariables
            Object that store the preprocessed problem variables.
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
        mesher: BaseMesher
            Object that store the mesh information.
        W: BlockFunctionSpace
            Object that store the function space of each state variable.
        problem: Problem
            Object that handles the battery cell simulation.

        Returns
        -------
        ProblemEquations
            Object containing the integral forms of each equation, as
            well as the dirichlet boundary conditions, that corresponds
            to each state variable.
        """
        equations = ProblemEquations(solvers_info['solver_transitory']['state_variables'])
        self._model_iterator('build_weak_formulation_transitory', return_action=None,
                             args=(equations, var, cell, mesher, W, problem))
        equations.check()
        return equations

    def build_weak_formulation_stationary(
        self, solvers_info: dict, var: ProblemVariables, cell: BatteryCell,
        mesher: BaseMesher, W: BlockFunctionSpace, problem
    ):
        """
        This method iterates over the active models to build the weak
        formulation of the stationary problem.

        Parameters
        ----------
        solvers_info: dict
            Dictionary containing solvers information
        var: ProblemVariables
            Object that store the preprocessed problem variables.
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
        mesher: BaseMesher
            Object that store the mesh information.
        W: BlockFunctionSpace
            Object that store the function space of each state variable.
        problem: Problem
            Object that handles the battery cell simulation.

        Returns
        -------
        ProblemEquations
            Object containing the integral forms of each equation, as
            well as the dirichlet boundary conditions, that corresponds
            to each state variable.
        """
        equations = ProblemEquations(solvers_info['solver_stationary']['state_variables'])
        self._model_iterator('build_weak_formulation_stationary', return_action=None,
                             args=(equations, var, cell, mesher, W, problem))
        equations.check()
        return equations

    def setup(self, problem):
        """
        This method iterates over the active models to setup the models
        if needed.

        Parameters
        ----------
        problem: Problem
            Object that handles the battery cell simulation.
        """
        return self._model_iterator('setup', return_action=None, args=(problem,))

    def update_control_variables(self, var: ProblemVariables, problem, **kwargs) -> None:
        """
        This method iterates over the active models to update the
        control variables of the problem.

        Parameters
        ----------
        var: ProblemVariables
            Object that store the preprocessed problem variables.
        problem: Problem
            Object that handles the battery cell simulation.
        kwargs: dict
            Dictionary containing the control variables.
        """
        valid_kwargs = set()
        for model in self:
            # Get valid kwargs for this model
            sig = inspect.signature(model.update_control_variables)
            model_kwargs = [key for key in list(sig.parameters.keys())[2:]]
            valid_kwargs.update(model_kwargs)

            # Set model cell state variables
            model.update_control_variables(
                var, problem, **{k: v for k, v in kwargs.items() if k in model_kwargs})

        invalid_kwargs = set(kwargs.keys()).difference(valid_kwargs)
        if invalid_kwargs:
            raise TypeError(f"ModelHandler.update_control_variables got unexpected keyword "
                            + "arguments: '" + "' '".join(invalid_kwargs) + "'")

    def update_reference_values(self, updated_values: dict,
                                cell: CellParser, problem=None) -> None:
        """
        This method iterates over the active models to update the
        reference cell properties.

        Parameters
        ----------
        updated_values: Dict[str, float]
            Dictionary containing the cell parameters that have already
            been updated.
        cell: CellParser
            Parser of the cell dictionary.
        problem: Problem, optional
            Object that handles the battery cell simulation.
        Notes
        -----
        This method is called each time a set of dynamic parameters have
        been updated. If problem is not given, then it is assumed that
        it have not been already defined.
        """
        if updated_values:
            self._model_iterator('update_reference_values', return_action=None,
                                 args=(updated_values, cell), kwargs={'problem': problem})

    def reset(self, problem, new_parameters=None, deep_reset=False) -> None:
        """
        This method iterates over the active models to reset the problem
        variables in order to be ready for running another simulation
        with the same initial conditions, and maybe using different
        parameters.

        Parameters
        ----------
        problem: Problem
            Object that handles the battery cell simulation.
        new_parameters: Dict[str, float], optional
            Dictionary containing the cell parameters that have already
            been updated.
        deep_reset: bool, optional
            Whether or not a deep reset will be performed. It means
            that the Problem setup stage will be run again as the mesh
            has been changed. Default to False.
        """
        self._model_iterator('reset', return_action=None, args=(problem,),
                             kwargs={'new_parameters': new_parameters, 'deep_reset': deep_reset})

    # ******************************************************************************************* #
    # ***                             Postprocessing and Outputs                              *** #
    # ******************************************************************************************* #

    def explicit_update(self, problem) -> None:
        """
        This method iterates over the active models to update the
        active explicit models once the implicit timestep is performed.

        Parameters
        ----------
        problem: Problem
            Object that handles the battery cell simulation.
        """
        self._model_iterator('explicit_update', return_action=None, args=(problem,))

    def get_outputs_info(self, warehouse: Warehouse) -> None:
        """
        This method iterates over the active models to return a
        dictionary containing the information of both the global and
        internal variables that can be outputed by the models.

        Parameters
        ----------
        warehouse: Warehouse
            Object that postprocess, store and write the outputs.
        """
        self._model_iterator('get_outputs_info', return_action=None, args=(warehouse,))

    def prepare_outputs(self, warehouse: Warehouse, var: ProblemVariables, cell: BatteryCell,
                        mesher: BaseMesher, DA: DimensionalAnalysis, problem) -> None:
        """
        This method iterates over the active models to compute the
        expression of the requested internal variables to be ready for
        being evaluated and stored.

        Parameters
        ----------
        warehouse: Warehouse
            Object that postprocess, store and write the outputs.
        var: ProblemVariables
            Object containing the problem variables.
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
        mesher: BaseMesher
            Object that store the mesh information.
        DA: DimensionalAnalysis
            Object where the dimensional analysis is performed.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        self._model_iterator('prepare_outputs', return_action=None,
                             args=(warehouse, var, cell, mesher, DA, problem))

    def get_cell_state(self, problem) -> OrderedDict:
        """
        This method iterates over the active models to get the current
        state of the cell.

        Parameters
        ----------
        problem: Problem
            Object that handles the battery cell simulation.
        kwargs: dict
            Dictionary containing the parameters that describe the cell
            state. To know more type :meth:`cideMOD.info(
            'get_cell_state', model_options=model_options)`
        """
        cell_state = OrderedDict(time=problem.time)
        self._model_iterator('get_cell_state', return_action=None, args=(cell_state, problem))
        return cell_state
