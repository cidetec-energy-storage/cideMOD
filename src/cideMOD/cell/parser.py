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

import json
import os

from collections import OrderedDict
from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, Dict
from typing_extensions import Self

from cideMOD.numerics.fem_handler import isinstance_dolfinx, _evaluate_parameter
from cideMOD.cell._factory import get_cell_component_parser
from cideMOD.cell.parameters import CellParameter

# FIXME: In python version greater or equal 3.11, Self is implemented in typing


class BaseComponentParser(ABC):
    """
    Base class for component parser creation.

    Parameters
    ----------
    root: Optional[BaseComponentParser]
        Component to which it belongs. If it does not the case, then it
        should be None.
    dic: dict
        Dictionary containing the parameters of the component.
    tag: str
        Tag to identify the component between the available tags.
    """

    @classmethod
    @property
    @abstractmethod
    def _name_(cls):
        """Name of the component"""
        # raise NotImplementedError

    @classmethod
    @property
    def _root_name_(cls):
        """
        Name of the root component. It should be None if there is no
        root component. This property could be also a list of strings.
        """
        return None

    # NOTE: This attribute will be replaced during the registration process
    _allowed_components_: list = []

    @classmethod
    @property
    def _tags_(cls):
        """
        Available tags for this component. Override this property to
        define the available tags. If there are an undetermined number
        of tags, then this attribute should be None.

        Examples
        --------
        >>> _tags_ = {
                'anode': {
                    'label': 'a',
                    'dict_entry': 'negative_electrode'
                }
                'cathode': {
                    'label': 'c',
                    'dict_entry': 'positive_electrode'
                }
            }

        Notes
        -----
        Every component type must be unique, not only the tags but also
        the label and dictionary entries.
        """
        return None

    _is_recursive_ = False

    @classmethod
    @property
    def name(cls):
        return cls._name_

    @property
    def tag(self):
        return self._tag_

    @property
    def label(self):
        return self._label_

    @property
    def dic(self):
        return self._dic_

    @property
    def complete_tag(self):
        if self._root_component_ is not None:
            return f"{self._root_component_.complete_tag}.{self.tag}"
        else:
            return self.tag

    def __init__(self, root, dic={}, tag=None):
        # Parse tag
        # NOTE: In this cases there should be only one instance of the component
        if self._tags_ is None:
            tag = tag or self._name_
        elif tag is None:
            if len(self._tags_.keys()) > 1:
                raise ValueError(f"A tag must be provided to create the component '{self._name_}'")
            else:
                tag = list(self._tags_.keys())[0]
        # NOTE: In this cases there could be more than one instance of the component
        elif tag not in self._tags_.keys():
            if not self._is_recursive_:
                raise ValueError(f"Unrecognized tag '{tag}' for the component '{self._name_}'. "
                                 + "Available options '" + "' '".join(self._tags_.keys()) + "'")

            # The component is recursive (e.g. am0, am1, am2_inc0)
            tag_info = None
            for available_tag in self._tags_.keys():
                if tag.startswith(available_tag):
                    tag_info = self._tags_[available_tag]
                    self._ref_tag = available_tag
                    break

            if tag_info is None:
                raise ValueError(
                    f"Unrecognized tag '{tag}' for the recursive component '{self._name_}'. "
                    + "Available options '" + "' '".join(self._tags_.keys()) + "'")

        self._tag_ = tag
        self._root_component_ = root
        self._components_: Dict[str, Self] = OrderedDict()
        self._parameters_: Dict[str, CellParameter] = OrderedDict()
        self._dynamic_parameters_: Dict[str, CellParameter] = OrderedDict()
        self._alias2parameter_ = dict()
        if self._tags_ is None:
            # In this case the label and dict_entry is not specified
            self._label_ = tag
            self._dict_entry_ = None
        elif self._is_recursive_:
            # In this case there could be more than one instance of the component
            self._label_ = tag_info['label'].format(tag=tag)
            self._dict_entry_ = None
        elif isinstance(self._tags_[tag]['dict_entry'], list):
            self._label_ = self._tags_[tag]['label']
            self._dict_entry_ = next(
                iter([key for key in self._tags_[tag]['dict_entry'] if key in dic]),
                None)
        else:
            self._label_ = self._tags_[tag]['label']
            self._dict_entry_ = self._tags_[tag]['dict_entry']

        # NOTE: A None dict_entry means that the component dictionary is directly provided
        if self._dict_entry_ is not None and self._dict_entry_ not in dic:
            raise ValueError(f"Dictionary entry '{self._dict_entry_}' of the component "
                             + f"{self._name_} is missing")
        self._dic_ = dic[self._dict_entry_] if self._dict_entry_ is not None else dic

        # Give access to the root cell object to each created component
        self.cell = root.cell if root is not None and root._name_ != 'cell' else root

        # Initialize some component attributes
        self.type = None
        self.N = None

    def _setup(self, models):
        """
        This method sets up both itself and the components it contains.
        """
        # Setup this component
        models.parse_component_parameters(self)
        # Setup each component
        for component in self._components_.values():
            component._setup(models)

    def _setup_dynamic_parameters(self, mesh):
        """
        This method set the mesh associated to the dynamic parameters.
        """
        for parameter in self._dynamic_parameters_.values():
            parameter.setup_dynamic_parameter(mesh)
        for component in self._components_.values():
            component._setup_dynamic_parameters(mesh)

    def set_component(self, component: Union[str, Self]) -> Self:
        """
        This method initialize and sets the given component as a component of this
        component.
        """

        # Initialize the new component
        if isinstance(component, str):
            parser_cls: BaseComponentParser = get_cell_component_parser(component)
            component = parser_cls(self, self.dic, component)

        # Check compatibility between the new component and this component
        tag = component._tag_
        if tag not in self._allowed_components_:
            if not component._is_recursive_ or component._ref_tag not in self._allowed_components_:
                raise ValueError(
                    f"Unable to set the component '{component.name}' to '{self._name_}': "
                    + "Allowed components: '" + "' '".join(self._allowed_components_) + "'")

        # Set the new component to this component
        if tag in self._components_.keys():
            raise RuntimeError(f"Component '{tag}' already added to component '{self._name_}'")
        else:
            self._components_[tag] = component
        return component

    def set_parameters(self, options_dict: dict):
        for param, options in options_dict.items():
            if param == 'required_parameters':
                for param_name in options_dict['required_parameters']:
                    param_req = self.get_parameter(param_name)
                    param_req.make_mandatory()
                continue
            setattr(self, param, self.set_parameter(param, **options))

    def set_parameter(self, name: str, element: Optional[str] = None, default=None,
                      dtypes: Optional[Union[Tuple[str], str]] = 'real',
                      is_optional=False, can_vary: bool = True, can_effective: bool = False,
                      can_arrhenius: bool = False, can_ref_temperature: bool = False,
                      can_hysteresis: bool = False, is_user_input: bool = True,
                      description: Optional[str] = None, aliases: Union[list, tuple, str] = []
                      ) -> CellParameter:
        """
        This method sets the given parameter to this component.

        Parameters
        ----------
        name: str
            Name of the cell parameter.
        element: Optional[str]
            Dictionary entry of the element where the parameter is
            defined. Assume that the parameter is defined within
            the component dictionary if the element is not specified.
        default: Any
            Default value of the parameter if it is optional and the
            user does not specified it.
        dtypes: Optional[Union[Tuple[str], str]]
            Available datatypes that are allowed. It must be one of
            `real`, `expression`, `spline` or `label`.
        is_optional: bool
            Whether this parameter is optional or not. Default to True.
        can_vary: bool
            Whether this parameter can be defined as a dynamic
            parameter. Default to True.
        can_effective: bool
            Whether this parameter can be defined as an effective
            parameter. Default to False.
        can_arrhenius: bool
            Whether the arrhenius correction can be applied to this
            parameter. Default to False.
        can_ref_temperature: bool
            Whether this parameter can have a reference temperature.
        can_hysteresis: bool
            Whether this parameter can have hysteresis.
        description: Optional[str]
            Description of the parameter.
        aliases: Union[list, tuple]
            List or tuple containing the aliases of the parameter if
            any.
        is_user_input: bool
            Whether or not this parameter is a user input.
        """
        # Parse inputs
        if isinstance(dtypes, str):
            dtypes = (dtypes,)
        if isinstance(aliases, str):
            aliases = [aliases]
        parameter_names = [name, *aliases]

        # Check if the parameter is already registered
        for parameter_name in parameter_names:
            if parameter_name in self._parameters_.keys():
                raise ValueError(f"Parameter '{parameter_name}' already registered "
                                 + f"in the component '{self.complete_tag}'")
            elif parameter_name in self._alias2parameter_.keys():
                raise ValueError(f"Parameter '{parameter_name}' already registered as an alias of "
                                 + f"'{self._alias2parameter_[parameter_name]}' in the component "
                                 + f"'{self.complete_tag}'")

        # Get the dictionary where the parameter is defined
        component_dic = self.dic if element is None else self.dic[element]

        parameter_key = next(
            iter([key for key in parameter_names if key in component_dic.keys()]),
            None)

        # Get the dictionary that contains the parameter information
        if not is_user_input:
            parameter_dic = None
        elif parameter_key is not None:
            parameter_dic = component_dic[parameter_key]
        elif not is_optional:
            raise KeyError(f"Error setting the parameter '{self.complete_tag}.{name}': "
                           + "It has not been provided by the user.")
        elif default is None:
            parameter_dic = None
        elif isinstance(default, str) and 'label' in dtypes:
            parameter_dic = default  # label
        elif not isinstance(default, dict) or 'discharge' in default.keys():
            parameter_dic = {'value': default, 'type': 'real'}
        else:
            parameter_dic = default

        data_path = None
        if 'spline' in dtypes:
            data_path = self.cell.data_path if self.cell is not None else self.data_path

        # Intialize the component parameter
        parameter = CellParameter(
            self, parameter_dic, name, dtypes=dtypes, is_optional=is_optional, can_vary=can_vary,
            can_effective=can_effective, can_arrhenius=can_arrhenius,
            can_ref_temperature=can_ref_temperature, can_hysteresis=can_hysteresis,
            data_path=data_path, description=description, aliases=aliases,
            is_user_input=is_user_input
        )

        # Register the component parameter
        self._parameters_[name] = parameter
        if parameter.is_dynamic_parameter:
            self._dynamic_parameters_[name] = parameter
        for alias in aliases:
            self._alias2parameter_[alias] = name

        return parameter

    def get_component(self, name) -> Self:
        """This method return the specified component"""
        if '.' in name:
            name = name[name.rfind('.') + 1:]
        if name not in self._components_.keys():
            raise ValueError(f"Unrecognized component '{name}' of {self.complete_tag}. "
                             + "Available options: '" + "' '".join(self._components_.keys()) + "'")
        return self._components_[name]

    def get_parameter(self, name) -> CellParameter:
        """This method return the specified parameter"""
        if '.' not in name:
            # Look for the parameter
            if name in self._parameters_.keys():
                return self._parameters_[name]
            elif name in self._alias2parameter_.keys():
                return self._parameters_[self._alias2parameter_[name]]
            else:
                raise KeyError(
                    f"Unrecognized parameter '{name}' of the component '{self.complete_tag}'. "
                    + "Available Options: '" + "' '".join(self._parameters_.keys()) + "'")
        elif name.startswith(f"{self.tag}."):
            return self.get_parameter(name[name.find('.') + 1:])
        else:
            # Get the parameter of a subcomponent
            # - Get subcomponent tag
            subcomponent_tag = name[:name.find('.')]
            # - Update the dynamic parameter
            if subcomponent_tag not in self._components_.keys():
                raise KeyError(f"Unable to get the parameter '{name}': component "
                               + f"'{self.complete_tag}' does not have '{subcomponent_tag}'")
            return self._components_[subcomponent_tag].get_parameter(name[name.find('.') + 1:])

    def get_value(self, name):
        """
        This method return the user value of the specified parameter
        """
        parameter = self.get_parameter(name)
        return parameter.user_value

    def _reset(self):
        """
        This method clear the preprocessed parameters in order to
        recompute them again with new parameters or problem.
        """
        for parameter in self._parameters_.values():
            parameter.reset()
        for component in self._components_.values():
            component._reset()


class ElectrodeParser(BaseComponentParser):

    _name_ = 'electrode'
    _root_name_ = 'cell'
    _tags_ = {
        'anode': {
            'label': 'a',
            'dict_entry': ['negative_electrode', 'negativeElectrode']
        },
        'cathode': {
            'label': 'c',
            'dict_entry': ['positive_electrode', 'positiveElectrode']
        }
    }

    def __init__(self, cell, dic, tag=None):
        super().__init__(cell, dic, tag=tag)
        self._set_active_materials()

    def _set_active_materials(self):
        # Get the list of active materials from the dictionary
        active_materials = self.dic.get('active_materials', [])
        if isinstance(active_materials, dict):
            active_materials = [active_materials]

        if not active_materials:
            raise ValueError(
                f"The active materials of the cell component '{self.name}' have not been provided")

        # Set each active material
        self.active_materials = []
        for am_idx, am_dic in enumerate(active_materials):
            am = self.ActiveMaterialParser(self, am_dic, am_idx, tag=f"am{am_idx}")
            self.set_component(am)
            self.active_materials.append(am)
        self.n_mat = len(self.active_materials)

    class ActiveMaterialParser(BaseComponentParser):

        _name_ = 'active_material'
        _root_name_ = ('electrode', 'active_material')
        _is_recursive_ = True
        _tags_ = {'am': {'label': '{tag}'}}

        @property
        def complete_tag(self):
            return f"cell.{self.electrode_tag}.{self.tag}"

        @property
        def electrode_tag(self):
            if self._root_component_.name == 'electrode':
                return self._root_component_.tag
            else:
                return self._root_component_.electrode_tag

        def __init__(self, electrode, dic: dict, index: int, tag=None):
            super().__init__(electrode, dic, tag=tag)
            self.electrode = electrode
            self.data_path = self.cell.data_path
            self.index = index

            self._set_inclusions()

        def _set_inclusions(self):
            # Get the list of inclusions from the dictionary
            inclusions = self.dic.get('inclusions', [])
            if isinstance(inclusions, dict):
                inclusions = [inclusions]

            # Set each inclusion
            self.inclusions = []
            for inc_idx, inc_dic in enumerate(inclusions):
                inc = self.__class__(self, inc_dic, inc_idx, tag=f"{self.tag}_inc{inc_idx}")
                self.set_component(inc)
                self.inclusions.append(inc)
            self.n_inc = len(self.inclusions)


class CellParser(BaseComponentParser):
    """
    Parser of cell dictionary/JSON into a python object. Resolves any
    links between different files.

    Parameters
    ----------
    cell_data : Union[dict,str]
        Dictionary of the cell parameters or path to a JSON file
        containing them.
    data_path : str
        Path to the folder where *cell_data* is together with extra data
        like materials OCVs.
    model_options: BaseModelOptions
        Object containing the simulation options.
    """

    _name_ = 'cell'

    def __init__(self, cell_data: Union[dict, str], data_path: str, model_options):
        self.data_path = data_path
        self._models = model_options._get_model_handler()
        self.verbose = model_options.verbose
        self._comm = model_options.comm
        self.problem = None

        # Parse cell_data
        if isinstance(cell_data, dict):
            dic = cell_data
        elif isinstance(cell_data, str):
            path = os.path.join(data_path, cell_data)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path {path} does not exists")
            dic = self.read_param_file(path)
        else:
            raise TypeError("cell_data argument must be dict or str")

        self.structure = dic['structure']

        # Initialize cell component
        super().__init__(None, dic)

        # Build cell components
        self._models.build_cell_components(self)

        # Pase cell structure
        self._models.parse_cell_structure(self)

        # Count the number of components inside the cell structure
        for component in self._components_.values():
            component.N = self.structure.count(component.label)

        # Parse cell parameters
        self._setup(self._models)

        # Compute reference cell properties
        self._models.compute_reference_cell_properties(self)

    def read_param_file(self, path):
        with open(path, 'r') as fin:
            dic = json.load(fin)
        return dic

    def write_param_file(self, path):
        with open(path, 'w') as fout:
            json.dump(self.dic, fout)

    def _set_problem(self, problem):
        if self.problem is not None:
            self._reset()
        self._setup_dynamic_parameters(problem.mesher.mesh)
        self.problem = problem

    def update_parameter(self, name: str, value) -> None:
        """
        This method updates the value of the specified parameter of this
        component or the components it has.

        Parameters
        ----------
        name: str
            Name of the parameter to be updated.
        value: Any
            Value of the parameter to be updated.

        Examples
        --------
        To update a parameter of this component:

        >>> cell.update_parameter('area', 0.1)

        or

        >>> cell.update_parameter('cell.area', 0.1)

        To update a parameter of a subcomponent:

        >>> cell.update_parameter('anode.thickness', 1e-4)
        """
        parameter = self.get_parameter(name)

        if self.problem is None:
            parameter.reset()  # NOTE: Just to ensure parameter.ref_value is None
            parameter.set_value(value)
        elif parameter.is_dynamic_parameter:
            parameter.update(value)
        else:
            raise RuntimeError(parameter._get_error_msg(
                reason=("Unable to set a new value of a non dynamic parameter: "
                        + "problem has already been defined"),
                action="handling"
            ))
        self._models.update_reference_values({str(parameter): value}, self, problem=self.problem)

    def update_parameters(self, parameters: dict) -> None:
        """
        This method updates the values of the parameters of this
        component and the components it has.

        Parameters
        ----------
        parameters: dict
            Dictionary containing the parameter names and values
            to be updated.

        Examples
        --------
        To update a parameter of this component:

        >>> cell.update_parameters({'area': 0.1})

        or

        >>> cell.update_parameters({'cell.area': 0.1})

        To update a parameter of a subcomponent:

        >>> cell.update_parameters({'anode.thickness': 1e-4})
        """
        for name, value in parameters.items():
            parameter = self.get_parameter(name)
            if self.problem is None:
                parameter.reset()  # NOTE: Just to ensure parameter.ref_value is None
                parameter.set_value(value)
            elif parameter.is_dynamic_parameter:
                parameter.update(value)
            else:
                raise RuntimeError(parameter._get_error_msg(
                    reason=("Unable to set a new value of a non dynamic parameter: "
                            + "problem has already been defined"),
                    action="handling"
                ))

        # NOTE: This method updates the current value and reset the reference values. Thats why the
        #       recomputation of reference dynamic parameters is needed.
        self._models.update_reference_values(parameters, self, problem=self.problem)

    def get_dict(self, base=None):
        if base is None:
            base = self.dic
        json_dict = {}
        for key, value in base.items():
            if isinstance_dolfinx(value):
                try:
                    json_dict[key] = _evaluate_parameter(value)
                except Exception:
                    json_dict[key] = str(value)
            elif isinstance(value, dict):
                json_dict[key] = self.get_dict(base=value)
            else:
                json_dict[key] = value
        return json_dict
