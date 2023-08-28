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
cell_components creates and initializes the corresponding battery components
attributes. This also includes the functions for the weak formulation.
"""
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing_extensions import Self

from cideMOD.helpers.logging import VerbosityLevel, _print
from cideMOD.cell._factory import get_cell_component_class
from cideMOD.cell.parser import BaseComponentParser

# FIXME: In python version greater or equal 3.11, Self is implemented in typing


class BaseCellComponent(ABC):
    """
    Base class for cell component creation.

    Parameters
    ----------
    root: Optional[BaseCellComponent]
        Component to which it belongs. If it does not the case, then it
        should be None.
    config: BaseComponentParser
        Object where the cell component parameters are parsed.
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
                    'label': 'a'
                }
                'cathode': {
                    'label': 'c'
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
    def parser(self) -> BaseComponentParser:
        return self._parser_

    @property
    def complete_tag(self):
        return self._parser_.complete_tag

    def __init__(self, root, parser: BaseComponentParser, tag: str = None,
                 verbose=VerbosityLevel.NO_INFO):
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
        if tag != parser._tag_ or self._name_ != parser._name_:
            raise ValueError(
                f"Component '{self._name_}' must be created with the corresponding parser class")

        self._tag_ = tag
        self._parser_ = parser
        self._root_component_ = root
        self._label_ = parser._label_
        self._components_ = OrderedDict()

        self.cell = root.cell if root is not None and root._name_ != 'cell' else root
        self.verbose = verbose

        self.N = parser.N

        self._set_components()

    def get_component(self, name) -> Self:
        """This method return the specified component"""
        if '.' in name:
            name = name[name.rfind('.') + 1:]
        if name not in self._components_.keys():
            raise ValueError(f"Unrecognized component '{name}' of {self.complete_tag}. "
                             + "Available options: '" + "' '".join(self._components_.keys()) + "'")
        return self._components_[name]

    def _setup(self, models, problem):
        """
        This method sets up both itself and the components it contains.
        """
        if self.verbose >= VerbosityLevel.DETAILED_PROGRESS_INFO:
            _print(f"    Setting up '{self.complete_tag}' parameters", comm=problem._comm)
        # Setup this component
        models.set_component_parameters(self, problem)
        # Setup each component
        for component in self._components_.values():
            component._setup(models, problem)

    def _set_component(self, component):
        """
        This method initialize and sets the given component as a
        component of this component.

        Parameters
        ----------
        component: Union[BaseCellComponent,str]
            Object (or name of the object) where component parameters
            are preprocessed and stored.
        """
        # NOTE: Assume that the BaseCellComponent has been initialized correctly
        # Initialize the new component
        if isinstance(component, str):
            component_cls: BaseCellComponent = get_cell_component_class(component)
            parser = self._parser_._components_[component]
            component = component_cls(self, parser, tag=component, verbose=self.verbose)

        # Set the new component to this component
        tag = component._tag_
        if tag in self._components_.keys():
            raise RuntimeError(f"Component '{tag}' already added to component '{self._name_}'")
        else:
            self._components_[tag] = component
        return component

    def _set_components(self):
        """
        This method initialize and sets the components following the
        parser class components.
        """
        for name in self._parser_._components_.keys():
            component = self._set_component(name)
            setattr(self, component._tag_, component)


class ElectrodeParameters(BaseCellComponent):

    _name_ = 'electrode'
    _root_name_ = 'cell'

    def _set_components(self):
        super()._set_components()

        # Set each active material
        self.active_materials = []
        for am_parser in self.parser.active_materials:
            am = self._components_[am_parser._tag_]
            self.active_materials.append(am)
        self.n_mat = len(self.active_materials)

    class ActiveMaterialParameters(BaseCellComponent):

        _name_ = 'active_material'
        _root_name_ = ('electrode', 'active_material')

        @property
        def electrode_tag(self):
            return self._parser_.electrode_tag

        def __init__(self, root, parser, tag=None, verbose=VerbosityLevel.NO_INFO):
            super().__init__(root, parser, tag, verbose)
            self.electrode = root  # NOTE: It could be electrode or active_material
            self.index = parser.index

        def _set_components(self):
            super()._set_components()

            # Set each active material inclusion
            self.inclusions = []
            for inc_parser in self.parser.inclusions:
                inc = self._components_[inc_parser._tag_]
                self.inclusions.append(inc)
            self.n_inc = len(self.inclusions)


class BatteryCell(BaseCellComponent):
    """
    Class that preprocesses and stores the cell parameters.

    Parameters
    ----------
    problem: Problem
        Object tha handles the battery cell simulation.
    """

    _name_ = 'cell'

    def __init__(self, problem):
        super().__init__(None, problem.cell_parser, verbose=problem.model_options.verbose)

        self.problem = problem
        self._comm = problem._comm
        self.verbose = problem.verbose
        self.structure = problem.cell_parser.structure

        # Set cell parameters
        self._setup(problem._models, problem)

        # Cell properties
        problem._models.compute_cell_properties(self)
