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
This module provides cell classes and functions to read/process battery
related information
"""

# __cell_parameters__ = dict() # Phase 2
# __problem_variables__ = dict() # Phase 2
__cell_components__ = dict()

__all__ = [
    "CellParameter",
    "CellParser",
    "BatteryCell",
    "DimensionalAnalysis",
    "Warehouse",
    "ProblemEquations",
    "ProblemVariables"
]

from typing import Optional, Union, List

from cideMOD.helpers.miscellaneous import generate_class_name
from cideMOD.cell.parameters import CellParameter
from cideMOD.cell.parser import CellParser, BaseComponentParser, ElectrodeParser
from cideMOD.cell.components import BatteryCell, BaseCellComponent, ElectrodeParameters
from cideMOD.cell.dimensional_analysis import DimensionalAnalysis
from cideMOD.cell.warehouse import Warehouse
from cideMOD.cell.equations import ProblemEquations
from cideMOD.cell.variables import ProblemVariables


# *********************************************************************************************** #
# ***                                  Registration Methods                                   *** #
# *********************************************************************************************** #

def register_cell_component(name: str,
                            parser_cls: Optional[Union[BaseComponentParser, dict]] = None,
                            component_cls: Optional[Union[BaseCellComponent, dict]] = None,
                            tags: Optional[Union[List[str], str]] = None,
                            root: Optional[str] = None):
    """
    This method allows to register new cell components

    Parameters
    ----------
    name: str
        Name of the component. It must be a valid identifier.
    parser_cls: Optional[Union[BaseComponentParser, dict]]
        Object where component parameters are parsed. If not provided or
        the class dictionary is provided instead, then the class is
        created dinamically.
    component_cls: Optional[Union[BaseCellComponent, dict]]
        Object where component parameters are preprocessed and stored.
        If not provided or the class dictionary is provided instead,
        then the class is created dinamically.
    tags: Optional[Union[List[str], str]]
        Allowed tags for the component. Used to differenciate the
        component instances.
    root: Optional[List[str], str]
        Name of the root component.
    """
    # TODO: Improve the design of this method and classes.
    # Get registered tags info
    info = {'tags': {}, 'labels': {}, 'dict_entries': {}}
    for comp_name, comp_dict in __cell_components__.items():
        comp_parser_cls = comp_dict['parser_cls']
        if comp_parser_cls._tags_ is None:
            for key in info.keys():
                info[key][comp_name] = comp_name
        else:
            for tag, tag_info in comp_parser_cls._tags_.items():
                info['tags'][tag] = comp_name
                if not comp_parser_cls._is_recursive_:
                    # TODO: revise this
                    dict_entry = (
                        tag_info['dict_entry'] if not isinstance(tag_info['dict_entry'], list)
                        else tag_info['dict_entry'][0])
                    info['labels'][tag_info['label']] = comp_name
                    info['dict_entries'][dict_entry] = comp_name

    # Parse component name
    if not name.isidentifier():
        raise ValueError(f"name '{name}' is not a valid identifier")
    elif name in __cell_components__:
        raise ValueError(f"Component '{name}' already registered!")

    # Parse component parser class
    base_namespace = {'_name_': name, '_tags_': tags, '_root_name_': root,
                      '__module__': 'cideMOD.cell'}
    if parser_cls is None or isinstance(parser_cls, dict):
        # Create the parser class
        namespace = base_namespace if parser_cls is None else {**parser_cls, **base_namespace}
        cls_name = generate_class_name(name, suffix='Parser')
        parser_cls = type(cls_name, (BaseComponentParser,), namespace)
    elif parser_cls._name_ != name:
        raise ValueError(f"Incorrect name of the parser class of '{name}'")
    elif root is not None and parser_cls._root_name_ != root:
        raise ValueError(f"Incorrect root name '{root}' of the parser class of '{name}'")
    elif tags is not None and parser_cls._tags_ != tags:
        raise ValueError(f"Incorrect tags of the parser class of '{name}'")

    root = parser_cls._root_name_

    # Ensure uniqueness of tags, labels and dict entries
    if parser_cls._tags_ is not None:
        for tag, tag_info in parser_cls._tags_.items():
            label = tag_info['label']
            entry = tag_info.get('dict_entry', None)
            if tag in info['tags']:
                raise ValueError(
                    f"Component tag '{tag}' already registered by '{info['tags'][tag]}'")
            elif parser_cls._is_recursive_:
                continue
            elif label in info['labels']:
                raise ValueError(
                    f"Component label '{label}' already registered by '{info['labels'][label]}'")
            elif entry and (entry if isinstance(entry, str) else entry[0]) in info['dict_entries']:
                raise ValueError(f"Component dict entry '{entry}' already registered by "
                                 + f"'{info['dict_entries'][entry]}'")

    # Parse component class
    if component_cls is None or isinstance(component_cls, dict):
        # Create the component class
        if tags is None:
            base_namespace['_tags_'] = parser_cls._tags_
            base_namespace['_is_recursive_'] = parser_cls._is_recursive_
        namespace = (base_namespace
                     if component_cls is None else {**component_cls, **base_namespace})
        cls_name = generate_class_name(name, suffix='Parameters')
        component_cls = type(cls_name, (BaseCellComponent,), namespace)
    elif component_cls._name_ != name:
        raise ValueError(f"Incorrect name of the component class of '{name}'")
    elif root is not None and component_cls._root_name_ != root:
        raise ValueError(f"Incorrect root name '{root}' of the parser class of '{name}'")
    else:
        # Make the parser and component classes share some attributes
        component_cls._tags_ = parser_cls._tags_
        component_cls._is_recursive_ = parser_cls._is_recursive_

    # Parse root component/s name/s
    if root is not None:
        root_names = root if isinstance(root, (list, tuple)) else [root]
        for root_name in root_names:
            if root_name not in __cell_components__.keys() and root_name != name:
                raise ValueError(f"Unrecognized cell component '{root_name}'. Available options '"
                                 + "' '".join(__cell_components__.keys()) + "'")

    # Registration
    __cell_components__[name] = {
        'root': root,
        'allowed_components': [],
        'parser_cls': parser_cls,
        'component_cls': component_cls
    }
    parser_cls._allowed_components_ = __cell_components__[name]['allowed_components']
    component_cls._allowed_components_ = __cell_components__[name]['allowed_components']
    if root is not None:
        for root_name in root_names:
            if parser_cls._tags_ is None:
                __cell_components__[root_name]['allowed_components'].append(name)
            else:
                __cell_components__[root_name]['allowed_components'].extend(parser_cls._tags_)


# *********************************************************************************************** #
# ***                                      Registration                                       *** #
# *********************************************************************************************** #
# Generic cell component implementation
register_cell_component('cell', parser_cls=CellParser, component_cls=BatteryCell)

register_cell_component('electrode', parser_cls=ElectrodeParser, component_cls=ElectrodeParameters,
                        root='cell')

register_cell_component('active_material', parser_cls=ElectrodeParser.ActiveMaterialParser,
                        component_cls=ElectrodeParameters.ActiveMaterialParameters)

register_cell_component('separator', root='cell',
                        tags={'separator': {'label': 's', 'dict_entry': 'separator'}})

register_cell_component('current_collector', root='cell',
                        tags={
                            'negativeCC': {
                                'label': 'ncc',
                                'dict_entry': ['negative_current_collector',
                                               'negativeCurrentCollector']
                            },
                            'positiveCC': {
                                'label': 'pcc',
                                'dict_entry': ['positive_current_collector',
                                               'positiveCurrentCollector']
                            }
                        })

register_cell_component('electrolyte', root='cell',
                        tags={'electrolyte': {'label': 'ly', 'dict_entry': 'electrolyte'}})

# NOTE: Each model can add more components if needed within their corresponding modules
