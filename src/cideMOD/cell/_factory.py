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

from cideMOD.cell import __cell_components__

from typing import Optional, Union, overload

# *********************************************************************************************** #
# ***                                     Factory Methods                                     *** #
# *********************************************************************************************** #


@overload
def _get_cell_component_info(component: str, include_tags: bool = False) -> dict:
    ...


@overload
def _get_cell_component_info(component: None, include_tags: bool = False) -> list:
    ...


def _get_cell_component_info(
        component: Optional[str] = None, include_tags: bool = False) -> Union[dict, list]:
    """
    Method that returns the information of the specified cell component.
    If no cell component is given, then returns a list of available
    cell components.

    Parameters
    ----------
    component : Optional[str]
        Name (or tag) of the specified cell component.
    include_tags : bool
        Whether or not to include the component tags in the list of
        available components. Default to False.

    Returns
    -------
    Union[dict, list]
        Dictionary containing the information of the specified cell
        component. If no cell component is given, then returns a list of
        available cell components.

    Raises
    ------
    ValueError("Unrecognized cell component")
    """
    if component is None:
        # Return a list of available cell components
        if not include_tags:
            return list(__cell_components__.keys())
        else:
            components = list()
            for name, info in __cell_components__.items():
                cls = info['parser_cls']
                tags = cls._tags_.keys() if cls._tags_ is not None else []
                components.extend([name, *tags])
            return components
    else:
        # Get the information of this component
        components = _get_cell_component_info(include_tags=True)
        if component not in components:
            # Check if it is a recursive component
            for name, info in __cell_components__.items():
                cls = info['parser_cls']
                if info['parser_cls']._is_recursive_:
                    for available_tag in cls._tags_:
                        if component.startswith(available_tag):
                            return __cell_components__[name]
            raise ValueError(f"Unrecognized cell component '{component}'. Available options: '"
                             + "' '".join(__cell_components__.keys()) + "'")
        elif component in __cell_components__.keys():
            return __cell_components__[component]
        else:
            for name, info in __cell_components__.items():
                cls = info['parser_cls']
                if cls._tags_ is not None and component in cls._tags_.keys():
                    return __cell_components__[name]


def get_cell_component_parser(component: str):
    """
    Method that returns the parser class of the specified component.
    """
    component_info = _get_cell_component_info(component)
    return component_info['parser_cls']


def get_cell_component_class(component: str):
    """
    Method that returns the component class of the specified component.
    """
    component_info = _get_cell_component_info(component)
    return component_info['component_cls']


def get_cell_component_label(component):
    cls = get_cell_component_parser(component)
    if cls._tags_ is None:
        raise ValueError(f"Component 'component' does not have a registered label")
    else:
        return cls._tags_[component]['label']


# TODO: Add here factory methods for cell parameters and problem variables. Phase 2.
