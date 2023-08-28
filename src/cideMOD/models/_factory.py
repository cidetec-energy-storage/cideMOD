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
from typing import Union, Optional, overload

from cideMOD.models import BaseCellModel, __models__, __mtypes__, __model_options__

# *********************************************************************************************** #
# ***                                     Factory Methods                                     *** #
# *********************************************************************************************** #


@overload
def model_factory(name: str) -> BaseCellModel:
    ...


@overload
def model_factory(name: None) -> list:
    ...


def model_factory(name: Optional[str] = None) -> Union[BaseCellModel, list]:
    """
    Method that returns the specified model. If no model name is given,
    then returns a list of available models. If a model type is given,
    then returns the list of models of this type.

    Parameters
    ----------
    name : Optional[str]
        Name of the specified model.

    Returns
    -------
    Union[BaseCellModel, list]
        class that implements the specified model. If no model name is
        given, then returns a list of available models. If a model type
        is given, then returns the list of models of this type.

    Raises
    ------
    ValueError("Unrecognized model class")
    """
    if name is None:
        return list(__models__.keys())
    elif name in __models__.keys():
        return __models__[name]

    mtypes = get_model_types()
    if name in mtypes:
        mtype = get_model_types(name)
        return [model for model in __models__.values() if model._mtype_ == mtype]
    else:
        raise ValueError(f"Unrecognized model class '{name}'. Available options: '"
                         + "' '".join(__models__.keys()) + "'")


@overload
def get_model_types(model_type: str) -> str:
    ...


@overload
def get_model_types(model_type: None) -> list:
    ...


def get_model_types(model_type: Optional[str] = None) -> Union[str, list]:
    """
    Method that returns the specified model type. If no model type is
    given, then returns a list of available model types.
    """
    if model_type is not None:
        # Return the mtype, search for aliases
        mtypes = get_model_types()
        if model_type not in mtypes:
            raise ValueError(f"Unrecognized mtype '{model_type}'. Available options: '"
                             + "' '".join(mtypes) + "'")
        elif model_type in __mtypes__.keys():
            return model_type
        else:
            for mtype, aliases in __mtypes__.items():
                if model_type in aliases:
                    return mtype
    else:
        # Return a list of available mtypes
        mtypes = list()
        for mtype, aliases in __mtypes__.items():
            mtypes.extend([mtype, *aliases])
        return mtypes
