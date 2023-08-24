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
This module provides a wide variety of battery cell models, as well as
the template to develop new ones.
"""

__models__ = dict()

__mtypes__ = dict()

__model_options__ = dict()

__all__ = [
    "ModelHandler",
    "get_model_options",
    "BaseCellModel",
    "model_factory"
]

import os
import pydantic
import importlib

from collections import OrderedDict
from typing import List

from cideMOD.helpers.miscellaneous import generate_class_name
from cideMOD.numerics.triggers import Trigger
from cideMOD.models.base import BaseCellModel
from cideMOD.models._factory import model_factory, get_model_types
from cideMOD.models.model_handler import ModelHandler, CellModelList
from cideMOD.models.model_options import BaseModelOptions, get_model_options


# *********************************************************************************************** #
# ***                                      Registration                                       *** #
# *********************************************************************************************** #


_registration_query = {'models': OrderedDict(), 'options': OrderedDict()}


def register_model_type(mtype: str, aliases: List[str] = []):
    """
    New model types can be added to cideMOD with the
    :func:`register_model_type` method.

    For example::

        register_model_type('PXD', aliases=['P2D', 'P3D', 'P4D'])

    """
    mtypes = get_model_types()
    if mtype in mtypes:
        raise ValueError(f"mtype '{mtype}' is already registered!")
    elif not mtype.isidentifier():
        raise ValueError(f"mtype '{mtype}' is not a valid identifier")
    for alias in aliases:
        if alias in mtypes:
            raise ValueError(f"mtype alias '{alias}' is already registered!")
    __mtypes__[mtype] = aliases


def register_model(cls):
    """
    New cell models can be added to cideMOD with the
    :func:`register_model` function decorator.

    For example::

        @register_model
        class ElectrochemicalModel(BaseCellModel):
            ...

    .. note::

        Notice that all models must implement the
        :class:`BaseCellModel` interface.

    """
    if not issubclass(cls, BaseCellModel):
        raise TypeError(f"Class {cls} is not a subclass of cideMOD.models.BaseCellModel")
    elif cls._name_ in _registration_query['models']:
        raise RuntimeError(f"The model '{cls._name_}' already registered!")
    elif cls._name_ in get_model_types():
        raise ValueError(f"The model '{cls._name_}' has the same name as a model type")
    _registration_query['models'][cls._name_] = cls
    return cls


def register_model_options(model_name):
    """
    New model options can be added to cideMOD with the
    :func:`register_model_options` function decorator.

    For example::

        @register_model_options(__model_name__)
        class ElectroquemicalModelOptions(pydantic.BaseModel)
            ...

    """

    def wrapper(cls):
        if not issubclass(cls, pydantic.BaseModel):
            raise TypeError(f"Class {cls} is not a subclass of pydantic.BaseModel")
        if model_name in _registration_query['options']:
            raise ValueError(f"model_name '{model_name}' already used")
        _registration_query['options'][model_name] = cls
        return cls

    return wrapper


def _register_model(cls: BaseCellModel):
    # NOTE: Once mtypes are set, we can register the cell models
    if cls._mtype_ not in __mtypes__.keys():
        raise ValueError(f"Unrecognized mtype '{cls._mtype_}'. Available options '"
                         + "' '".join(__mtypes__.keys()) + "'")
    __models__[cls._name_] = cls


def _register_model_options(mtype, cls):
    # NOTE: The model option extension should be performed following the models hierarchy
    if mtype not in __mtypes__.keys():
        raise ValueError(f"Unrecognized mtype '{mtype}'. Available options '"
                         + "' '".join(__mtypes__.keys()) + "'")
    if mtype not in __model_options__:
        # Intilize the model options of this model type
        cls_name = generate_class_name(mtype, prefix='ModelOptions')
        cls_namespace = {'__doc__': BaseModelOptions.__doc__, '__module__': 'cideMOD.models'}
        __model_options__[mtype] = type(cls_name, (BaseModelOptions,), cls_namespace)
    __model_options__[mtype]._extend(cls)


def register_trigger(name, label, units, need_abs=False, atol=1e-6):
    """
    Register a new trigger variable.

    Parameters
    ----------
    name: str
        Name of the variable
    label: str
        Label of the variable. It is used as an alias.
    units: str
        Units of the trigger variable.
    need_abs: bool
        Whether or not to check the absolute value of the trigger
        variable. Default to False.
    atol: float
        Default absolute tolerance for the trigger. Default to 1e-6.
    """
    Trigger.register_variable(name, label, units, need_abs=need_abs, atol=atol)

# *********************************************************************************************** #
# ***                                     Model Discovery                                     *** #
# *********************************************************************************************** #

# Filter Model Subpackages. Model folder structure:
# NOTE: This structure is prone to modifications
#
# model_folder/
#  +-- __init__.py
#  +-- inputs.py
#  +-- preprocessing.py
#  +-- equations.py
#  +-- outputs.py
#


def _is_cell_model_folder(folderpath):
    """
    This method checks if the given folder has the cell model structure
    """
    mandatory_model_files = ['__init__.py', 'inputs.py', 'preprocessing.py',
                             'equations.py', 'outputs.py']
    model_name = os.path.basename(folderpath)
    if (not os.path.isdir(folderpath)
            or model_name.startswith('_') or not model_name.isidentifier()):
        return False
    model_files = os.listdir(folderpath)
    return all([file in model_files for file in mandatory_model_files])


def _import_cell_models(path, root='', ignored_dirs=[], max_depth=1):
    """
    This method imports dinamically the cell models that it is
    discovering
    """
    path = os.path.normpath(path)
    # Iterate over the folder tree of the given path
    for dirpath, dirnames, filenames in os.walk(path, topdown=True):
        if '__init__.py' not in filenames:
            # This folder is not a valid python module
            dirnames.clear()
            continue
        depth = dirpath[len(path):].count('/') + 1

        # Remove ignored directories
        for ignored_dir in ignored_dirs:
            if ignored_dir in dirnames:
                dirnames.remove(ignored_dir)

        # Iterate over the folders at this depth level
        for model_name in dirnames.copy():
            model_path = os.path.join(dirpath, model_name)
            if _is_cell_model_folder(model_path):
                # Import cell model
                # NOTE: Cell model is registered once its module is imported
                if depth > 1:
                    modules = dirpath[len(path) + 1:].replace('/', '.')
                    importlib.import_module(f'{root}.{modules}.{model_name}')
                else:
                    importlib.import_module(f'{root}.{model_name}')
                # Do not walk inside the cell model folder
                dirnames.remove(model_name)
            elif model_name.startswith('_') or not model_name.isidentifier():
                # This is not a valid public module name
                dirnames.remove(model_name)
        if depth >= max_depth:
            dirnames.clear()  # Do not walk deeper


def _complete_registration():
    """
    This method complete the registration of model classes once the
    model types and model hierarchy is known.
    """
    # Register models
    for model in _registration_query['models'].values():
        _register_model(model)

    # Register model options following the models hierarchy
    for mtype in __mtypes__:
        models = CellModelList([__models__[k]
                                for k in _registration_query['options'] if
                                __models__[k]._mtype_ == mtype], instances=False)
        for model in models:
            model_options = _registration_query['options'][model._name_]
            _register_model_options(mtype, model_options)

        # TODO: register components. Merge phase 2 magic dictionaries.


_import_cell_models(os.path.dirname(__file__), root='cideMOD.models',
                    ignored_dirs=['base'], max_depth=5)

_complete_registration()

del _registration_query, _import_cell_models, _complete_registration


# *********************************************************************************************** #
# ***                                     User Interface                                      *** #
# *********************************************************************************************** #

def models_info(model_method: str, model_name: str = None, model_options: BaseModelOptions = None,
                not_exist_ok=True):
    """
    This method print the docstring of the specified model method/s.

    Parameters
    ----------
    model_method: str
        Name of the model method to be described.
    model_name: Optional[str]
        Name of the model on which we are interested.
    model_options: BaseModelOptions
        Model options already configured by the user.

    Notes
    -----
    If model_name is specified, then print the information of the model
    method. Otherwise, print the information of the active models if
    model_options is given, else print the information of the method of
    every registered model.
    """
    if model_name is not None:
        model = model_factory(model_name)
        info = getattr(model, model_method).__doc__
    elif model_options is not None:
        info = ""
        models = model_options._get_model_handler()
        for model in models:
            if not_exist_ok and not hasattr(model, model_method):
                continue
            model_fnc = getattr(model.__class__, model_method)
            if model_fnc is getattr(BaseCellModel, model_method):
                continue
            model_info = model_fnc.__doc__
            info += "\n        ".join(['', model._name_, "~" * len(model._name_)]) + model_info
    else:
        raise NotImplementedError("Not implemented yet.")
    print(info)
