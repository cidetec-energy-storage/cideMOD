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
import os
from mpi4py import MPI
from typing import Optional, Union
from pydantic import BaseModel, BaseConfig, PrivateAttr, validator

from cideMOD.helpers.logging import VerbosityLevel
from cideMOD.helpers.miscellaneous import init_results_folder
from cideMOD.models import ModelHandler, get_model_types, __mtypes__, __model_options__


class BaseModelOptions(BaseModel):
    """
    Settings for the cideMOD's cell model simulation.

    General Parameters
    ------------------
    model: str
        Simulation mode, default "P2D"
    dimensionless: bool
        Whether to use the dimensionless version or not. Default to
        False
    solve_LAM: bool
        Whether to solve LAM problem or not. Default to False
    N_x: int
        Discretization in x direction. Default to 30
    N_y: int
        Discretization in y direction. Default to 10
    N_z: int
        Discretization in z direction. Default to 10
    FEM_order: int
        Order of interpolating finite elements. Default to 1
    time_scheme: str
        Time discretization scheme, default "euler_implicit"
    raise_errors_on_exit: bool
        Whether to raise the SolverCrashed error on exit if it happens.
        Default to True.
    clean_on_exit: bool
        Whether to clean from memory saved data at the end of the solve
        cycle. Default to True.
    save_on_exit: bool
        Whether to save global variables on exit. They will be saved
        always before cleaning. Default to True.
    globals_txts: bool
        Whether to save global variables on individual .txt files or
        just as a single 'condensated.txt' file. Default to True.
    comm : MPI.Intracomm, optional
        MPI Communicator for running tests in parallel. Default to
        MPI.COMM_WORLD.
    overwrite: bool, optional
        Whether or not to override existing data (if so). Default to
        False.
    save_path: str, optional
        Path to the folder outputs. If it does not exist, create it.
        Otherwise it will check `overwrite` to override the existing
        data or change the given save_path
    verbose: int
        Verbosity level. Defaults to VerbosityLevel.BASIC_PROBLEM_INFO.
        For more information type `help(cideMOD.VerbosityLevel)`
    """
    model: str = "P2D"
    dimensionless: bool = False
    solve_LAM: bool = False
    N_x: Union[int, list] = 30
    N_y: int = 10
    N_z: int = 10
    FEM_order: int = 1
    raise_errors_on_exit: bool = True
    clean_on_exit: bool = True
    save_on_exit: bool = True
    globals_txts: bool = True
    comm: object = None  # NOTE: MPI.Intracomm is not pickable (a deepcopy is performed)
    overwrite: bool = False
    save_path: Optional[str] = None
    mode: str = 'ERROR'
    verbose: int = VerbosityLevel.BASIC_PROBLEM_INFO
    _model_handler = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._update_save_path(self.save_path)
        self._model_handler = ModelHandler(self)

    def _update_save_path(self, save_path, copy_files=[], filenames=[], prefix='results_'):
        if save_path is not None:
            save_path = init_results_folder(
                save_path, overwrite=self.overwrite, comm=self.comm, copy_files=copy_files,
                filenames=filenames, verbose=self.verbose >= VerbosityLevel.BASIC_PROBLEM_INFO,
                prefix=prefix)
        self.save_path = save_path
        return save_path

    @validator('model')
    def validate_model(cls, v):
        mtypes = get_model_types()
        if v not in mtypes:
            raise ValueError("'model' keyword must be one of: '" + "' '".join(mtypes) + "'")
        return v

    @validator('mode')
    def validate_mode(cls, v):
        allowed_modes = ('DEBUG', 'WARNING', 'ERROR')
        if v not in allowed_modes:
            raise ValueError("'model' keyword must be one of: '" + "' '".join(allowed_modes) + "'")
        return v

    @validator("FEM_order")
    def validate_FEM_order(cls, v):
        if v != 1:
            raise NotImplementedError("Only FEM_order = 1 is implemented")
        return v

    @validator("verbose")
    def validate_verbose(cls, v):
        if v < VerbosityLevel.NO_INFO:
            return VerbosityLevel.NO_INFO
        elif v > VerbosityLevel.DETAILED_SOLVER_INFO:
            return VerbosityLevel.DETAILED_SOLVER_INFO
        else:
            return v

    @validator("comm", always=True)
    def validate_comm(cls, v):
        if v is None:
            return MPI.COMM_WORLD
        elif not isinstance(v, MPI.Intracomm):
            raise TypeError('comm is not a valid MPI.Intracomm')
        else:
            return v

    @validator('solve_LAM')
    def validate_solve_LAM(cls, v):
        if v:
            raise NotImplementedError("The LAM model is not available yet")
        return v

    @classmethod
    def _extend(cls, new_options: BaseModel):
        # Parse the input
        if not issubclass(new_options, BaseModel):
            raise TypeError("'new_options' must be a subclass of pydantic.BaseModel")
        ignored_keys = ['prepare_field', 'get_field_info']
        config, new_config = cls.Config, new_options.Config
        if new_config is not BaseConfig and not all(
            [getattr(config, k) == getattr(new_config, k) for k in dir(new_config)
             if not k.startswith('_') and k not in ignored_keys]):
            raise NotImplementedError("'new_options' must have the same config options")
            # TODO: Merge config options and validate the extended model or just ignore them
        if new_options.__private_attributes__:
            raise NotImplementedError("Pending to extend a pydantic model with private attributes")

        # Update internal variables
        cls.__annotations__.update(new_options.__annotations__)
        cls.__validators__.update(new_options.__validators__)
        cls.__fields__.update(new_options.__fields__)
        cls.__doc__ += "\n" + new_options.__doc__

    def _get_model_handler(self) -> ModelHandler:
        return self._model_handler

    class Config:

        allow_mutation = True  # False
        arbitrary_types_allowed = True
        extra = 'forbid'


def get_model_options(model='P2D', **kwargs) -> BaseModelOptions:
    """
    Factory method that returns the model options object of the selected
    model
    """
    mtype = get_model_types(model)
    return __model_options__[mtype](model=model, **kwargs)
