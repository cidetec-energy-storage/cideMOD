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
from pydantic import BaseModel, validator


class ModelOptions(BaseModel):
    """Settings for the PXD Model

    :param mode: Simulation mode, default "P2D" 
    :type mode: str
    :param solve_thermal: Wether to solve thermal problem or not, default False 
    :type solve_thermal: bool
    :param solve_mechanic: Wether to solve mechanic problem or not, default False 
    :type solve_mechanic: bool
    :param solve_SEI: Wether to solve SEI problem or not, default False 
    :type solve_SEI: bool
    :param N_x: Discretization in x direciton, default 30 
    :type N_x: int
    :param N_y: Discretization in y direciton, default 10 
    :type N_y: int
    :param N_z: Discretization in z direciton, default 10 
    :type N_z: int
    :param FEM_order: Order of interpolating finite elements, default 1 
    :type FEM_order: int
    :param particle_coupling: Coupling between cell and particle problem, one of ("implicit","explicit"), default "implicit" 
    :type particle_coupling: str
    :param N_p: Particle discretization, only relevant fot explicit coupling, default 20 
    :type N_p: int
    :param particle_order: Order of spectral finite elements interpolation in particle, default 2 
    :type particle_order: int
    :param time_scheme: Time discretization scheme, default "euler_implicit" 
    :type time_scheme: str
    :param clean_on_exit: Wether to clean from memory saved data at the end of the solve cycle or not, default True 
    :type clean_on_exit: bool
    :return: ModelOptions instance
    :rtype: ModelOptions
    """
    mode: str = "P2D"
    solve_thermal: bool = False
    solve_mechanic: bool = False
    solve_SEI: bool = False
    N_x: int = 30
    N_y: int = 10
    N_z: int = 10
    FEM_order: int = 1
    particle_coupling: str = "implicit"
    N_p: int = 20
    particle_order: int = 2
    time_scheme: str = "euler_implicit"
    clean_on_exit: bool = False

    @validator("mode")
    def validate_mode(cls, v):
        assert v in ("P2D", "P3D", "P4D"), "mode keyword must be one of P2D, P3D or P4D"
        return v

    @validator("particle_coupling")
    def validate_coupling(cls, v):
        assert v in (
            "implicit",
            "explicit",
        ), "particle_coupling must be implicit or explicit"
        return v

