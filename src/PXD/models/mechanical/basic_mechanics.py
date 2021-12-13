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
from dolfin import as_vector, as_tensor
import numpy as np
from numpy.linalg import inv as invert

class BaseMechanicalModel:
    """
    Implements standard functionalities to calculate and manipulate mechanical tensors.
    """

    def elasticity_tensor(self, E:float, nu:float):
        """
        Elasticity tensor for isotropic materials

        Parameters
        ----------
        E : float
            Young Modulus
        nu : float
            Poisson ratio

        Returns
        -------
        Matrix
            Elasticity tensor
        """
        L = 1/E * np.array([
                [1 , -nu , -nu,  0,   0,   0 ],
                [-nu , 1 , -nu,  0,   0,   0 ],
                [-nu , -nu , 1,  0,   0,   0 ],
                [ 0 ,  0 ,  0, 1+nu,  0,   0 ],
                [ 0 ,  0 ,  0,   0, 1+nu,  0 ],
                [ 0 ,  0 ,  0,   0,   0, 1+nu],
            ])
        C = invert(L)
        return C

    def elsheby_tensor(self, nu: float):
        """
        Elsheby's tensor for spherical inclusions as defined by Mura (1982)

        Parameters
        ----------
        nu : float
            Poisson ratio

        Returns
        -------
        rank 2 Tensor
            Elsheby's tensor
        """
        Siiii = (7 - 5*nu)/(15*(1 - nu))
        Sijij = (4 - 5*nu)/(15*(1 - nu))
        Siijj = (5*nu - 1)/(15*(1 - nu))
        S = np.array([
                [ Siiii , Siijj , Siijj,    0  ,    0  ,    0   ],
                [ Siijj , Siiii , Siijj,    0  ,    0  ,    0   ],
                [ Siijj , Siijj , Siiii,    0  ,    0  ,    0   ],
                [   0   ,   0   ,   0  ,  Sijij,    0  ,    0   ],
                [   0   ,   0   ,   0  ,    0  ,  Sijij,    0   ],
                [   0   ,   0   ,   0  ,    0  ,    0  ,  Sijij ],
            ])
        return S

    def homogeneized_elasticity_tensor(self, C_matrix:np.ndarray, C_inclusion:np.ndarray, elsheby_tensor:np.ndarray, volume_ratio:float):
        """
        Calculates the homogeneized elasticity tensor with the Mori-Tanaka theory.
        See Golmon et al. (2009) for more references

        Parameters
        ----------
        C_matrix : numpy.ndarray
            Host matrix elasticity tensor
        C_inclusion : numpy.ndarray
            Inclusion elasticity tensor
        elsheby_tensor : numpy.ndarray
            Elsheby's tensor for the inclusions
        volume_ratio : float
            Volume ratio of the inclusion in the host matrix

        Returns
        -------
        Tensor
            Effective elasticity tensor
        """
        eps = 1-volume_ratio
        assert eps >= 0 and eps <= 1 , "Volume ratio not in range"
        C_m = C_matrix
        C_i = C_inclusion
        S = elsheby_tensor
        I = np.identity(C_m.shape[0])

        A_D = invert(I + S @ invert(C_m) @ (C_i - C_m))
        A_S = A_D @ invert(eps * I + (1 - eps) * A_D)
        C_eff = C_m + (1 - eps) * (C_i - C_m) @ A_S
        
        return C_eff

    def lame_parameters(self, E:float, nu:float):
        lmbda = E*nu/(1+nu)/(1-2*nu)
        mu = E/(2.*(1+nu))
        return lmbda, mu

    def _tensor2voight(self, tensor):
        """Takes a 2nd-order tensor, returns its Voigt vectorial representation"""
        if isinstance(tensor, np.ndarray):
            raise Exception("Expected a dolfin ListTensor")
        else:
            assert len(tensor.ufl_shape) == 2, "Tensor must be of rank 2"
            assert tensor.ufl_shape[0] == tensor.ufl_shape[1], "Tensor must be square"
            dim = tensor.geometric_dimension()
            voight_vector = []
            for i in range(dim):
                voight_vector.append(tensor[i,i])
            for i in range(dim):
                for j in range(i+1,dim):
                    voight_vector.append(tensor[i,j])
            return as_vector(voight_vector)

    def _voight2tensor(self,voight_vector):
        """Takes a stress-like Voigt vector, returns its tensorial representation"""
        if isinstance(voight_vector, np.ndarray):
            raise Exception("Expected a dolfin ListTensor")
        if len(voight_vector.ufl_shape) == 0:
            return as_tensor([[voight_vector]])
        assert len(voight_vector.ufl_shape) == 1, "Vector must be of rank 1"
        dim = voight_vector.geometric_dimension()
        L = max(voight_vector.ufl_shape)
        tensor = [[0 for i in range(dim)] for j in range(dim)]
        for i in range(dim):
            tensor[i][i] = voight_vector[i]
        for i in range(dim-1):
            for j in range(i+1, dim):
                tensor[i][j] = voight_vector[(dim-1)+i+j] 
                tensor[j][i] = voight_vector[(dim-1)+i+j] 
        return as_tensor(tensor)

    def _reduce_tensor(self, tensor, dim:int=3):
        """
        Reduces a 4th-order tensor for problems with lower dimension

        Parameters
        ----------
        tensor : numpy.array
            4th-order tensor in voight notation to be reduced
        dim : int, optional
            Dimension of the problem, by default 3

        Returns
        -------
        dolfin List-Tensor
            Reduced 4th-order tensor to given dimension

        Raises
        ------
        Exception
            Dimension must be a value in (1,2,3)
        """
        fot = tensor
        if dim == 1:
            fot = np.array([[fot[0,0]]])
        elif dim == 2:
            fot = np.array([
                [ fot[0,0], fot[0,1], fot[0,3] ],
                [ fot[1,0], fot[1,1], fot[1,3] ],
                [ fot[3,0], fot[3,1], fot[3,3] ]
            ])
        elif dim != 3:
            raise Exception("Dimension must be 1, 2 or 3")
        return as_tensor(fot)