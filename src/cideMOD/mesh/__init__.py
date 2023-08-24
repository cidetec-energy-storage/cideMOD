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
This module provides the logic needed to create, store, and reuse
battery meshes. This module is not intended to interact with users
directly, the Problem class automatically handle the meshing process
"""

# TODO: Remove SubdomainMapper from here (make it private) and let's the mesher be the public face
from cideMOD.mesh.base_mesher import DolfinMesher, BaseMesher, SubdomainMapper
from cideMOD.mesh.gmsh_adapter import GmshMesher

__all__ = ["DolfinMesher", "BaseMesher", "SubdomainMapper", "GmshMesher"]
