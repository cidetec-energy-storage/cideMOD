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
import sys

from mpi4py import MPI
from enum import IntFlag


class VerbosityLevel(IntFlag):
    """
    This method define different verbosity levels to be taken into
    account when printing information.

    Levels
    ------
    NO_INFO : 0
        No information will be displayed.
    BASIC_PROGRESS_INFO : 1
        Show information about the progress of the simulation.
    BASIC_PROBLEM_INFO : 2
        Show information about the problem (e.g. dofs, cell, etc.)
    DETAILED_PROGRESS_INFO : 3
        Detailed information about the progress.
    DETAILED_SOLVER_INFO : 4
        Detailed information about the resolution progress.
    """
    NO_INFO = 0
    BASIC_PROGRESS_INFO = 1
    BASIC_PROBLEM_INFO = 2
    DETAILED_PROGRESS_INFO = 3
    DETAILED_SOLVER_INFO = 4


# TODO: Use the logging built-in python package

class LogLevel(IntFlag):

    DEBUG = 0
    WARNING = 1
    ERROR = 2


def _print_dict(dic, name='', tab_ini='', tab='\t'):
    if name:
        print(tab_ini + name)
        tab_ini += tab

    for key, value in dic.items():
        if isinstance(value, dict):
            _print_dict(value, name=key + ' :', tab_ini=tab_ini, tab=tab)
        else:
            print(tab_ini + key + f' : {value}')


_pad = 0
_need_padding = False


def _print(*args, comm: MPI.Intracomm = None, sep=' ', end='\n', flush=False,
           print_dic_kwargs={}, **kwargs):
    if comm is None or comm.rank == 0:
        global _pad, _need_padding
        if len(args) == 1 and isinstance(args[0], dict):
            # Print the dictionary
            if _need_padding:
                name = print_dic_kwargs.get('name', '')
                print_dic_kwargs['name'] = name.ljust(_pad)
            _print_dict(args[0], **print_dic_kwargs)
            _need_padding = False
        else:
            # Print the strings
            if _need_padding:
                msg = sep.join([str(e) for e in args]).ljust(_pad)
            else:
                msg = sep.join([str(e) for e in args])

            print(msg, end=end, **kwargs)

            if flush:
                sys.stdout.flush()

            if flush or end == '\r' or msg.endswith('\r'):
                if len(msg) > _pad:
                    _pad = len(msg)
                _need_padding = True
            else:
                _need_padding = False
