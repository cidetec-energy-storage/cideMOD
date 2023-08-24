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
"""miscellaneous.py involves all the auxiliary functions which
dosen't belong to any class.

Functions
---------
get_spline(data, spline_type = "not-a-knot")

"""
import functools
import dolfinx as dfx
import multiphenicsx.fem
import ufl
from ufl import conditional, ge, lt, gt
from mpi4py import MPI
from abc import ABC, abstractmethod
from typing import Optional

import glob
import json
import os
import pathlib
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, Akima1DInterpolator, interp1d
from scipy.sparse import csr_matrix

from cideMOD.helpers.logging import _print


def refine_ocp(data, tol=5e-3):
    # TODO: adapt this to Akima and review
    """
    Refine the OCP dataset to shorten the number of data points

    Args:
        data (numpy.dnarray): the ocp data of shape (n,2)
        tol (float, optional): the maximum error allowed in V. Defaults to 5e-3.

    Returns:
        numpy.ndarray: the refined data if was possible to refine the OCP otherwise,
            returns the original data
    """
    high_res_x = data[:, 0]
    high_res_ocp = data[:, 1]
    x_min = high_res_x.min()
    x_max = high_res_x.max()

    ocp_l = interp1d(high_res_x, high_res_ocp)

    low_res_x = np.linspace(x_min, x_max, 10)
    low_res_ocp = ocp_l(low_res_x)
    low_res_ocp_c = CubicSpline(low_res_x, low_res_ocp)
    error = np.abs(high_res_ocp - low_res_ocp_c(high_res_x))
    idx = np.arange(len(high_res_x))
    regions = [idx[(high_res_x < low_res_x[i + 1]) & (high_res_x > low_res_x[i])]
               for i in range(len(low_res_x) - 1)]
    err = [(error[reg].max(), error[reg].mean()) for reg in regions]
    refined_x = low_res_x.tolist()
    while error.max() > tol and len(refined_x) < len(high_res_ocp):
        k = 0  # number of inserts
        for i, region in enumerate(regions):
            if err[i][0] > tol:
                new_x = high_res_x[region].mean()
                refined_x.insert(i + k + 1, new_x)
                k += 1
        regions = [idx[(high_res_x < refined_x[i + 1]) & (high_res_x > refined_x[i])]
                   for i in range(len(refined_x) - 1)]
        err = [(error[reg].max(), error[reg].mean()) if reg.any() else (0, 0) for reg in regions]
        refined_ocp = CubicSpline(refined_x, ocp_l(refined_x))
        error = np.abs(high_res_ocp - refined_ocp(high_res_x))
    refined_data = np.array([refined_x, ocp_l(refined_x)]).T
    return refined_data if len(refined_x) < len(high_res_x) else data


def plot_ocvs(problem, dpi=150):
    return plot_ocps(problem, dpi=150)


def plot_ocps(problem, dpi=150):
    """Plots the OCVs comparing the splines used with the interpolated data

    Args:
       problem: Problem class
       dpi(float, optional): resolution of the plot
    Returns:
        N/A
    """
    # TODO: include option to save figure to file, import figure options as kwargs
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)

    for material in problem.cell.cathode.active_materials:
        ocp = material.parser.ocp
        if ocp['type'] == "expression":
            raise NotImplementedError("Not implemented for expressions")
        else:
            ocp_data = np.loadtxt(ocp['value'])
            ax[0].plot(ocp_data[:, 0], ocp_data[:, 1], 'o', label='Exp')
            xx = np.linspace(min(ocp_data[:, 0]), max(ocp_data[:, 0]), 100)
            yy = get_spline(ocp_data, spline_type=ocp['spline_type'], return_fenics=False)(xx)
            ax[0].plot(xx, yy, label='Spline')

    for material in problem.cell.anode.active_materials:
        ocp = material.parser.ocp
        if ocp['type'] == "expression":
            raise NotImplementedError("Not implemented for expressions")
        else:
            ocp_data = np.loadtxt(ocp['value'])
            ax[1].plot(ocp_data[:, 0], ocp_data[:, 1], 'o', label='Exp')
            xx = np.linspace(min(ocp_data[:, 0]), max(ocp_data[:, 0]), 100)
            yy = get_spline(ocp_data, spline_type=ocp['spline_type'], return_fenics=False)(xx)
            ax[1].plot(xx, yy, label='Spline')

    ax[0].set_title("Positive AMs")
    ax[0].set_xlabel("Stoichiometry [-]")
    ax[0].set_ylabel("U [V]")
    ax[0].legend()
    ax[1].set_title("Negative AMs")
    ax[1].set_xlabel("Stoichiometry [-]")
    ax[1].set_ylabel("U [V]")
    ax[1].legend()
    plt.tight_layout()
    plt.show()


def get_spline(data, spline_type="Akima1D", return_fenics=True):
    """This function adapts the CubicSpline of scipy package to
    the UFL classes type.

    It gets the spline coefficients and uses them for computing
    the spline in a given point, y.

    Parameters
    ----------
    data : array [x_vector, y_vector]
        Data in array form, the first column is the x values,
        and the second column is the function value for x.
    spline_type : str, optional
        If 'Akima1D' use scipy.interpolate.Akima1DInterpolator
        Else spline type for the scipy.interpolate.CubicSpline. See
        scipy documentation for more types, by default "not-a-knot".
    return_fenics : bool, optional
        Whether or not to return a spline which result is a dolfinx
        object.

    Returns
    -------
    UFL Expression
        Spline expression.

    Raises
    ------
    ValueError
        'Unknown type of spline'
    """
    data = np.array(data)
    if data[0, 0] > data[-1, 0]:
        data = np.flip(data, 0)
    x_array = data[:, 0]
    v_array = data[:, 1]

    if spline_type == "Akima1D":
        spline = Akima1DInterpolator(x_array, v_array)
        c = spline.c
        k = len(c) - 1
    elif spline_type in ['not-a-knot', 'clamped', 'natural', 'periodic']:
        spline = CubicSpline(x_array, v_array, bc_type=spline_type)
        c = spline.c
        k = len(c) - 1
    else:
        raise ValueError(f"Unknown type of spline '{spline_type}'")
    if return_fenics:
        def f(y):
            S_list = []

            for j in range(len(x_array) - 1):
                S_list.append(sum(c[m, j] * (y - x_array[j])**(k - m) for m in range(k + 1)))

            fy = 0

            for j in range(len(S_list)):
                fy += S_list[j] * conditional(ge(y, x_array[j]),
                                              conditional(lt(y, x_array[j + 1]), 1, 0), 0)

            fy += S_list[+0] * conditional(lt(y, x_array[+0]), 1, 0)
            fy += S_list[-1] * conditional(ge(y, x_array[-1]), 1, 0)

            return fy
        return f
    else:
        return spline


def hysteresis_property(property: dict):
    def value_selector(x, current=None):
        if current:
            return conditional(
                gt(current, 0),
                property['charge'](x),
                conditional(
                    lt(current, 0),
                    property['discharge'](x),
                    property['charge'](x) * 0.5 + property['discharge'](x) * 0.5
                )
            )
        else:
            return property['discharge'](x)
    return value_selector


def add_to_results_folder(save_path, files, filenames=None, comm=None, overwrite=False):
    """
    Add the given files to the results folder.

    Parameters
    ----------
    save_path: str
        Path to the results folder.
    files: List[Union[str, dict]]
        Files to be saved. Could be the path to an existing file that
        will be copied or a dictionary that will be saved as a json
        file.
    filenames: Optional[List[str]]
        List containing the name of the files.
    comm:Optional[MPI.Intracomm]
        MPI intracommunicator for parallel computing. Default to None.
    overwrite : bool, optional
        Switch for overwriting an existing file. Default to False.
    """
    if comm is not None and comm.rank != 0:
        return
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"save path '{save_path}' does not exist")
    for i, file in enumerate(files):
        fname = filenames[i] if filenames and i < len(filenames) else None
        if isinstance(file, str):
            path = pathlib.Path(file)
            if path.exists():
                shutil.copy(path, os.path.join(save_path, (fname or f'conf_{i}') + path.suffix))
            else:
                raise FileNotFoundError(f"file '{path}' not found")
        elif isinstance(file, dict):
            fpath = os.path.join(save_path, (fname or f'conf_{i}') + '.json')
            if overwrite:
                j = 1
                while os.path.exists(fpath):
                    save_path = os.path.join(save_path, (fname or f'conf_{i}') + f'_v{j}.json')
                    j = j + 1
            with open(fpath, 'w') as fout:
                json.dump(file, fout, indent=4, sort_keys=True)


def init_results_folder(case_name, overwrite=False, copy_files: list = [], filenames: list = [],
                        prefix='results_', comm: Optional[MPI.Intracomm] = None, verbose=True):
    """
    Function to initialize the results folder.

    Parameters
    ----------
    case_name : str
        String containing the case name.
    overwrite : bool, optional
        Switch for overwriting an existing case_name,
        by default False.
    copy_files: List[Union[str, dict]]
        Files to be saved. Could be the path to an existing file that
        will be copied or a dictionary that will be saved as a json
        file.
    filenames: Optional[List[str]]
        List containing the name of the files.
    comm:Optional[MPI.Intracomm]
        MPI intracommunicator for parallel computing. Default to None.
    verbose: bool
        Whether or not to print the save path.

    Returns
    -------
    str
        Complete saving path.
    """
    dir_path, foldername = os.path.split(case_name)
    save_path = os.path.join(
        dir_path, prefix + foldername if not foldername.startswith(prefix) else foldername)
    if comm is None or comm.rank == 0:
        if not overwrite:
            dir_path, foldername = os.path.split(save_path)
            i = 1
            while os.path.exists(save_path):
                save_path = os.path.join(dir_path, foldername + f'_v{i}')
                i = i + 1
    if comm is not None and comm.size > 1:
        save_path = comm.bcast(save_path, root=0)
    if comm is None or comm.rank == 0:
        try:
            os.stat(save_path)
            shutil.rmtree(save_path)
            os.makedirs(save_path, exist_ok=True)
        except Exception:
            os.makedirs(save_path, exist_ok=True)
        if verbose:
            _print('Saving results to', os.path.realpath(save_path))
        dfx.log.set_log_level(dfx.log.LogLevel.WARNING)
        add_to_results_folder(save_path, copy_files, filenames, comm=comm)
    else:
        dfx.log.set_log_level(dfx.log.LogLevel.ERROR)
    return save_path


def constant_expression(expression, return_fenics=True, **kwargs):
    """Evaluates expression with given arguments

    Parameters
    ----------
    expression : str
        String form of the expression in python syntax.
    return_fenics : bool, optional
        Whether or not to return a spline which result is a dolfinx
        object.
    **kwargs: dict
        variables to replace inside the expression.

    Returns
    -------
    value: Float, Constant
        Evaluation of the expression or constant given.
    """
    if not isinstance(expression, str):
        value = expression
    elif return_fenics:
        value = eval(expression, {**ufl.__dict__, **kwargs})
    else:
        value = eval(expression, {**np.__dict__, **kwargs})
    return value


def plot_list_variable(x, y, name='plot', save_path='.', show=False, hide_ax_tick_labels=False,
                       label_axes=True, title='', hide_axis=False, xlabel='x',
                       ylabel='y', ymin=None, ymax=None, xmin=None, xmax=None,
                       i_app=None, data_path=None, save=True, ref="", close=True, fig_kwargs={}):
    """This function plots y values in x axis.

    Parameters
    ----------
    x : list
        List values for x axis.
    y : list
        List values for y axis.
    name : str
        Saving image file name.
    save_path : str
        Saving path.
    show : bool, optional
        Switch for showing the figure, by default True
    hide_ax_tick_labels : bool, optional
        Switch for hiding the axis tick labels, by default False
    label_axes : bool, optional
        Switch for writing the axis labels, by default True
    title : str, optional
        Plot title, by default ''
    hide_axis : bool, optional
        Switch for hiding the axis, by default False
    xlabel : str, optional
        X label name, by default 'x'
    ylabel : str, optional
        Y label name, by default 'y'
    """

    fig = plt.figure(**{'figsize': (5, 5), 'dpi': 200, **fig_kwargs})
    ax = fig.add_subplot(111)

    p = ax.plot(x, y, '-', lw=1, label="Simulation")
    if i_app:
        if "Time" in xlabel:
            try:
                path_list = glob.glob(f'./{data_path}/t_V_{i_app}C*{ref}.txt', recursive=False)
                data = read_input_data(path_list[0], init_line=1)
                xp = data[:, 0]
                yp = data[:, 1]
                p = ax.plot(xp, yp, '.', lw=1, label="Validation data")
                plt.legend()
            except Exception:
                pass
        if "Capacity" in xlabel:
            try:
                path_list = glob.glob(f'./{data_path}/C_V_{i_app}C*{ref}.txt', recursive=False)
                data = read_input_data(path_list[0], init_line=1)
                xp = data[:, 0]
                yp = data[:, 1]
                p = ax.plot(xp, yp, '.', lw=1, label="Validation data")
                plt.legend()
            except Exception:
                pass

    if ymin is None:
        ymin = min(y)
    if ymax is None:
        ymax = max(y)
    if xmin is None:
        xmin = min(x)
    if xmax is None:
        xmax = max(x)

    Dy = (ymax - ymin) * 0.05
    Dx = (xmax - xmin) * 0.05

    ax.set_xlim([xmin - Dx, xmax + Dx])

    ax.set_ylim([ymin - Dy, ymax + Dy])

    if label_axes:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    if hide_ax_tick_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    if hide_axis:
        plt.axis('off')

    tit = plt.title(title)
    plt.tight_layout()
    if save:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, name + '.png'))

    if show:
        plt.show()
    if close:
        plt.close(fig)


def read_input_data(filename, init_line=0):
    file = open(filename, 'r')
    f = []
    line = '0'
    i = 0

    while i < init_line:
        i += 1
        file.readline()

    while len(line) > 0:
        line = np.array(file.readline().split()).astype(float)
        if len(line) > 0:
            f.append(line)
    file.close()
    return np.array(f)


def analyze_jacobian(J, fields):
    J_lab = [[i for i in fields] for i in fields]
    abs_J = np.zeros((len(fields), len(fields)))
    times = np.zeros((len(fields), len(fields)))
    start_time = time.time()
    for i, F in enumerate(fields):
        for j, fld in enumerate(fields):
            try:
                reference = time.time()
                norm = dfx.fem.assemble_matrix(J[i, j]).norm()
                times[i, j] = time.time() - reference
                abs_J[i, j] = norm
            except Exception as e:
                pass
            J_lab[i][j] = "F_{}/d_{}".format(F, fld)
            print(J_lab[i][j], abs_J[i, j], times[i, j])
    print('Total time:', time.time() - start_time)
    start_time = time.time()
    J = multiphenicsx.fem.petsc.assemble_matrix_block(J)
    print('Multiphenics time:', time.time() - start_time)
    mat = J.mat()
    spm = csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)
    plt.spy(spm)
    plt.savefig('J.png')


def save_plot(
        filename='plot', suffix='', isuffix='_v{i}', extension='.png', foldername='',
        overwrite=False):
    if foldername and not os.path.exists(foldername):
        os.makedirs(foldername)
    save_path = os.path.join(foldername, filename + suffix + extension)
    if not overwrite:
        if isuffix == isuffix.format(i=0):
            raise ValueError(f"isuffix must be a string that can be changed via .format(i=i)")
        i = 1
        while os.path.exists(save_path):
            save_path = os.path.join(foldername,
                                     filename + suffix + isuffix.format(i=i) + extension)
            i = i + 1
    plt.savefig(save_path)


def plot_jacobian(problem, x, J, filename='J', save_path='', extension='.png', overwrite=False,
                  save_fig=True):
    plt.clf()
    problem._prepare_variable_indices(x)
    size = x.size - 1
    frontiers = [d.max() for d in problem.dofs]
    locations = [(d.max() + d.min()) / 2 for d in problem.dofs]
    fields = [sol.name for sol in problem._solutions]
    csmat = csr_matrix(J.getValuesCSR()[::-1], shape=J.size)
    mat_dict = csmat.todok()
    xy = np.array(list(mat_dict.keys()))
    vals = np.array(list(mat_dict.values()))
    plt.scatter(xy[:, 1], size - xy[:, 0], s=5, c=np.log10(np.abs(vals)), cmap='inferno')
    for line in frontiers:
        plt.axhline(y=size - line, linestyle='--', linewidth=0.5)
        plt.axvline(x=line, linestyle='--', linewidth=0.5)
    for i, (text, xloc) in enumerate(zip(fields, locations)):
        factor = 1.1 if i % 2 == 0 else 1.05
        plt.text(xloc, max(frontiers) * factor, text, fontsize=8, ha='center')
    plt.colorbar()
    plt.xlim([0, size])
    plt.ylim([0, size])
    plt.xticks([], [])
    plt.yticks([], [])
    if save_fig:
        save_plot(filename=filename, isuffix='_{i}', extension=extension,
                  foldername=save_path, overwrite=overwrite)
    plt.show()


def format_time(timespan, small=True):
    # we have more than a minute, format that in a human readable form
    # Idea from http://snipplr.com/view/5713/
    if small:
        parts = [("d", 60 * 60 * 24), ("h", 60 * 60), ("min", 60), ("s", 1)]
    else:
        parts = [(" days", 60 * 60 * 24), (" hours", 60 * 60), (" minutes", 60), (" seconds", 1)]
    time = []
    leftover = timespan
    for suffix, length in parts:
        value = int(leftover / length)
        if value > 0:
            leftover = leftover % length
            time.append(u'%s%s' % (str(value), suffix))
        if leftover < 1:
            break
    return " ".join(time)


def generate_CV_txt(fname: str, v_max=4.2, v_min=2.8, v_rate=1e-3, v_init=None, dT=1):
    # Function to generate a cyclic voltammetry profile
    # Starts in v_init and goes until v_max, then goes down from to v_min
    # TODO: test robustness, maybe define voltage in mV to avoid float operations
    if v_init is None:
        v_init = (((v_max + v_min) / 2) // v_rate) * v_rate

    dv = v_rate * dT
    steps_up = int((v_max - v_init) // dv)
    steps_down = int((v_max - v_min + dv) // dv)
    print(steps_up, steps_down)
    steps_total = steps_up + steps_down
    voltage = np.concatenate((
        np.linspace(v_init, v_max - dv, steps_up),
        np.linspace(v_max, v_min, steps_down)))

    time = np.linspace(0, (steps_total - 1) * dT, steps_total)

    np.savetxt(fname, np.column_stack((time, voltage)), delimiter=";", fmt="%.3f")


def generate_class_name(name: str, prefix: str = '', suffix: str = ''):
    """
    This method returns a valid class name from the given one.

    Parameters
    ----------
    name: str
        Initial name from which the class name will be generated. It
        must be a valid identifier.
    prefix: str
        Prefix of the class name.
    suffix: str
        Suffix of the class name.

    Examples
    --------
    >>> generate_class_name('electrode', suffix='Parameters')
    'ElectrodeParameters'

    >>> generate_class_name('active_material', suffix='Parser')
    'ActiveMaterialParser'

    >>> generate_class_name('SEI', prefix= 'ModelOptions')
    'ModelOptionsSEI'
    """
    if not name.isidentifier():
        raise ValueError(f"name '{name}' is not a valid identifier")
    cls_name = name if name.isupper() else name.capitalize()
    cls_name = functools.reduce(lambda key1, key2: key1 + key2.capitalize(), cls_name.split('_'))
    cls_name = prefix + cls_name[0].upper() + cls_name[1:] + suffix
    return cls_name


class ParsedList(list, ABC):
    """
    Container abstracting a list of elements of some specific types.
    """
    @classmethod
    @property
    @abstractmethod
    def _white_list_(cls):
        raise NotImplementedError

    def __init__(self, iterable=None):
        super(ParsedList, self).__init__()
        if iterable is not None:
            self.extend(iterable)

    def __setitem__(self, key, value):
        self._check_item(value)
        return super(ParsedList, self).__setitem__(key, value)

    def __add__(self, value):
        _value = value if not type(value).__name__ == "list" else self.__class__(value)
        self._check_concatenation_object(_value)
        out = self.copy()
        out.extend(_value)
        return out

    def __iadd__(self, value):
        _value = value if not type(value).__name__ == "list" else self.__class__(value)
        self._check_concatenation_object(_value)
        self._check_items(_value)
        return super(ParsedList, self).__iadd__(_value)

    def __radd__(self, value):
        self._check_concatenation_object(value)
        return NotImplemented

    def __mul__(self, value):
        if not isinstance(value, int):
            raise TypeError(f"can't multiply sequence by non-int of type '{type(value)}'")
        out = self.copy()
        for _ in range(max(value - 1, 0)):
            out.extend(self)
        return out

    def __imul__(self, value):
        if not isinstance(value, int):
            raise TypeError(f"can't multiply sequence by non-int of type '{type(value)}'")
        return super(ParsedList, self).__imul__(value)

    def __rmul__(self, value):
        return self.__mul__(value)

    def insert(self, index, value):
        self._check_item(value)
        return super(ParsedList, self).insert(index, value)

    def append(self, value):
        self._check_item(value)
        return super(ParsedList, self).append(value)

    def extend(self, value):
        self._check_items(value)
        return super(ParsedList, self).extend(value)

    def copy(self):
        copy = self.__class__()
        copy.extend(self)
        return copy

    def _check_items(self, lst=None):
        if lst is None:
            lst = self
        if not isinstance(lst, self.__class__):
            for item in lst:
                self._check_item(item)

    @classmethod
    def _check_item(cls, value):
        if isinstance(value, cls._white_list_):
            return
        elif not isinstance(cls._white_list_, tuple):
            raise TypeError(
                f"{cls.__name__} only admit elements of type '{cls._white_list_.__name__}'")
        else:
            raise TypeError(f"{cls.__name__} only admit elements of type '"
                            + "' '".join([item.__name__ for item in cls._white_list_]) + "'")

    @classmethod
    def _check_concatenation_object(cls, obj):
        if not isinstance(obj, cls):
            raise TypeError(f"Can only concatenate {cls.__name__} "
                            + f"(not '{type(obj).__name__}') to {cls.__name__}")
