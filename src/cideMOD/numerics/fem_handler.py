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
import math
import operator
import numpy as np

import ufl
import ufl.algebra
import ufl.operators
import ufl.exproperators
import ufl.constantvalue
import ufl.mathfunctions
import ufl.core.operator
import dolfinx as dfx
import multiphenicsx.fem as mpx

from mpi4py import MPI
from petsc4py import PETSc
from typing import Any, List, Tuple, Union, Optional, Sequence, overload

dolfinx_classes = (ufl.core.operator.Operator, ufl.constantvalue.ConstantValue,
                   dfx.fem.Constant, dfx.fem.Function)


@overload
def isinstance_dolfinx(*args: Union[Tuple, Tuple[Sequence]],
                       dolfinx_classes=dolfinx_classes) -> List[bool]:
    ...


@overload
def isinstance_dolfinx(*args: Tuple[Any],
                       dolfinx_classes=dolfinx_classes) -> bool:
    ...


def isinstance_dolfinx(*args, dolfinx_classes=dolfinx_classes) -> Union[bool, List[bool]]:
    """
    This function checks whether a sequence contains instances of
    dolfinx classes. Returns a boolean or a list of booleans.
    """
    if not args:
        raise TypeError('isinstance_dolfinx expected at least 1 argument, got 0')
    elif len(args) > 1:
        return [isinstance(value, dolfinx_classes) for value in args]
    elif isinstance(args[0], (list, tuple)):
        return [isinstance(value, dolfinx_classes) for value in args[0]]
    else:
        return isinstance(args[0], dolfinx_classes)


def interpolate(ex, V: Union[dfx.fem.FunctionSpace, dfx.fem.Function],
                cells: Optional[np.ndarray] = None, dofs: Optional[np.ndarray] = None):
    """
    Method to interpolate ex into V.

    Parameters
    ----------
    ex : Union[float,int,numpy.ndarray, dolfinx.fem.Function,
               ufl.Operator, dolfinx.Expression]
        The object to interpolate to.
    V : Union[dfx.fem.FunctionSpace,dfx.fem.Function]
        The function (V or Function(V)) to interpolate into.
    cells : Optional[numpy.ndarray]
        The cells (mesh entities) to interpolate over. If None then all
        cells are interpolated over. Used only if ex is an instance of
        dolfinx.fem.Function, ufl.Operator or dolfinx.Expression.
    dofs : Optional[numpy.ndarray]
        The dofs (of V or Function(V)) to interpolate over. If None then
        all dofs are interpolated over. Used only if ex is an instance
        of float, int, numpy.ndarray.

    Returns
    -------
    dfx.fem.Function
        The interpolated function.
    """
    if isinstance(V, dfx.fem.FunctionSpace):
        points = V.element.interpolation_points()
        f = dfx.fem.Function(V)
    elif isinstance(V, dfx.fem.Function):
        points = V.function_space.element.interpolation_points()
        f = V
    else:
        raise Exception("V must be Function or FunctionSpace")
    if isinstance(ex, (float, int)):
        if dofs is None:
            f.x.array[:] = ex
        else:
            f.x.array[dofs] = ex
        f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    elif isinstance(ex, np.ndarray):
        ndofs = (f.function_space.dofmap.index_map.size_local
                 + f.function_space.dofmap.index_map.num_ghosts)
        if dofs is None:
            if len(ex) == ndofs:
                f.vector.setValuesLocal(np.arange(ndofs, dtype=np.int32), ex)
            elif len(ex) == 1:
                f.x.array[:] = ex[0]
            else:
                raise ValueError("Invalid source of type numpy.ndarray for interpolate: "
                                 + "Number of dofs have to coincide")
        else:
            if len(ex) == ndofs:
                f.vector.setValuesLocal(dofs, ex[dofs])
            elif len(ex) == len(dofs):
                f.vector.setValuesLocal(dofs, ex)
            elif len(ex) == 1:
                f.x.array[dofs] = ex[0]
            else:
                raise ValueError("Invalid source of type numpy.ndarray for interpolate: "
                                 + "Number of dofs have to coincide")
        f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    elif isinstance(ex, dfx.fem.Function):
        f.interpolate(ex, cells=cells)
    else:
        expr = dfx.fem.Expression(ex, points)
        f.interpolate(expr, cells=cells)
    return f


class BlockFunction:
    functions: List[Union[dfx.fem.Function, ufl.Argument]]
    var_names: List[str]

    def __init__(self, var_names, functions) -> None:
        self.functions = functions
        self.var_names = var_names
        for function, name in zip(self.functions, self.var_names):
            setattr(self, name, function)

    def __call__(self, varname: str) -> dfx.fem.Function:
        return self.functions[self.var_names.index(varname)]

    def __getitem__(self, i: int):
        return self.functions[i]

    def _asdict(self):
        return {k: f for k, f in self.items()}

    def items(self):
        return zip(self.var_names, self.functions)

    def clear(self):
        for fnc in self.functions:
            interpolate(0., fnc)


class BlockFunctionSpace:
    function_spaces: List[dfx.fem.FunctionSpace]
    restriction_elements: Tuple[int, List[np.ndarray]]
    restrictions: List[mpx.DofMapRestriction]
    var_names: List[str]

    def __init__(self, var_names, function_spaces, restriction_elements) -> None:
        self.var_names = var_names
        self.function_spaces = function_spaces
        self.restriction_elements = restriction_elements
        self._create_restrictions()

    def create_block_function(self, prefix='', suffix=''):
        return BlockFunction(self.var_names,
                             [dfx.fem.Function(fs, name=prefix + name + suffix)
                              for fs, name in zip(self.function_spaces, self.var_names)])

    def create_test_function(self):
        return BlockFunction(self.var_names, [ufl.TestFunction(fs) for fs in self.function_spaces])

    def create_trial_function(self):
        return BlockFunction(self.var_names,
                             [ufl.TrialFunction(fs) for fs in self.function_spaces])

    def _create_restriction(self, dim, entities, function_space) -> mpx.DofMapRestriction:
        if entities is None:
            ndofs = (function_space.dofmap.index_map.size_local
                     + function_space.dofmap.index_map.num_ghosts)
            dofs = np.arange(0, ndofs)
        else:
            dofs = dfx.fem.locate_dofs_topological(function_space, dim, entities)
        return mpx.DofMapRestriction(function_space.dofmap, dofs)

    def _create_restrictions(self):
        restrictions = []
        for function_space, rest_elements in zip(self.function_spaces, self.restriction_elements):
            if rest_elements is None:
                (dim, entities) = None, None
            else:
                (dim, entities) = rest_elements
            restrictions.append(self._create_restriction(dim, entities, function_space))
        self.restrictions = restrictions

    def get_restriction(self, varname: str) -> mpx.DofMapRestriction:
        return self.restrictions[self.var_names.index(varname)]

    def __call__(self, varname: str) -> dfx.fem.FunctionSpace:
        return self.function_spaces[self.var_names.index(varname)]

    def __getitem__(self, i: int) -> dfx.fem.FunctionSpace:
        return self.function_spaces[i]


def assign(destination: Union[BlockFunction, dfx.fem.Function],
           source: Union[BlockFunction, dfx.fem.Function]):
    assert isinstance(source, type(destination))
    if isinstance(source, dfx.fem.Function):
        destination.interpolate(source)
    else:
        assert all(sf == df for sf, df in zip(source.var_names, destination.var_names))
        for sf, df in zip(source.functions, destination.functions):
            df.interpolate(sf)


def assemble_scalar(form: Union[ufl.Form, dfx.fem.Form], comm=MPI.COMM_WORLD):
    value = dfx.fem.assemble_scalar(dfx.fem.form(form))
    return comm.allreduce(value, MPI.SUM)


def block_derivative(F: List[ufl.Form], u: List[dfx.fem.Function],
                     du: Optional[List[ufl.Argument]]) -> List[List[ufl.Form]]:
    return [[ufl.derivative(F_i, u_i, du_i) if F_i else None for u_i, du_i in zip(u, du)]
            for F_i in F]


def _max(*args):
    """
    This function adapts the max built-in method to the dolfinx classes.
    """
    if not args:
        raise TypeError('_max expected 1 arguments, got 0')
    elif len(args) == 1:
        seq = args[0]
        if not isinstance(seq, (tuple, list, np.ndarray)):
            raise TypeError(f"Invalid type '{type(seq).__name__}'")
    elif len(args) >= 2:
        seq = args

    is_ufl = np.array(isinstance_dolfinx(seq), dtype=bool)
    if not is_ufl.any():
        return max(seq)

    # iter over no ufl instances
    res = None
    for i in np.where(~is_ufl)[0]:
        res = max(res, seq[i]) if res is not None else seq[i]

    # iter over ufl instances
    for i in np.where(is_ufl)[0]:
        res = ufl.max_value(res, seq[i]) if res is not None else seq[i]
    return res


def _min(*args):
    """
    This function adapts the min built-in method to the UFL classes
    type.
    """
    if not args:
        raise TypeError('_min expected 1 arguments, got 0')
    elif len(args) == 1:
        seq = args[0]
        if not isinstance(seq, (tuple, list, np.ndarray)):
            raise TypeError(f"Invalid type '{type(seq).__name__}'")
    elif len(args) >= 2:
        seq = args

    is_ufl = np.array(isinstance_dolfinx(seq), dtype=bool)
    if not is_ufl.any():
        return min(seq)

    # iter over no ufl instances
    res = None
    for i in np.where(~is_ufl)[0]:
        res = min(res, seq[i]) if res is not None else seq[i]

    # iter over ufl instances
    for i in np.where(is_ufl)[0]:
        res = ufl.min_value(res, seq[i]) if res is not None else seq[i]
    return res


def _linspace(a, b, **kwargs):
    """
    This function adapts the linspace method of numpy package to the
    UFL classes type.
    """
    return [a + (b - a) * i for i in np.linspace(0., 1., **kwargs)]


def _mean(seq):
    """
    This function adapts the mean method of numpy package to the UFL
    classes type.
    """
    return sum(seq) / len(seq)


_ufl_binoperators_dict = {
    ufl.algebra.Sum: operator.add,
    ufl.algebra.Product: operator.mul,
    ufl.algebra.Division: operator.truediv,
    ufl.algebra.Power: operator.pow,
    ufl.operators.MaxValue: max,
    ufl.operators.MinValue: min,
    ufl.operators.EQ: operator.eq,
    ufl.operators.NE: operator.ne,
    ufl.exproperators.LE: operator.le,
    ufl.exproperators.GE: operator.ge,
    ufl.exproperators.LT: operator.lt,
    ufl.exproperators.GT: operator.gt,
    ufl.operators.AndCondition: operator.and_,
    ufl.operators.OrCondition: operator.or_,
}


def _evaluate_parameter(ex):
    """
    This method evaluate the given expression assuming that its a ufl
    expression of dolfinx.fem.Constant, neither coefficients nor
    variables.
    """
    if isinstance(ex, (float, int)):
        return ex
    elif isinstance(ex, ufl.core.operator.Operator):
        # Binary operator
        binop = _ufl_binoperators_dict.get(type(ex), None)
        if binop is not None:
            return binop(*[_evaluate_parameter(op) for op in ex.ufl_operands])
        # Trinary operator
        elif isinstance(ex, ufl.operators.Conditional):
            condition = bool(_evaluate_parameter(ex.ufl_operands[0]))
            res = ex.ufl_operands[1] if condition else ex.ufl_operands[2]
            return _evaluate_parameter(res)
        # Unary operator
        if ex.ufl_operands:
            op = _evaluate_parameter(ex.ufl_operands[0])
        if isinstance(ex, ufl.mathfunctions.MathFunction):
            return getattr(math, ex._name)(op)
        elif isinstance(ex, ufl.algebra.Abs):
            return abs(op)
        elif isinstance(ex, ufl.operators.NotCondition):
            return not bool(op)
    elif isinstance(ex, dfx.fem.Constant):
        return ex.value
    elif isinstance(ex, (ufl.constantvalue.RealValue, ufl.constantvalue.Zero)):
        return ex.value()
    ex_dtype = '.'.join([type(ex).__module__, type(ex).__name__]).lstrip('.')
    raise RuntimeError(f"Unable to get the value from a '{ex_dtype}' object")
