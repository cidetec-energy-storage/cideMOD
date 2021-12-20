#
# Copyright (c) 2021 CIDETEC Energy Storage.
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
from ufl.core.expr import Expr
from ufl.core.ufl_type import ufl_type
from ufl.operators import _mathfunction
from ufl.constantvalue import Zero, ScalarValue, as_ufl, FloatValue, is_true_ufl_scalar

# TODO: Esta parte está bien de momento, pero no funciona.
#       Faltaría: 
#       1. añadir un estimador del grado de cuadratura (en ufl/algorithms/estimate_degrees.py) 
#       2. hacer la traducción a C++ (en fcc/uflacs/language/ufl_to_cnodes.py)
#       La parte de C++ es la más complicada, aún no sé muy bien como hacerlo. El punto 1 es sencillo.

@ufl_type(is_scalar=True, num_ops=1, is_terminal=False)
class Spline(Expr):
    __slots__ = ('_name','_spline_object', 'ufl_operands')
    
    def __new__(cls, argument, spline_object):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(float(spline_object(float(argument))))
        return Expr.__new__(cls)

    def __init__(self, argument, spline_object):
        Expr.__init__(self)
        self.ufl_operands = (argument,)
        if not is_true_ufl_scalar(argument):
            error("Expecting scalar argument.")
        self._name = 'math_function'
        self._spline_object = spline_object

    def evaluate(self, x, mapping, component, index_values):
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        try:
            res = float(self._spline_object(a))
        except ValueError:
            warning('Value error in evaluation of function %s with argument %s.' % (self._name, a))
            raise
        return res

    def derivative(self):
        f, = self.ufl_operands
        return MathFunction(f , self.spline_object.derivative())

    def __str__(self):
        return "%s(%s)" % (self._name, self.ufl_operands[0])

    def _ufl_expr_reconstruct_(self, *operands):
        "Return a new object of the same type with new operands."
        return self._ufl_class_(*operands, self._spline_object)

    def _ufl_signature_data_(self):
        return self._ufl_typecode_

    def _ufl_compute_hash_(self):
        "Compute a hash code for this expression. Used by sets and dicts."
        return hash((self._ufl_typecode_,) + tuple(hash(o) for o in self.ufl_operands))

    def __repr__(self):
        "Default repr string construction for operators."
        # This should work for most cases
        r = "%s(%s)" % (self._ufl_class_.__name__,
                        ", ".join(repr(op) for op in self.ufl_operands))
        return r


def spline(f, spline_object):
    f = as_ufl(f)
    r = Spline(f, spline_object)
    if isinstance(r, (ScalarValue, Zero, int, float)):
        return float(r)
    return r
