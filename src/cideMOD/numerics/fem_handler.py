import ufl
import dolfinx as dfx
import multiphenicsx.fem as mpx

from typing import List, Tuple, Union, Optional

import numpy as np

def interpolate(ex, V:Union[dfx.fem.FunctionSpace,dfx.fem.Function]):
    if isinstance(V, dfx.fem.FunctionSpace):
        points = V.element.interpolation_points
        f = dfx.fem.Function(V)
        expr = dfx.fem.Expression(ex, V.element.interpolation_points)
    elif isinstance(V, dfx.fem.Function):
        points = V.function_space.element.interpolation_points
        f = V
    else:
        raise Exception("V must be Function or FunctionSpace")
    if isinstance(ex, (float,int)):
        f.vector.array = ex
    elif isinstance(ex,dfx.fem.Function):
        f.interpolate(ex)
    else:
        expr = dfx.fem.Expression(ex, points)
        f.interpolate(expr)
    return f

class BlockFunction:
    functions: List[Union[dfx.fem.Function, ufl.Argument]]
    var_names: List[str]
    def __init__(self, functions, var_names) -> None:
        self.functions=functions
        self.var_names = var_names
        for function, name in zip(self.functions, self.var_names):
            setattr(self, name, function)

    def __call__(self, varname:str) -> dfx.fem.Function:
        return self.functions[self.var_names.index(varname)]
    
    def __getitem__(self, i:int):
        return self.functions[i]

    def _asdict(self):
        return {k: f for k,f in zip(self.var_names, self.functions)}

class BlockFunctionSpace:
    function_spaces: List[dfx.fem.FunctionSpace]
    restriction_elements: Tuple[int,List[np.ndarray]]
    restrictions: List[mpx.DofMapRestriction]
    var_names: List[str]
    def __init__(self, var_names, function_spaces, restriction_elements) -> None:
        self.var_names = var_names
        self.function_spaces = function_spaces
        self.restriction_elements=restriction_elements
        self._create_restrictions()

    def create_block_function(self):
        return BlockFunction([dfx.fem.Function(fs,name=name) for fs, name in zip(self.function_spaces,self.var_names)], self.var_names)

    def create_test_function(self):
        return BlockFunction([ufl.TestFunction(fs) for fs in self.function_spaces], self.var_names)

    def create_trial_function(self):
        return BlockFunction([ufl.TrialFunction(fs) for fs in self.function_spaces], self.var_names)

    def _create_restriction(self, dim, entities, function_space)-> mpx.DofMapRestriction:
        if entities is None:
            dofs = np.arange(0, function_space.dofmap.index_map.size_local + function_space.dofmap.index_map.num_ghosts)
        else:
            dofs = dfx.fem.locate_dofs_topological(function_space, dim, entities)
        return mpx.DofMapRestriction(function_space.dofmap, dofs)

    def _create_restrictions(self):
        restrictions = []
        for function_space, rest_elemets in zip(self.function_spaces, self.restriction_elements):
            if rest_elemets is None:
                (dim, entities) = None, None
            else:
                (dim, entities) = rest_elemets
            restrictions.append(self._create_restriction(dim, entities, function_space))
        self.restrictions = restrictions

    def __call__(self, varname:str) -> dfx.fem.Function:
        return self.function_spaces[self.var_names.index(varname)]
    
    def __getitem__(self, i:int):
        return self.function_spaces[i]


def assign(destination:Union[BlockFunction,dfx.fem.Function], source:Union[BlockFunction,dfx.fem.Function]):
    assert type(source)==type(destination)
    if isinstance(source, dfx.fem.Function):
        destination.interpolate(source)
    else:
        assert all(sf==df for sf,df in zip(source.var_names, destination.var_names))
        for sf, df in zip(source.functions, destination.functions):
            df.interpolate(sf)

def assemble_scalar(form: Union[ufl.Form,dfx.fem.FormMetaClass]):
    return dfx.fem.assemble_scalar(dfx.fem.form(form))

def block_derivative(F:List[ufl.Form], u:List[dfx.fem.Function], du:Optional[List[ufl.Argument]])->List[List[ufl.Form]]:
    return [[ufl.derivative(F_i,u_i,du_i) if F_i else None for u_i, du_i in zip(u,du)] for F_i in F ]