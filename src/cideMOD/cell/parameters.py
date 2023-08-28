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
import numpy as np
import dolfinx as dfx
import ufl

from petsc4py.PETSc import ScalarType
from typing import Union, Optional, Tuple

from cideMOD.helpers.miscellaneous import constant_expression, hysteresis_property, get_spline
from cideMOD.numerics.fem_handler import _evaluate_parameter, isinstance_dolfinx


class CellParameter:
    """
    Class abstracting a cell parameter.

    Parameters
    ----------
    component: BaseComponentParser
        Object that parses the component parameters.
    dic: Optional[dict]
        Dictionary with cell parameter information. Contains:

        * 'value': Value of the parameter.
        * 'unit': Units of the parameter. Optional.
        * 'effective': Whether or not is an effective value. Optional
        * 'arrhenius': Dictionary containing the arrhenius parameters.
          Contains `activationEnergy` and `referenceTemperature`.
          Optional.
        * 'referenceTemperature': Temperature of reference of this
          parameter. Optional.
        * 'source': Type of source of the data. Right now 'file' is the
          only one implemented. Only used if the dtype is 'spline'.
          Optional.
        * 'spline_type': Type of spline to used. It must be either
          `not-a-knot` for cubic interpolation or `Akima1D` for Akima
          interpolation. Only used if the dtype is 'spline'. Optional.

    name: str
        Name of the cell parameter.
    dtypes: Optional[Union[Tuple[str], str]]
        Available datatypes that are allowed. It must be one of
        `real`, `expression`, `spline` or `label`.
    is_optional: bool
        Whether this parameter is optional or not. Default to True.
    can_vary: bool
        Whether this parameter can be defined as a dynamic parameter.
        Default to True.
    can_effective: bool
        Whether this parameter can be defined as an effective parameter.
        Default to False.
    can_arrhenius: bool
        Whether the arrhenius correction can be applied to this
        parameter. Default to False.
    can_ref_temperature: bool
        Whether this parameter can have a reference temperature.
    can_hysteresis: bool
        Whether this parameter can have hysteresis.
    data_path: Optional[str]
        Path to the referenced data if the given source is a file.
    description: Optional[str]
        Description of the parameter.
    aliases: Union[list, tuple]
        List or tuple containing the aliases of the parameter if any.
    """

    @property
    def value(self):
        # NOTE: Assume _value_ has already been setup otherwise return None
        return self._value_

    @property
    def ref_value(self):
        # NOTE: Assume _ref_value_ has already been setup otherwise return None
        return self._ref_value_

    @property
    def user_value(self):
        return self._user_value_

    def __init__(self, component, dic: Optional[dict], name: str,
                 dtypes: Optional[Union[Tuple[str], str]] = 'real', is_optional=False,
                 can_vary: bool = True, can_effective: bool = False, can_arrhenius: bool = False,
                 can_ref_temperature: bool = False, can_hysteresis: bool = False,
                 data_path: Optional[str] = None, description: Optional[str] = None,
                 aliases: Union[list, tuple] = [], is_user_input: bool = True):
        self.name = name
        self.component = component
        self.dic = dic
        self.is_optional = is_optional
        self.is_user_input = is_user_input
        self.can_vary = can_vary
        self.can_effective = can_effective
        self.can_arrhenius = can_arrhenius
        self.can_ref_temperature = can_ref_temperature
        self.can_hysteresis = can_hysteresis
        allowed_dtypes = ('real', 'expression', 'spline', 'label')
        if dtypes is None:
            self.allowed_dtypes = allowed_dtypes[:-1]
        else:
            if not isinstance(dtypes, (tuple,)):
                dtypes = (dtypes,)
            for dtype in dtypes:
                if dtype not in allowed_dtypes:
                    raise ValueError(self._get_error_msg(
                        reason=(f"Unrecognized dtype '{dtype}'. Available dtypes are: '"
                                + "' '".join(allowed_dtypes) + "'"),
                        action='initialization'
                    ))
            self.allowed_dtypes = dtypes
        self.data_path = data_path
        self.description = description
        if not isinstance(aliases, list):
            raise TypeError(self._get_error_msg(
                reason="'aliases' must be a list of strings", action='initialization'))
        self.aliases = aliases

        # Default parameter settings
        self.dtype = 'real' if 'real' in self.allowed_dtypes else self.allowed_dtypes[0]
        self.is_effective = True
        self.need_arrhenius = False
        self.is_dynamic_parameter = False
        self._mesh = None
        self._dolfinx_object = None
        self.has_hysteresis = False
        self.has_ref_temperature = False
        self.T_ref = None
        self.mesh_dependent = False
        self.units = None
        self.notes = None

        self.was_provided = False
        self._user_value_ = None
        self._ref_value_ = None
        self._value_ = None
        self._is_dependent = False

        self._parse_parameter_dict()

        if self._user_value_ is None and self.dic is not None:
            self._user_value_ = self.dic.get('value', None)
        self.was_provided = self._user_value_ is not None

    def __str__(self) -> str:
        # TODO: print the information of the parameter also
        return f"{self.component.complete_tag}.{self.name}"

    def _parse_parameter_dict(self):
        """Parse the user json"""
        # Parse dic
        if self.dic is None and (self.is_optional or not self.is_user_input):
            # NOTE: Either the parameter is optional or a value will be provided later
            return
        elif isinstance(self.dic, str) and 'label' in self.allowed_dtypes:
            self.dtype = 'label'
            self._value_ = self.dic
            self._ref_value_ = self.dic
            self._user_value_ = self.dic
            return
        elif not isinstance(self.dic, dict):
            raise TypeError(self._get_error_msg(
                reason=f"'dic' field must be a dictionary, not {type(self.dic)}",
                action='initialization'
            ))
        # Parse user value
        if 'value' not in self.dic:
            raise KeyError(self._get_error_msg(
                reason="Missing 'value' mandatory field", action='initialization'))
        elif self.dic['value'] is None:
            # NOTE: A parameter which value will be set later, real with no correction needed
            return
        value = self.dic['value']

        # Parse dtype
        if 'type' in self.dic:
            dtype = self.dic['type']
        elif isinstance(value, (str, dict)):
            dtype = 'expression'
        elif isinstance(value, (float, int)) or isinstance_dolfinx(value):
            dtype = 'real'
        else:
            raise TypeError(self._get_error_msg(
                reason="Unable to recognized the dtype", action='initialization'))

        if dtype not in self.allowed_dtypes:
            raise TypeError(self._get_error_msg(
                reason=(f"dtype '{dtype}' not allowed. Available options: '"
                        + "' '".join(self.allowed_dtypes) + "'"),
                action='initialization'
            ))
        else:
            self.dtype = dtype

        # Parse effective
        self.is_effective = self.dic.get('effective', True)
        if not self.is_effective and not self.can_effective:
            raise ValueError(self._get_error_msg(
                reason="'effective' field is not allowed", action='initialization'))

        # Parse arrhenius
        arrhenius = self.dic.get('arrhenius', None)
        arrhenius_keys = ('activationEnergy', 'referenceTemperature')
        if arrhenius is None:
            pass
        elif not self.can_arrhenius:
            raise ValueError(self._get_error_msg(
                reason="'arrhenius' field is not allowed", action='initialization'))
        elif (not isinstance(arrhenius, dict)
              or any(k not in arrhenius_keys for k in arrhenius.keys())):
            raise ValueError(self._get_error_msg(
                reason=("'arrhenius' field must be a dictionary containing '"
                        + "' '".join(arrhenius_keys) + "'"),
                action='initialization'
            ))
        else:
            self.need_arrhenius = True
            self.arrhenius = arrhenius

        # Parse reference temperature
        self.T_ref = self.dic.get('referenceTemperature', None)
        self.has_ref_temperature = self.T_ref is not None
        if not self.can_ref_temperature and self.has_ref_temperature:
            raise KeyError(self._get_error_msg(
                reason="'referenceTemperature' field not allowed", action='initialization'))

        # Parse dynamic parameter
        vary = self.dic.get('vary', False)
        if not vary:
            self.is_dynamic_parameter = False
        elif not self.can_vary:
            raise ValueError(self._get_error_msg(
                reason="'vary' field is not allowed", action='initialization'))
        elif self.dtype in ('spline', 'expression'):
            raise ValueError(self._get_error_msg(
                reason=f"Parameter of dtype '{self.dtype}' can't be defined as dynamic parameter",
                action='initialization'))
        else:
            self.is_dynamic_parameter = True

        # Parse information fields
        self.unit = self.dic.get('unit', None)  # TODO: Parse units
        self.notes = self.dic.get('notes', None)

        # Parse splines
        spline_keys = ('not-a-knot', 'Akima1D')
        self.spline_type = self.dic.get('spline_type', None)
        source_keys = ('file',)
        self.source = self.dic.get('source', None)
        if self.dtype == 'spline':
            if not self.is_effective or self.need_arrhenius:
                raise NotImplementedError(self._get_error_msg(
                    "Implement additional preprocessing steps for spline type parameters",
                    action='initialization'
                ))
            elif self.spline_type is None:
                raise KeyError(self._get_error_msg(
                    reason="'spline_type' key is missing", action='initialization'))
            elif self.spline_type not in spline_keys:
                raise ValueError(self._get_error_msg(
                    reason=(f"Unrecognized spline type '{self.spline_type}'. "
                            + "Available options: '" + "' '".join(spline_keys) + "'"),
                    action='initialization'
                ))
            if self.source is None:
                self.source = 'file'
                # raise KeyError(self._get_error_msg(
                #     reason="'source' key is missing", action='initialization'))
            elif self.source not in source_keys:
                raise ValueError(self._get_error_msg(
                    reason=(f"Unrecognized source '{self.source}'. "
                            + "Available options: '" + "' '".join(source_keys) + "'"),
                    action='initialization'
                ))
            # Get value from source
            if self.source == 'file':
                if self.data_path is None:
                    raise ValueError(self._get_error_msg(
                        reason="'data_path' input is missing", action='initialization'))

                # Get numpy.loadtxt kwargs
                valid_kwargs = np.loadtxt.__code__.co_varnames[1:]
                loadtxt_kwargs = {k: v for k, v in self.dic.items() if k in valid_kwargs}

                # Load file data and check if there are hysteresis
                if isinstance(value, dict):
                    if not self.can_hysteresis:
                        raise ValueError(self._get_error_msg(
                            reason="It cannot have hysteresis", action='initialization'))
                    elif not all(key in value for key in ['charge', 'discharge']):
                        raise ValueError(self._get_error_msg(
                            reason=("'value' dictionary does not contains the mandatory keys "
                                    + "'charge' and 'discharge' to identify hysteresis"),
                            action='initialization'
                        ))

                    self._user_value_ = dict()
                    for k, filename in value.items():
                        filepath = os.path.join(self.data_path, filename)
                        self._user_value_[k] = np.loadtxt(
                            filepath, **{**loadtxt_kwargs, 'dtype': float})
                    self.has_hysteresis = True
                else:
                    filename = value
                    filepath = os.path.join(self.data_path, filename)
                    self._user_value_ = np.loadtxt(filepath, **{**loadtxt_kwargs, 'dtype': float})

        elif self.dtype in ('real', 'expression'):
            if isinstance(value, dict):
                if self.dtype == 'real':
                    raise TypeError(self._get_error_msg(
                        reason="'value' can not be a dictionary for parameters of dtype 'real'",
                        action='initialization'
                    ))
                elif not self.can_hysteresis:
                    raise ValueError(self._get_error_msg(
                        reason="It cannot have hysteresis", action='initialization'))
                elif not all(key in value for key in ['charge', 'discharge']):
                    raise ValueError(self._get_error_msg(
                        reason=("'value' dictionary does not contains the mandatory keys "
                                + "'charge' and 'discharge' to identify hysteresis"),
                        action='initialization'
                    ))
                self.has_hysteresis = True
            elif isinstance(value, str):
                # TODO: Perform additional checks like if it is a valid expression, maybe require
                # state variables to be passed
                pass

    def setup_dynamic_parameter(self, mesh):
        """
        This method set the mesh associated to the dynamic parameter.
        """
        if not self.is_dynamic_parameter:
            raise RuntimeError(self._get_error_msg(
                reason="Attempted to set a mesh to a non dynamic parameter", action='handling'))
        elif any([v is not None for v in [self._value_, self._dolfinx_object]]):
            raise RuntimeError(self._get_error_msg(
                reason="Attempted to set a mesh after setting up the parameter",
                action='handling'
            ))
        elif self._mesh is not None:
            raise RuntimeError(self._get_error_msg(reason="Mesh already set!", action='handling'))
        self._mesh = mesh
        self._dolfinx_object = dfx.fem.Constant(mesh, ScalarType(0.))

    def set_value(self, value):
        """
        This method set the value of the parameter. Tipically used to
        set the default value of optional parameters if they have not
        been provided, for internal parameters or even to allow the user
        modifing its own inputs.

        Parameters
        ----------
        value : Any
            Value of the parameter.
        """
        if self._value_ is not None or self._ref_value_ is not None:
            raise RuntimeError(self._get_error_msg(
                reason="Parameter value can only be set before setting up the parameter",
                action='handling'
            ))
        elif self.dic is not None:
            self.dic['value'] = value
        elif isinstance(value, dict):
            self.dic = value
        elif isinstance(value, str):
            if self.dtype == 'label':
                self.dic = {'value': value, 'type': 'label'}
            elif 'expression' in self.allowed_dtypes:
                # TODO: Check dependencies
                self.dic = {'value': value, 'type': 'expression'}
            else:
                raise RuntimeError(self._get_error_msg(
                    reason=(f"Unable to set the value '{value}' to a "
                            + f"parameter of type '{self.dtype}'"),
                    action='handling'
                ))
        elif ('real' in self.allowed_dtypes
              and (isinstance(value, (float, int)) or isinstance_dolfinx(value))):
            self.dic = {'value': value, 'type': 'real'}
        else:
            raise RuntimeError(self._get_error_msg(
                reason="Unable to update the value", action='handling'))

        self._user_value_ = self.dic.get('value', None)
        self.was_provided = True
        self._parse_parameter_dict()

    def get_reference_value(self, eps=None, brug=None, tau=None, **kwargs):
        # TODO: called by NondimensionalModel, it should preproccess the parameter with the given
        #       set of reference values for physical vars (as can not be unscaled yet)
        if self._ref_value_ is not None:
            if eps is not None or kwargs:
                raise RuntimeError(self._get_error_msg(
                    reason='referece value has already been set up', action='handling'))
        else:
            # TODO: If the given state variables are list of values and self.dtype == 'expression',
            #       then compute the average value
            self._ref_value_ = self._setup(eps, brug, tau, return_fenics=False, **kwargs)
        return self._ref_value_

    def get_value(self, problem=None, eps=None, brug=None, tau=None, **kwargs):
        # TODO: called by BaseCellComponent, it should preproccess the parameter with the given
        #       set of physical vars (already defined and unscaled)
        if self._value_ is not None:
            if eps is not None or kwargs:
                raise RuntimeError(self._get_error_msg(
                    reason='value has already been set up', action='handling'))
        else:
            if problem is not None and self.dtype == 'expression':
                kwargs = {**problem._vars.f_1._asdict(), **kwargs}
            self._value_ = self._setup(eps, brug, tau, **kwargs)
        return self._value_

    def update(self, value):
        """This method update the value of the dynamic parameter"""
        if not self.is_dynamic_parameter:
            raise RuntimeError(self._get_error_msg(
                reason=f"It has not been defined as a dynamic parameter", action='handling'))
        elif not isinstance(value, (float, int)):
            # NOTE: value is a dolfinx_object. Set it if the parameter has not been already set up
            if self._value_ is None:
                # Update the current value of the parameter
                self._dolfinx_object = value
                self._is_dependent = True
            else:
                raise RuntimeError(self._get_error_msg(
                    reason=("Unable to update the dynamic parameter as a new dolfinx object after "
                            + "setting it up"),
                    action='handling'
                ))
        elif self._dolfinx_object is None:
            # NOTE: _dolfinx_object has not been already defined as a dolfinx.fem.Constant,
            #       as the mesh has not been already created
            raise RuntimeError(self._get_error_msg(
                reason=("Unable to update the dynamic parameter before setting it up"),
                action='handling'
            ))
        else:
            # Update the current value of the parameter
            self._dolfinx_object.value = value

        # Update the user value and reset the reference value to be recomputed later
        self.dic['value'] = value
        self._user_value_ = value
        self._ref_value_ = None

    def get_arrhenius(self):
        """
        Get arrhenius parameters if given

        Returns
        -------
        float, int
            Activation energy
        float, int
            Reference temperature

        Raises
        ------
        RuntimeError
            'arrhenius' field has not been specified
        """
        if self.arrhenius is None:
            raise RuntimeError(self._get_error_msg(
                reason="'arrhenius' field has not been specified", action='handling'))
        else:
            return self.arrhenius['activationEnergy'], self.arrhenius['referenceTemperature']

    def get_reference_temperature(self):
        """
        Get the reference temperature if given

        Returns
        -------
        float, int
            Reference temperature

        Raises
        ------
        RuntimeError
            'referenceTemperature' field has not been specified
        """
        if not self.has_ref_temperature:
            raise RuntimeError(self._get_error_msg(
                reason="'referenceTemperature' field has not been specified", action='handling'))
        else:
            return self.T_ref

    def make_reference_temperature_mandatory(self):
        """
        This method makes this parameter has a reference temperature.
        """
        if not self.has_ref_temperature:
            raise KeyError(self._get_error_msg(
                reason="'referenceTemperature' field is missing", action='initialization'))
        else:
            return self.has_ref_temperature

    def make_mandatory(self):
        """This method makes this parameter mandatory"""
        if self.is_optional:
            if not self.was_provided:
                raise KeyError(self._get_error_msg(
                    reason="Parameter not found", action='initialization'))
            else:
                self.is_optional = False

    def make_dynamic_parameter(self):
        """
        This method makes this parameter be a dynamic parameter. The
        parameter's dtype must be `real`. This method must be called
        before setting up this parameter.
        """
        if self.is_dynamic_parameter:
            return
        elif self.dtype != 'real':
            raise RuntimeError(self._get_error_msg(
                reason=f"Attempted to make dynamic a parameter of dtype '{self.dtype}'",
                action="handling"
            ))
        elif self._value_ is not None:
            raise RuntimeError(self._get_error_msg(
                reason="Attempted to make dynamic a parameter after setting it up",
                action='handling'
            ))
        elif not self.can_vary:
            raise RuntimeError(self._get_error_msg(
                reason=f"Attempted to make dynamic a parameter that cannot vary",
                action="handling"
            ))
        else:
            self.is_dynamic_parameter = True
            self.component._dynamic_parameters_[self.name] = self

    def make_mesh_dependent(self, value: dfx.fem.Function, ref_value: Union[float, int]):
        """
        This method makes this parameter be mesh dependent. The
        parameter's dtype must be `real` and no further preprocessing
        should be needed.

        Parameters
        ----------
        value : dfx.fem.Function
            Value of the parameter.
        ref_value : Union[float, int]
            Reference value to be used.
        """
        if self.dtype != 'real':
            raise RuntimeError(self._get_error_msg(
                reason=f"Attempted to make mesh dependent a parameter of dtype '{self.dtype}'",
                action="handling"
            ))
        if (not self.is_effective or self.need_arrhenius
                or self.is_dynamic_parameter or self.has_hysteresis):
            raise RuntimeError(self._get_error_msg(
                reason=f"Attempted to make mesh dependent a parameter that needs preprocessing",
                action="handling"
            ))
        self.mesh_dependent = True
        self.was_provided = True
        self._user_value_ = ref_value
        self._ref_value_ = ref_value
        self._value_ = value

    def reset(self):
        """
        This method clear the preprocessed values in order to recompute
        them again with new parameters or problem.
        """
        self._value_ = None
        self._ref_value_ = None
        self._dolfinx_object = None
        self._mesh = None

    def _setup(self, eps=None, brug=None, tau=None, R=8.314472, return_fenics=True, **kwargs):
        """
        This method perform the preprocessing steps to set up the cell
        parameter, such as the evaluation of non linear properties, the
        application of corrections (arrhenius, effective, etc.) or
        defining dynamic parameters.

        Parameters
        ----------
        eps: Union[float, dolfinx.fem.Constant, dolfinx.fem.Function]
            Porosity of the component to compute the effective parameter
        brug: Union[float, dolfinx.fem.Constant, dolfinx.fem.Function]
            Bruggeman exponent compute the effective parameter
        tau: Union[float, dolfinx.fem.Constant, dolfinx.fem.Function]
            Tortuosity to compute the effective parameter
        kwargs: dict
            Dictionary containing the expression dependencies
        """
        value = self._user_value_
        if value is None:
            if not self.is_optional:
                raise ValueError(self._get_error_msg(
                    "A value has not been provided yet", action='setup'))
            elif self.dic is not None and self.dic.get('value', None) is not None:
                # NOTE: In this case, self.set_value has been called
                value = self.dic['value']
            else:
                # NOTE: A default value has not been set yet, maybe it is not necessary
                return value
        elif not return_fenics and isinstance_dolfinx(value):
            if self._is_dependent:
                value = _evaluate_parameter(value)
            else:
                raise ValueError(self._get_error_msg(
                    f"Invalid user value '{value}'", action='setup'))

        # Parse dynamic parameter
        if return_fenics and self.is_dynamic_parameter:
            if isinstance(self._dolfinx_object, dfx.fem.Constant):
                self._dolfinx_object.value = value
            elif self._dolfinx_object is None:
                raise ValueError(self._get_error_msg(
                    reason="Dynamic parameter has not been associated to an specific mesh yet",
                    action='setup'
                ))
            value = self._dolfinx_object

        # Parse expression
        elif self.dtype == 'expression':
            if self.has_hysteresis:
                # NOTE: return a function whose inputs are stoichiometry and current
                value = hysteresis_property({
                    'charge': lambda y: constant_expression(value['charge'], return_fenics,
                                                            y=y, **kwargs),
                    'discharge': lambda y: constant_expression(value['discharge'], return_fenics,
                                                               y=y, **kwargs),
                })
                return value  # TODO: Allow to apply additional corrections
            elif self.can_hysteresis:
                # NOTE: 'current' kwarg is expected for this parameters
                expression = lambda y: constant_expression(value, return_fenics, y=y, **kwargs)
                value = hysteresis_property({'charge': expression, 'discharge': expression})
                return value  # TODO: Allow to apply additional corrections
            else:
                value = constant_expression(value, return_fenics, **kwargs)

        # Parse spline
        elif self.dtype == 'spline':
            if self.has_hysteresis:
                value = hysteresis_property({
                    'charge': get_spline(value['charge'], self.spline_type, return_fenics),
                    'discharge': get_spline(value['discharge'], self.spline_type, return_fenics),
                })
            else:
                spline = get_spline(value, self.spline_type, return_fenics)
                if self.can_hysteresis:
                    # FIXME: hysteresis_property is called to allow additional kwarg 'current'
                    value = hysteresis_property({'charge': spline, 'discharge': spline})
                else:
                    value = spline
            # TODO: Allow spline parameters to apply additional corrections
            return value

        # Preprocess non effective values
        if not self.is_effective:
            value = self._effective_value(value, eps, brug, tau)

        # Apply arrhenius correction
        if self.need_arrhenius:
            if 'temp' not in kwargs:
                raise ValueError(self._get_error_msg(
                    "Unable to apply the arrhenius correction, temperature has not been specified",
                    action='setup'
                ))
            value = self._apply_arrhenius(value, R, kwargs['temp'], return_fenics)

        return value

    def _effective_value(self, value, eps=None, brug=None, tau=None):
        if eps is None:
            raise ValueError(self._get_error_msg(
                "Unable to compute the effective value, porosity has not been provided",
                action='setup'
            ))
        elif brug is not None:
            return value * eps**brug
        elif tau is not None:
            return value * eps / tau
        else:
            raise ValueError(self._get_error_msg(
                ("Unable to apply effective parameter corrections. "
                 + "Neither tortuosity nor porosity has been provided"),
                action='setup'
            ))

    def _apply_arrhenius(self, value, R, T, return_fenics=True):
        """Apply Arrhenius correction."""
        Ea, T_ref = self.arrhenius['activationEnergy'], self.arrhenius['referenceTemperature']
        if Ea == 0:
            return value
        elif return_fenics:
            return value * ufl.exp((Ea / R) * (1 / T_ref - 1 / T))
        else:
            return value * np.exp((Ea / R) * (1 / T_ref - 1 / T))

    def _get_error_msg(self, reason, action='handling'):
        # TODO: Make a rework, maybe implement internal stages within self._stage_
        #       to avoid the 'action' kwarg
        if action == 'initialization':
            return f"Error initializing the parameter '{self}': {reason}"
        elif action == 'setup':
            return f"Error setting up the value of the parameter '{self}': {reason}"
        elif action == 'handling':
            return f"Error handling the parameter '{self}': {reason}"
        else:
            return reason
