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
import numpy as np
import dolfinx as dfx
from petsc4py.PETSc import ScalarType
from pydantic import BaseModel, validator

from cideMOD.helpers.logging import VerbosityLevel, _print
from cideMOD.cell._factory import get_cell_component_label
from cideMOD.cell.parser import CellParser, BaseComponentParser
from cideMOD.models import model_factory, register_model_options, register_trigger, __mtypes__
from cideMOD.models.model_options import BaseModelOptions
from cideMOD.models.PXD.base_model import BasePXDModelInputs
from cideMOD.models.PXD.electrochemical import __model_name__

register_trigger(name='voltage', label='v', units='V', atol=1e-4)
register_trigger(name='current', label='i', units='A', need_abs=True, atol=1e-6)


@register_model_options(__model_name__)
class ElectrochemicalModelOptions(BaseModel):
    """
    Electrochemical Model
    ---------------------
    particle_coupling: str
        Coupling between cell and particle problem. Available options:
        `implicit`, `explicit`. Default to `implicit`.
    particle_model: int
        Particle model to be used. Available models:
        - `SGM`: Spectral Galerkin Model. Default option.
    """
    model: str = 'P2D'
    particle_coupling: str = 'implicit'
    particle_model: str = 'SGM'
    time_scheme: str = 'euler_implicit'  # TODO: Make this useful

    @validator('particle_coupling')
    def validate_particle_coupling(cls, v):
        if v not in ('implicit', 'explicit'):
            raise ValueError("'particle_coupling' must be 'implicit' or 'explicit'")
        elif v == 'explicit':
            raise NotImplementedError(f"The explicit particle model is not available yet")
        return v

    @validator('model')
    def validate_model(cls, v):
        pxd_mtypes = __mtypes__['PXD']
        if v not in pxd_mtypes:
            raise ValueError("'model' keyword must be one of: '" + "' '".join(pxd_mtypes) + "'")
        return v

    @validator('particle_model')
    def validate_particle_model(cls, v):
        particle_models = [name for name in model_factory() if name.startswith('PM_')]
        if f"PM_{v}" not in particle_models:
            raise ValueError(f"Unrecognized particle model '{v}'. Available particle models: '"
                             + "' '".join(particle_models) + "'")
        return v


class ElectrochemicalModelInputs(BasePXDModelInputs):

    # ******************************************************************************************* #
    # ***                                    ModelOptions                                     *** #
    # ******************************************************************************************* #

    @classmethod
    def is_active_model(cls, model_options: BaseModelOptions) -> bool:
        """
        This method checks the model options configured by the user to
        evaluate if this model should be added to the cell model.

        Parameters
        ----------
        model_options: BaseModelOptions
            Model options already configured by the user.

        Returns
        -------
        bool
            Whether or not this model should be added to the cell model.
        """
        # TODO: Ensure that model_options has been extended with ElectrochemicalModelOptions
        return True

    # ******************************************************************************************* #
    # ***                                       Problem                                       *** #
    # ******************************************************************************************* #

    def set_cell_state(self, problem, SoC=None, T_ext=None, T_ini=None) -> None:
        """
        This method set the current state of the cell.

        Parameters
        ----------
        problem: Problem
            Object that handles the battery cell simulation.
        SoC: float, optional
            Current State of Charge of the battery cell. Default initial
            value to 1.
        T_ext: float, optional
            External temperature. Default initial value to 298,15 K.
        T_ini: float, optional
            Uniform value of the internal temperature. Default initial
            value to `T_ext`.
        """

        if not problem._ready:
            # The user is setting the initial cell state
            problem.SoC_ini = SoC if SoC is not None else 1

            T_ext = T_ext if T_ext is not None else 298.15
            if self._T_ext is None:
                self._T_ext = dfx.fem.Constant(problem.mesher.mesh, ScalarType(T_ext))
                problem.T_ext = self._T_ext
            else:
                problem.T_ext.value = T_ext

            T_ini = T_ini if T_ini is not None else T_ext
            if self._T_ini is None:
                self._T_ini = dfx.fem.Constant(problem.mesher.mesh, ScalarType(T_ini))
                problem.T_ini = self._T_ini
            else:
                problem.T_ini.value = T_ini

        else:
            # TODO: Think about creating a method called set_new_cell_state
            # Setting a new cell state
            if SoC is not None and not np.isclose(SoC, problem.SoC_ini):
                raise NotImplementedError(
                    "SoC is used by the initial guess. Set c_s directly to set a new cell state")
            if T_ext is not None:
                problem.T_ext.value = T_ext
            if T_ini is not None:
                problem.T_ini.value = T_ini

    # ******************************************************************************************* #
    # ***                                     CellParser                                      *** #
    # ******************************************************************************************* #

    def build_cell_components(self, cell: CellParser) -> None:
        """
        This method builds the components of the cell that fit our model
        type, e.g. electrodes, separator, current collectors, etc.

        Parameters
        ----------
        cell: CellParser
            Parser of the cell dictionary.

        Examples
        --------
        >>> cell.set_component('anode')

        It is also possible to create the class dinamically:

        >>> cell.set_component('anode')
        """
        # NOTE: Notice that ordering matters, in this case some components will need some
        #       parameters from the electrolyte, that's why it is built first.
        cell.electrolyte = cell.set_component('electrolyte')
        cell.anode = cell.set_component('anode')
        cell.cathode = cell.set_component('cathode')
        cell.separator = cell.set_component('separator')
        # Check if there are collectors
        ncc_tag = get_cell_component_label('negativeCC')
        pcc_tag = get_cell_component_label('positiveCC')
        cell.has_collectors = ncc_tag in cell.structure or pcc_tag in cell.structure

        cell.negativeCC = cell.set_component('negativeCC') if cell.has_collectors else None
        cell.positiveCC = cell.set_component('positiveCC') if cell.has_collectors else None

    def parse_cell_structure(self, cell: CellParser):
        """
        This method parse the cell structure. If there are any component
        this model does not know, then this method should return the
        list of unrecognized components. Maybe this components has been
        defined by other models, so this task should be delegated to
        these model.

        Parameters
        ----------
        cell: CellParser
            Parser of the cell dictionary.

        Returns
        -------
        Union[bool, list]
            Whether or not the cell structure is valid. If there are any
            component this model does not know, then this method should
            return the list of unrecognized components.
        """

        # Get recognized component labels
        if cell.has_collectors:
            recognized_components = [cell.negativeCC, cell.anode, cell.separator,
                                     cell.cathode, cell.positiveCC, cell.electrolyte]
        else:
            recognized_components = [cell.anode, cell.separator, cell.cathode, cell.electrolyte]
        recognized_labels = [component._label_ for component in recognized_components[:-1]]
        unrecognized_labels = set(cell.structure).difference(recognized_labels)

        # If there are unrecognized labels, return them
        if unrecognized_labels:
            return list(unrecognized_labels)

        # Minimum cell structure length
        if len(cell.structure) <= 2:
            raise ValueError("The cell structure should be composed at least by "
                             + "an anode, a separator and a cathode")

        # Check the mandatory cell structure components
        mandatory_components = [cell.anode, cell.separator, cell.cathode]
        for component in mandatory_components:
            if component._label_ not in cell.structure:
                raise ValueError(f"Component '{component._name_}' with label '{component._label}' "
                                 + "is missing in the cell structure")

        # Check cell structure
        def _check_component(idx, component, valid_components=[], different=True, extreme=False):
            valids = [valid_component._label_ for valid_component in valid_components]

            # Check previous element
            prev = cell.structure[idx - 1] if idx > 0 else None
            if prev is not None and prev not in valids:
                prev_name = valid_components[valids.index(prev)]._name_
                raise ValueError(f"Component '{prev_name}' cannot be next to '{component._name_}'")

            # Check next element
            next_ = cell.structure[idx + 1] if idx + 1 < len(cell.structure) else None
            if next_ is not None and next_ not in valids:
                next_name = valid_components[valids.index(next_)]._name_
                raise ValueError(f"Component '{next_name}' cannot be next to '{component._name_}'")

            # Final checks
            if not extreme and idx in [0, len(cell.structure) - 1]:
                raise ValueError(f"Component '{component._name_}' cannot be "
                                 + "at the extreme of the cell structure")
            elif different and None not in [prev, next_] and prev == next_:
                raise ValueError(f"Bad element in structure: {(prev, component._label_, next_)}")

        if cell.has_collectors:
            for idx, label in enumerate(cell.structure):
                component = recognized_components[recognized_labels.index(label)]
                if label == cell.negativeCC._label_:
                    _check_component(idx, component, [cell.anode], different=False, extreme=True)
                elif label == cell.anode._label_:
                    _check_component(idx, component, [cell.negativeCC, cell.separator])
                elif label == cell.separator._label_:
                    _check_component(idx, component, [cell.anode, cell.cathode])
                elif label == cell.cathode._label_:
                    _check_component(idx, component, [cell.positiveCC, cell.separator])
                elif label == cell.positiveCC._label_:
                    _check_component(idx, component, [cell.cathode], different=False, extreme=True)
        else:
            for idx, label in enumerate(cell.structure):
                component = recognized_components[recognized_labels.index(label)]
                if label == cell.anode._label_:
                    _check_component(idx, component, [cell.separator], extreme=True)
                elif label == cell.separator._label_:
                    _check_component(idx, component, [cell.anode, cell.cathode])
                elif label == cell.cathode._label_:
                    _check_component(idx, component, [cell.separator], extreme=True)
        return True

    def _parse_component_parameters(self, component: BaseComponentParser, default_area=None):

        component.set_parameters(__cell_parameters__['component'])

        if not component.area.was_provided:
            if component.width.was_provided and component.height.was_provided:
                component.area.set_value(component.width.user_value * component.height.user_value)
            elif default_area is not None:
                component.area.set_value(default_area)
            else:
                raise KeyError(component.area._get_error_msg(
                    reason="Parameter not found", action='initialization'))
        elif component.width.was_provided and component.height.was_provided:
            if not np.isclose(component.width.user_value * component.height.user_value,
                              component.area.user_value, rtol=1e-3):
                raise ValueError(
                    f"Error parsing '{component.complete_tag}'. The given height*width != area")

    def _parse_porous_component_parameters(self, component: BaseComponentParser):
        self._parse_component_parameters(component)
        component.set_parameters(__cell_parameters__['porous_component'])
        if not component.bruggeman.was_provided and not component.tortuosity_e.was_provided:
            raise KeyError(f"'{component.tag}' bruggeman or tortuosity must be provided")

        # Set the porous parameters info within the electrolyte component
        electrolyte = component.cell.electrolyte
        component.D_e.set_value(electrolyte.diffusion_constant.dic)
        component.kappa.set_value(electrolyte.ionic_conductivity.dic)

    def parse_cell_parameters(self, cell: CellParser) -> None:
        """
        This methods parses the cell parameters of the electrochemical
        model.

        Parameters
        ----------
        cell: CellParser
            Parser of the cell dictionary.
        """
        cell.set_parameters(__cell_parameters__['cell'])

    def parse_electrode_parameters(self, electrode: BaseComponentParser) -> None:
        """
        This method parses the electrode parameters of the
        electrochemical model.

        Parameters
        ----------
        electrode: BaseComponentParser
            Object that parses the electrode parameters.
        """
        self._parse_porous_component_parameters(electrode)
        electrode.set_parameters(__cell_parameters__['electrode'])
        if electrode.type.user_value != 'porous':
            raise NotImplementedError(f"Only 'porous' electrodes available in this version")

    def parse_active_material_parameters(self, am: BaseComponentParser) -> None:
        """
        This method parses the active material parameters of the
        electrochemical model.

        Parameters
        ----------
        am: BaseComponentParser
            Object that parses the active material parameters.
        """
        am.material = am.set_parameter(
            'material', is_optional=True, default=am.tag, dtypes='label')
        am.set_parameters(__cell_parameters__['active_material'])

        # Setup parameters that are not provided by the user
        am.porosity.set_value(1)

        # Additional checks
        if not am.volume_fraction.was_provided:
            required = [am.density, am.electrode.density, am.mass_fraction]
            if not all([p.was_provided for p in required]):
                raise ValueError(am.volume_fraction._get_error_msg(
                    reason=("Unable to compute it and it has not been provided. Required "
                            + "parameters: '" + "' '".join([str(p) for p in required]) + "'"),
                    action='initialization'
                ))
            elif not am.electrode.density.is_effective:
                raise ValueError(am.volume_fraction._get_error_msg(
                    reason=("Unable to compute it from the mass fraction if "
                            + f"{str(am.electrode.density)} is not effective"),
                    action='initialization'
                ))
            else:
                mass_fraction = am.mass_fraction.user_value
                rho_el = am.electrode.density.user_value
                rho_am = am.density.user_value
                am.volume_fraction.set_value(mass_fraction * rho_el / rho_am)

        if am.entropy_coefficient.was_provided:
            am.ocp.make_reference_temperature_mandatory()

        # Add the contribution of this inclusion (if so) to the porosity of the active material
        if 'inc' in am.electrode.tag:
            eps_am = am.electrode.porosity
            if eps_am.is_dynamic_parameter:
                raise NotImplementedError(eps_am._get_error_msg(
                    reason=f"It can't be defined as dynamic parameter yet",
                    action='initialization'))
            eps_am.set_value(eps_am.user_value - am.volume_fraction.user_value)
            if not 0 <= eps_am.user_value <= 1:
                raise RuntimeError(f"Parameter '{eps_am}' has a value of {eps_am.user_value}")

    def parse_separator_parameters(self, separator: BaseComponentParser) -> None:
        """
        This method parses the separator parameters of the
        electrochemical model.

        Parameters
        ----------
        separator: BaseComponentParser
            Object that parses the separator parameters.
        """
        self._parse_porous_component_parameters(separator)
        separator.set_parameters(__cell_parameters__['separator'])
        if separator.type.user_value != 'porous':
            raise NotImplementedError(f"Only 'porous' separator available in this version")

    def parse_current_collector_parameters(self, cc: BaseComponentParser) -> None:
        """
        This method parses the current collector parameters of the
        electrochemical model.

        Parameters
        ----------
        cc: BaseComponentParser
            Object that parses the current collector parameters.
        """
        self._parse_component_parameters(cc)
        cc.set_parameters(__cell_parameters__['current_collector'])
        if cc.type.user_value != 'solid':
            raise NotImplementedError(f"Current collectors must be 'solid'.")

    def parse_electrolyte_parameters(self, electrolyte: BaseComponentParser) -> None:
        """
        This method parses the electrolyte parameters of the
        electrochemical model.

        Parameters
        ----------
        electrolyte: BaseComponentParser
            Object that parses the electrolyte parameters.
        """

        electrolyte.set_parameters(__cell_parameters__['electrolyte'])

        if electrolyte.type.get_value() != 'liquid':
            raise ValueError("Solid electrolytes not supported in this version")

        if electrolyte.intercalation_type.get_value() != 'binary':
            raise ValueError("Only binary electrolytes are supported in this version")

        # Perform some hacking actions
        # NOTE: The electrolyte itself is not porous, but we will use the same dictionary to
        #       build the corresponding porous component parameters.
        electrolyte.diffusion_constant.is_effective = True
        electrolyte.ionic_conductivity.is_effective = True

    def compute_reference_cell_properties(self, cell: CellParser):
        """
        This method computes the general reference cell properties of
        the electrochemical model.

        Parameters
        ----------
        cell: CellParser
            Parser of the cell dictionary.

        Notes
        -----
        This method is called once the cell parameters has been parsed.
        """
        cell.anode.ref_capacity = self._get_reference_electrode_capacity(cell.anode)
        cell.cathode.ref_capacity = self._get_reference_electrode_capacity(cell.cathode)
        cell.ref_capacity = min(cell.anode.ref_capacity or 9e99, cell.cathode.ref_capacity or 9e99)

        if cell.verbose >= VerbosityLevel.BASIC_PROBLEM_INFO:
            _print(f"Negative electrode capacity: {cell.anode.ref_capacity:.6f}", comm=cell._comm)
            _print(f"Positive electrode capacity: {cell.cathode.ref_capacity :.6f}",
                   comm=cell._comm)
            _print(f"Cell capacity: {cell.ref_capacity:.6f}", comm=cell._comm)

        cell.ref_area = min(cell.anode.area.get_reference_value() or 9e99,
                            cell.cathode.area.get_reference_value() or 9e99)

        components = [v for k, v in cell._components_.items() if k != 'electrolyte']
        if any([element.height.was_provided for element in components]):
            cell.ref_height = max([element.height.get_reference_value()
                                   for element in components
                                   if element.height.get_reference_value()])
        else:
            cell.ref_height = None

        if any([element.width.was_provided for element in components]):
            cell.ref_width = max([element.width.get_reference_value()
                                  for element in components
                                  if element.width.get_reference_value()])
        else:
            cell.ref_width = None

    def _get_reference_electrode_capacity(self, electrode: BaseComponentParser):
        area = electrode.area.get_reference_value()
        L = electrode.thickness.get_reference_value()
        F = electrode.cell.F.get_reference_value()
        cap = 0
        for am in electrode.active_materials:
            am_eps_s = am.volume_fraction.get_reference_value()
            am_porosity = am.porosity.get_reference_value()
            am_c_s_max = am.maximum_concentration.get_reference_value()
            am_stoichiometry0 = am.stoichiometry0.get_reference_value()
            am_stoichiometry1 = am.stoichiometry1.get_reference_value()
            cap += (am_eps_s * am_porosity * am_c_s_max
                    * abs(am_stoichiometry1 - am_stoichiometry0) / 3600)
            for inc in am.inclusions:
                inc_eps_s = inc.volume_fraction.get_reference_value()
                inc_porosity = inc.porosity.get_reference_value()
                inc_c_s_max = inc.maximum_concentration.get_reference_value()
                inc_stoichiometry0 = inc.stoichiometry0.get_reference_value()
                inc_stoichiometry1 = inc.stoichiometry1.get_reference_value()
                cap += (am_eps_s * inc_eps_s * inc_c_s_max * inc_porosity
                        * abs(inc_stoichiometry1 - inc_stoichiometry0) / 3600)
        return cap * area * L * F


__cell_parameters__ = {
    'cell': {
        'R': {'element': 'constants', 'default': 8.314472, 'is_optional': True},
        'F': {'element': 'constants', 'default': 96485.3365, 'is_optional': True},
        'doubleLayerCapacitance_cc': {'element': 'properties', 'is_optional': True}
    },
    'component': {
        'thickness': {},
        'area': {'is_optional': True},
        'width': {'is_optional': True},
        'height': {'is_optional': True},
        'density': {'is_optional': True, 'aliases': 'rho'}
    },
    'porous_component': {
        'porosity': {'aliases': 'eps_e'},
        'bruggeman': {'is_optional': True, 'aliases': 'brug'},
        'tortuosity_e': {'is_optional': True, 'aliases': 'tortuosity'},
        'tortuosity_s': {'is_optional': True},
        'D_e': {'can_effective': True, 'can_arrhenius': True,
                'aliases': 'diffusionConstantElectrolyte', 'dtypes': ('real', 'expression'),
                'is_user_input': False},
        'kappa': {'can_effective': True, 'can_arrhenius': True,
                  'aliases': 'ionic_conductivity', 'dtypes': ('real', 'expression'),
                  'is_user_input': False},
        'kappa_D': {'can_vary': False, 'aliases': 'ionicConductivityDiffusion',
                    'is_user_input': False},
    },
    'electrode': {
        'type': {'is_optional': True, 'default': 'porous'},
        'electronic_conductivity': {'can_effective': True,
                                    'aliases': ['electronicConductivity']},
        'double_layer_capacitance': {'is_optional': True,
                                     'aliases': ['doubleLayerCapacitance', 'dl_capacitance']}
    },
    'active_material': {
        # 'material': {'is_optional': True, 'default': ['!self.tag'], 'dtypes': 'label'},
        'volume_fraction': {'is_optional': True, 'aliases': ['volFrac_active', 'volumeFraction']},
        'density': {'is_optional': True},
        'mass_fraction': {'is_optional': True, 'aliases': ['massFraction']},
        'particle_radius': {'aliases': ['particleRadius', 'Rp']},
        'maximum_concentration': {'aliases': ['maximumConcentration', 'c_s_max']},
        'stoichiometry0': {'aliases': ['stoi0']},
        'stoichiometry1': {'aliases': ['stoi1']},
        'kinetic_constant': {'can_arrhenius': True,
                             'dtypes': ('real', 'expression'),
                             'aliases': ['kineticConstant']},
        'alpha': {'default': 0.5, 'is_optional': True, 'aliases': ['charge_transfer_coefficient']},
        'ocp': {'dtypes': ('expression', 'spline'),
                'aliases': ['OCP', 'openCircuitPotential', 'open_circuit_potential'],
                'can_ref_temperature': True, 'can_hysteresis': True},
        'entropy_coefficient': {'dtypes': ('expression', 'spline'),
                                'aliases': ['entropyCoefficient'],
                                'is_optional': True, 'can_hysteresis': True},
        'a_s': {'is_user_input': False},
        'tortuosity_s': {'is_optional': True},
        'porosity': {'is_user_input': False}
    },
    'separator': {
        'type': {'is_optional': True, 'default': 'porous'}
    },
    'current_collector': {
        'type': {'is_optional': True, 'default': 'solid'},
        'electronic_conductivity': {'aliases': ['electronicConductivity']}
    },
    'electrolyte': {
        'type': {'default': 'liquid', 'is_optional': True, 'dtypes': 'label'},
        'intercalation_type': {'default': 'binary', 'is_optional': True, 'dtypes': 'label'},
        'transference_number': {'aliases': ['transferenceNumber', 't_p']},
        'activity_dependence': {'dtypes': ('real', 'expression'),
                                'default': 1, 'is_optional': True,
                                'aliases': ['activityDependence', '(1+d(lnf)/d(lnc))']},
        'initial_concentration': {'aliases': ['initialConcentration', 'c_0']},
        'diffusion_constant': {'dtypes': ('real', 'expression'),
                               'can_arrhenius': True, 'can_effective': True,
                               'aliases': ['D_e', 'diffusionConstant']},
        'ionic_conductivity': {'dtypes': ('real', 'expression'),
                               'can_arrhenius': True, 'can_effective': True,
                               'aliases': ['kappa', 'ionicConductivity']}
    }
}
