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

__model_name__ = 'SEI'
__mtype__ = 'PXD'
__root_model__ = False
__hierarchy__ = 300

from pydantic import BaseModel, validator
from typing import Optional

from cideMOD.helpers.logging import VerbosityLevel
from cideMOD.cell import register_cell_component
from cideMOD.cell.parser import BaseComponentParser
from cideMOD.cell.components import BaseCellComponent
from cideMOD.models import register_model_options
from cideMOD.models.model_options import BaseModelOptions
from cideMOD.models.PXD.base_model import BasePXDModel
from cideMOD.models import register_model
from cideMOD.numerics.fem_handler import BlockFunctionSpace
from cideMOD.numerics.time_scheme import TimeScheme
from cideMOD.cell.components import BatteryCell
from cideMOD.cell.equations import ProblemEquations
from cideMOD.cell.variables import ProblemVariables
from cideMOD.mesh.base_mesher import BaseMesher


class SEIParser(BaseComponentParser):

    _name_ = 'SEI'
    _root_name_ = 'electrode'
    _tags_ = {
        'SEI': {
            'label': 'SEI',
            'dict_entry': 'SEI'
        }
    }

    def __init__(self, electrode, dic, tag=None):
        super().__init__(electrode, dic, tag=tag)
        self._set_SEI_layers()

    def _set_SEI_layers(self):

        # Set porous layer
        porous_dic = self.dic.get('porous', self.dic)
        self.porous = self.PorousLayerParser(self, porous_dic, tag='porous')
        self.set_component(self.porous)

        # Set compact layer
        self.compact = None
        if 'compact' in self.dic.keys():
            compact_dic = self.dic['compact']
            self.compact = self.CompactLayerParser(self, compact_dic, tag='compact')
            self.set_component(self.compact)
        self.has_compact = bool(self.compact)

    class PorousLayerParser(BaseComponentParser):
        _name_ = 'porousSEI'
        _root_name_ = 'SEI'
        _tags_ = {
            'porous': {
                'label': 'porous',
                'dict_entry': None
            }
        }

        def __init__(self, SEI, dic, tag=None):
            super().__init__(SEI, dic, tag=tag)
            self.SEI = SEI

    class CompactLayerParser(BaseComponentParser):
        _name_ = 'compactSEI'
        _root_name_ = 'SEI'
        _tags_ = {
            'compact': {
                'label': 'compact',
                'dict_entry': None
            }
        }

        def __init__(self, SEI, dic, tag=None):
            super().__init__(SEI, dic, tag=tag)
            self.SEI = SEI

        def __bool__(self):
            return bool(self.dic)


class SEIParameters(BaseCellComponent):

    _name_ = 'SEI'
    _root_name_ = 'electrode'

    def _set_components(self):
        super()._set_components()
        # Set porous layer
        self.porous = self._components_['porous']
        self.compact = self._components_.get('compact', None)
        self.has_compact = self.parser.has_compact

    class PorousLayerParameters(BaseCellComponent):
        _name_ = 'porousSEI'
        _root_name_ = 'SEI'

        def __init__(self, SEI, parser, tag=None, verbose=VerbosityLevel.NO_INFO):
            super().__init__(SEI, parser, tag, verbose)
            self.SEI = SEI

    class CompactLayerParameters(BaseCellComponent):
        _name_ = 'compactSEI'
        _root_name_ = 'SEI'

        def __init__(self, SEI, parser, tag=None, verbose=VerbosityLevel.NO_INFO):
            super().__init__(SEI, parser, tag, verbose)
            self.SEI = SEI

        def __bool__(self):
            return bool(self.parser)


register_cell_component('SEI', parser_cls=SEIParser, component_cls=SEIParameters)
register_cell_component('porousSEI', parser_cls=SEIParser.PorousLayerParser,
                        component_cls=SEIParameters.PorousLayerParameters)
register_cell_component('compactSEI', parser_cls=SEIParser.CompactLayerParser,
                        component_cls=SEIParameters.CompactLayerParameters)


@register_model_options(__model_name__)
class BaseSEIModelOptions(BaseModel):
    """
    SEI Model
    ---------
    SEI_model: Optional[str]
        Which limiting mechanism is applied to the SEI model, one of
        `solvent_diffusion`, `electron_migration`. Use None in order to
        deactivate the SEI model. Defaults to None.
    """
    SEI_model: Optional[str] = None

    @validator('SEI_model')
    def validate_particle_coupling(cls, v):
        available_SEI_models = ('solvent_diffusion', 'electron_migration')
        if v is not None and v not in available_SEI_models:
            raise ValueError(f"Unrecognized SEI model '{v}'. Available options: '"
                             + "' '".join(available_SEI_models) + "'")
        return v


@register_model
class BaseSEIModel(BasePXDModel):

    _name_ = __model_name__
    _mtype_ = __mtype__
    _root_ = __root_model__
    _hierarchy_ = __hierarchy__

    # ******************************************************************************************* #
    # ***                             Inputs. ModelOptions                                    *** #
    # ******************************************************************************************* #

    @classmethod
    def is_active_model(cls, model_options: BaseModelOptions) -> bool:
        """
        This method checks the model options configured by the user to
        evaluate if this SEI model should be added to the cell model.

        Parameters
        ----------
        model_options: BaseModelOptions
            Model options already configured by the user.

        Returns
        -------
        bool
            Whether or not this model should be added to the cell model.
        """
        return model_options.SEI_model is not None

    # ******************************************************************************************* #
    # ***                                     CellParser                                      *** #
    # ******************************************************************************************* #

    def build_cell_components(self, cell) -> None:
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
        cell.anode.SEI = cell.anode.set_component('SEI')

    def parse_SEI_parameters(self, SEI) -> None:
        """
        This method parses the electrode parameters of the
        SEI model.

        Parameters
        ----------
        SEI: BaseComponentParser
            Object that parses the SEI parameters.
        """

    def parse_porousSEI_parameters(self, porous) -> None:
        """
        This method parses the electrode parameters of the
        SEI model.

        Parameters
        ----------
        SEI: BaseComponentParser
            Object that parses the SEI parameters.
        """
        porous.set_parameters(__cell_parameters__['porous'])

    def parse_compactSEI_parameters(self, compact) -> None:
        """
        This method parses the electrode parameters of the
        compact SEI model.

        Parameters
        ----------
        compactSEI: BaseComponentParser
            Object that parses the compact SEI parameters.
        """
        if not bool(compact):
            return
        compact.set_parameters(__cell_parameters__['compact'])

    # ******************************************************************************************* #
    # ***                            Preprocessing. BatteryCell                               *** #
    # ******************************************************************************************* #

    def set_SEI_parameters(self, SEI, problem) -> None:
        """
        This method preprocesses the electrode parameters of the
        electrochemical model.

        Parameters
        ----------
        SEI: BaseCellComponent
            Object where electrode parameters are preprocessed and
            stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """

    def set_porousSEI_parameters(self, porous, problem) -> None:
        """
        This method preprocesses the electrode parameters of the
        electrochemical model.

        Parameters
        ----------
        SEI: BaseCellComponent
            Object where electrode parameters are preprocessed and
            stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        porous.R = porous.parser.resistance.get_value(problem)
        porous.M = porous.parser.molecular_weight.get_value(problem)
        porous.rho = porous.parser.density.get_value(problem)
        porous.kappa = porous.parser.conductivity.get_value(problem)
        porous.delta0 = porous.parser.delta0.get_value(problem)

    def set_compactSEI_parameters(self, compact, problem) -> None:
        """
        This method preprocesses the electrode parameters of the
        compact SEI model.

        Parameters
        ----------
        compactSEI: BaseCellComponent
            Object where electrode parameters are preprocessed and
            stored.
        problem: Problem
            Object that handles the battery cell simulation.
        """
        compact.M = compact.parser.molecular_weight.get_value(problem)
        compact.rho = compact.parser.density.get_value(problem)
        compact.delta0 = compact.parser.delta0.get_value(problem)
        compact.f = compact.parser.growth_factor.get_value(problem)

    # ******************************************************************************************* #
    # ***                                     Equations                                       *** #
    # ******************************************************************************************* #

    def get_solvers_info(self, solvers_info, problem) -> None:
        """
        This method get the solvers information that concerns this
        specific model.

        Parameters
        ----------
        solvers_info: dict
            Dictionary containing solvers information.
        problem: Problem
            Object that handles the battery cell simulation.
        """

    def build_weak_formulation(self, equations: ProblemEquations, var: ProblemVariables,
                               cell: BatteryCell, mesher: BaseMesher, DT: TimeScheme,
                               W: BlockFunctionSpace, problem) -> None:
        """
        This method builds the weak formulation of this specific model.

        Parameters
        ----------
        equations: ProblemEquations
            Object that contains the system of equations of the problem.
        var: ProblemVariables
            Object that store the preprocessed problem variables.
        cell: BatteryCell
            Object where cell parameters are preprocessed and stored.
        mesher: BaseMesher
            Object that store the mesh information.
        DT: TimeScheme
            Object that provide the temporal derivatives with the
            specified scheme.
        W: BlockFunctionSpace
            Object that store the function space of each state variable.
        problem: Problem
            Object that handles the battery cell simulation.
        """


__cell_parameters__ = {
    'porous': {
        'resistance': {'default': 0, 'is_optional': True},
        'molecular_weight': {'aliases': ['molecularWeight']},
        'density': {},
        'conductivity': {},
        'delta0': {'default': 0, 'is_optional': True}
    },
    'compact': {
        'molecular_weight': {'aliases': ['molecularWeight']},
        'density': {},
        'delta0': {'default': 0, 'is_optional': True},
        'growth_factor': {'default': 0.5, 'is_optional': True}
    }
}
