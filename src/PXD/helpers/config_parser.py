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
# along with this program. If not, see <http://www.gnu.org/licenses/>.#
import json
import os

from mpi4py import MPI


def _print(*args, **kwargs):
    if MPI.COMM_WORLD.rank == 0:
        print(*args,**kwargs, flush=True)

def _select_property(dictionary, prop, default = None):
    if not isinstance(dictionary, dict):
        return {'value': default}
    if prop in dictionary.keys():
        return dictionary[prop]
    else:
        # print("WARNING - '{}' not found".format(prop))
        return {'value': default}

def _get_value(dictionary):
    return dictionary['value']

def _get_arrhernius(dictionary):
    return dictionary['arrhenius']['activationEnergy'], dictionary['arrhenius']['referenceTemperature']

def _parse_file_source(property_dict, data_path):
    if isinstance(property_dict['value'], dict):
        assert all(key in property_dict['value'] for key in ['charge','discharge'])
        for k, val in property_dict['value'].items():
            path = os.path.join(data_path, val) 
            # assert os.path.exists(path)
            property_dict['value'][k]=path
    else:
        path = os.path.join(data_path, property_dict['value']) 
        # assert os.path.exists(path), f'{path} does not exists'
        property_dict['value'] = path
    return property_dict

class parser:
    def _check_arrhenius(self, prop):
        return 'arrhenius' in _select_property(self.dic, prop)
    
    def _parse_arrhenius(self, prop):
        return _get_arrhernius(_select_property(self.dic, prop))

    def _check_effective(self, prop):
        dic = _select_property(self.dic, prop)
        if not isinstance(dic, dict):
            return dic
        return 'effective' in dic.keys()

    def _parse_value(self, prop, default = None):
        if self._check_effective(prop):
            return _select_property(self.dic, prop)
        else:
            return _get_value(_select_property(self.dic, prop, default))

    def __bool__(self) -> bool:
        return bool(self.dic) 
            
class electrolyteInterface(parser):
    def __init__(self, dic):
        self.resitance = 0
        self.referenceVoltage = 0
        self.sideReactionExchangeCurrentDensity = 0
        self.molecularWeight = 0
        self.density = 1
        self.conductivity = 1
        self.delta0 = 0
        self.chargeTransferCoefficient = 0.5
        self.EC_diffusion = 0
        self.EC_eps = 0
        self.solventSurfConcentration = 0
        self.rateConstant = 0

        if "CEI" in dic.keys():
            self.dic = dic["CEI"]
            self.set_properties()
        elif "SEI" in dic.keys():
            self.dic = dic["SEI"]
            self.set_properties()
        else:
            self.dic = {}

    def set_properties(self):
        self.resitance = self._parse_value("resistance")
        self.referenceVoltage = self._parse_value("referenceVoltage")
        self.sideReactionExchangeCurrentDensity = self._parse_value("sideReactionExchangeCurrentDensity")
        self.molecularWeight = self._parse_value("molecularWeight")
        self.density = self._parse_value("density")
        self.conductivity = self._parse_value("conductivity")
        self.delta0 = self._parse_value("delta0", default=0)
        self.chargeTransferCoefficient = self._parse_value("chargeTransferCoefficient",default=0.5)
        self.EC_diffusion = self._parse_value("solventDiffusion")
        self.EC_eps = self._parse_value("solventPorosity")
        self.solventSurfConcentration = self._parse_value("solventSurfConcentration")
        self.rateConstant = self._parse_value("rateConstant")


class electrode(parser):
    def __init__(self, cell, dic):
        self.dic = dic
        self.active_materials = []
        self.build_electrode(cell)
        self.F = cell.F

    def set_properties(self):
        self.thickness = self._parse_value('thickness')
        self.area = self._parse_value('area')
        self.width = self._parse_value('width')
        self.height =  self._parse_value('height')
        if (self.area is None) and (self.width and self.height):
            self.area = self.width * self.height 
        self.porosity = self._parse_value('porosity')
        self.density = self._parse_value('density')
        self.bruggeman = self._parse_value('bruggeman')
        self.tortuosity = self._parse_value('tortuosity')
        self.tortuosity_s = self._parse_value('tortuosity_s')
        self.electronicConductivity = self._parse_value('electronicConductivity')
        self.thermalConductivity = self._parse_value('thermalConductivity')
        self.specificHeat = self._parse_value('specificHeat')
        self.doubleLayerCapacitance = self._parse_value('doubleLayerCapacitance')

    class active_material(parser):
        def __init__(self, dic:dict, index:int, data_path:str, electrode):
            self.dic = dic
            self.electrode = electrode
            self.index = index
            self.name = str(index)
            self.set_properties(data_path, electrode)
            self.inclusions = [self.__class__(inc, i, data_path, electrode) for i, inc in enumerate(self.dic.get("inclusions",[]))]
            self.porosity = 1-sum([inc.volumeFraction for inc in self.inclusions])
            
        def set_properties(self, data_path, electrode):
            if 'material' in self.dic.keys():
                self.name = self.dic['material']
            self.volumeFraction = self._parse_value('volumeFraction') or self._parse_value('volFrac_active')
            self.density = self._parse_value('density')
            self.massFraction = self._parse_value('massFraction')
            if self.volumeFraction is None:
                assert all([self.density, electrode.density, self.massFraction])
                self.volumeFraction = self.massFraction*electrode.density/self.density
            self.particleRadius = self._parse_value('particleRadius')
            self.maximumConcentration = self._parse_value('maximumConcentration')
            self.stoichiometry0 = self._parse_value('stoichiometry0')
            self.stoichiometry1 = self._parse_value('stoichiometry1')
            self.openCircuitPotential = _parse_file_source(self.dic['openCircuitPotential'], data_path)
            if 'entropyCoefficient' in self.dic:
                self.entropyCoefficient = _parse_file_source(self.dic['entropyCoefficient'], data_path)
            else:
                self.entropyCoefficient = None

            self.kineticConstant = self._parse_value('kineticConstant')
            if self._check_arrhenius('kineticConstant'):
                self.kineticConstant_Ea, self.kineticConstant_Tref = self._parse_arrhenius('kineticConstant')
            else:
                self.kineticConstant_Ea = 0.
                self.kineticConstant_Tref = 298.15    

            if self.dic['diffusionConstant'].get('source') == 'file':
                self.diffusionConstant = _parse_file_source(self.dic['diffusionConstant'], data_path)
            else:
                self.diffusionConstant = self._parse_value('diffusionConstant')

            if self._check_arrhenius('diffusionConstant'):
                self.diffusionConstant_Ea, self.diffusionConstant_Tref = self._parse_arrhenius('diffusionConstant')
            else:
                self.diffusionConstant_Ea = 0.
                self.diffusionConstant_Tref = 298.15   

            self.omega = self._parse_value('partial_molar_volume', default=0)
            self.young = self._parse_value('young_modulus')
            self.poisson = self._parse_value('poisson_ratio')

    def set_active_materials(self, materials_list, data_path, electrode):
        for i, active_material_dic in enumerate(materials_list):
            self.active_materials.append(self.active_material(active_material_dic, i, data_path, electrode))

    def parse_active_materials(self):
        val = _select_property(self.dic, 'active_materials')
        if isinstance(val, dict):
            return []
        return val

    def build_electrode(self, cell):
        self.set_properties()
        self.set_active_materials(self.parse_active_materials(), cell.data_path, self)
        self.SEI = electrolyteInterface(self.dic)

    def capacity(self):
        cap = 0
        for am in self.active_materials:
            if None not in [am.volumeFraction, am.maximumConcentration, am.stoichiometry1, am.stoichiometry0]:
                cap += am.volumeFraction* am.porosity * am.maximumConcentration * abs(am.stoichiometry1 - am.stoichiometry0) / 3600
            for inc in am.inclusions:
                if None not in [inc.volumeFraction, inc.maximumConcentration, inc.stoichiometry1, inc.stoichiometry0]:
                    cap += am.volumeFraction *inc.volumeFraction* inc.porosity * inc.maximumConcentration * abs(inc.stoichiometry1 - inc.stoichiometry0) / 3600
        if self.area and self.thickness:
            return cap * self.area * self.thickness * self.F


class curent_colector(parser):
    def __init__(self, cell, dic):
        self.dic = dic
        self.build_curent_colector(cell)

    def build_curent_colector(self, cell):

        self.thickness = self._parse_value('thickness')
        self.width = self._parse_value('width', default=cell.parse_value('separator', 'width'))
        self.height =  self._parse_value('height', default=cell.parse_value('separator', 'height'))
        self.area = self._parse_value('area')
        
        self.electronicConductivity = self._parse_value('electronicConductivity')
        
        self.thermalConductivity = self._parse_value('thermalConductivity')
        self.specificHeat = self._parse_value('specificHeat')
        self.density = self._parse_value('density')
        
        self.young = self._parse_value('young_modulus')
        self.poisson = self._parse_value('poisson_ratio')


class separator(parser):
    def __init__(self, cell, dic):
        self.dic = dic
        self.build_separator()

    def build_separator(self):
        self.thickness = self._parse_value('thickness')
        self.height = self._parse_value('height')
        self.width = self._parse_value('width')
        self.density = self._parse_value('density')

        self.porosity = self._parse_value('porosity')
        self.bruggeman = self._parse_value('bruggeman')
        self.tortuosity = self._parse_value('tortuosity')

        self.thermalConductivity = self._parse_value('thermalConductivity')
        self.specificHeat = self._parse_value('specificHeat')

class electrolyte(parser):
    def __init__(self, cell, dic):
        self.dic = dic
        self.type = self._parse_value('type', default='liquid')
        assert self.type in ('liquid', ) , "Solid electrolytes not supported in this version"
        self.intercalation_type = self._parse_value('intercalation_type', default='binary')
        assert self.intercalation_type in ('binary', ) , "Only binary electrolytes are supported in this version"
        self.build_electrolyte()

    def build_electrolyte(self):
        self.transferenceNumber = self._parse_value('transferenceNumber')
        self.activityDependence = self._parse_value('activityDependence', default=1)
        self.initialConcentration = self._parse_value('initialConcentration')

        self.diffusionConstant = self._parse_value('diffusionConstant')
        if self._check_arrhenius('diffusionConstant'):
            self.diffusionConstant_Ea, self.diffusionConstant_Tref = self._parse_arrhenius('diffusionConstant')
        else:
            self.diffusionConstant_Ea = 0.
            self.diffusionConstant_Tref = 298.15

        self.ionicConductivity = self._parse_value('ionicConductivity')
        if self._check_arrhenius('ionicConductivity'):
            self.ionicConductivity_Ea, self.ionicConductivity_Tref = self._parse_arrhenius('ionicConductivity')
        else:
            self.ionicConductivity_Ea = 0.
            self.ionicConductivity_Tref = 298.15

class CellParser:
    def __init__(self, params, data_path, log=True):
        self.log =log
        assert isinstance(params, (dict, str)), "params argument must be dict or str"
        self.data_path = data_path
        if isinstance(params, dict):
            self.dic = params
        else:
            path = os.path.join(data_path,params)
            assert os.path.exists(path), "Path {} does not exists".format(path)
            self.read_param_file(path)
        self.structure = self.dic['structure']
        self.check_dict_structure()

        # Constants
        self.R = self.parse_value('constants','R')
        self.F = self.parse_value('constants','F')
        self.alpha = self.parse_value('constants','alpha')

        # Cell parameters
        self.heatConvection = self.parse_value('properties','heatConvection')
        self.thermalExpansionRate = self.parse_value('properties','thermalExpansionRate')
        self.doubleLayerCapacitance_cc = self.parse_value('properties', 'doubleLayerCapacitance_cc')

        # Negative current collector
        self.negative_curent_colector = curent_colector(self, self._select_element("negativeCurrentCollector"))

        #Negative electrode
        self.negative_electrode = electrode(self, self._select_element("negativeElectrode"))

        # Separator
        self.separator = separator(self, self._select_element("separator"))

        #Positive electrode
        self.positive_electrode = electrode(self, self._select_element("positiveElectrode"))

        # Positive current collector
        self.positive_curent_colector = curent_colector(self, self._select_element("positiveCurrentCollector"))

        # Electrolyte
        self.electrolyte = electrolyte(self, self._select_element("electrolyte"))

        # Cell capacity
        self.compute_cell_properties()

    def _select_element(self, element):
        if self.type == 'li-ion':
            mandatory_entries = ['constants', 'separator', 'positiveElectrode', 'negativeElectrode', 'electrolyte', 'structure']
        if 'ncc' in self.structure:
            mandatory_entries.append('negativeCurrentCollector')
        if 'pcc' in self.structure:
            mandatory_entries.append('positiveCurrentCollector')
        if element not in mandatory_entries:
            return self.dic.get(element, {})
        else:
            return self.dic[element]

    def read_param_file(self, path):
        with open(path,'r') as fin:
            self.dic = json.load(fin)

    def write_param_file(self, path):
        with open(path,'w') as fout:
            json.dump(self.dic, fout)

    def check_arrhenius(self, element, prop):
        dic = _select_property(self._select_element(element), prop)
        return 'arrhenius' in dic.keys()
    
    def check_effective(self, element, prop):
        dic = _select_property(self._select_element(element), prop)
        if not isinstance(dic, dict):
            return dic
        return 'effective' in dic.keys()

    def parse_value(self, element, prop, default = None):
        if self.check_effective(element, prop):
            return _select_property(self._select_element(element), prop)
        else:
            val = _get_value(_select_property(self._select_element(element), prop, default))
            return val

    def parse_active_materials(self, element):
        val = _select_property(self._select_element(element), 'active_materials')
        if isinstance(val, dict):
            return []
        return val

    def parse_arrhenius(self, element, prop):
        return _get_arrhernius(_select_property(self._select_element(element), prop))

    def compute_cell_properties(self):

        self.capacity = min(self.negative_electrode.capacity() or 9e99, self.positive_electrode.capacity() or 9e99)
        if self.log:
            _print('Capacidad Anodo: {}'.format(self.negative_electrode.capacity()))
            _print('Capacidad Catodo: {}'.format(self.positive_electrode.capacity()))
            _print('Capacidad Celda: {}'.format(self.capacity))

        self.area = min(self.negative_electrode.area or 9e99, self.positive_electrode.area or 9e99)

        if any([element.height for element in [self.negative_electrode, self.negative_curent_colector, self.separator, self.positive_electrode, self.positive_curent_colector]]):
            self.height = max([element.height for element in [self.negative_electrode, self.negative_curent_colector, self.separator, self.positive_electrode, self.positive_curent_colector] if element.height])
        else:
            self.height = None

        if any([element.width for element in [self.negative_electrode, self.negative_curent_colector, self.separator, self.positive_electrode, self.positive_curent_colector]]):
            self.width = max([element.width for element in [self.negative_electrode, self.negative_curent_colector, self.separator, self.positive_electrode, self.positive_curent_colector] if element.height])
        else:
            self.width = None

    def check_dict_structure(self):
        if 'li' in self.structure:
            raise Exception('Lithium metal anode not supported in this version')
        elif 'a' not in self.structure or not 'c' in self.structure:
            raise Exception('Cell structure must have anode, separator and cathode')
        else:
            self.type = 'li-ion'
        if 'a' in self.structure:
            assert 'negativeElectrode' in self.dic
        assert 's' in self.structure
        if 's' in self.structure:
            assert 'separator' in self.dic
        if 'c' in self.structure:
            assert 'positiveElectrode' in self.dic
        if 'pcc' in self.structure:
            assert 'positiveCurrentCollector' in self.dic
        if 'ncc' in self.structure:
            assert 'negativeCurrentCollector' in self.dic
        assert 'electrolyte' in self.dic
        for i, el in enumerate(self.structure):
            if el == 'a':
                if i >0:
                    assert self.structure[i-1] in ['ncc','s']
                if i+1 < len(self.structure):
                    assert self.structure[i+1] in ['ncc','s']
            elif el == 'c':
                if i >0:
                    assert self.structure[i-1] in ['pcc','s']
                if i+1 < len(self.structure):
                    assert self.structure[i+1] in ['pcc','s']
            elif el == 's':
                if i >0:
                    assert self.structure[i-1] in ['a','c']
                if i+1 < len(self.structure):
                    assert self.structure[i+1] in ['a','c']
            elif el == 'ncc':
                if i >0:
                    assert self.structure[i-1] in ['a']
                if i+1 < len(self.structure):
                    assert self.structure[i+1] in ['a']
            elif el == 'pcc':
                if i >0:
                    assert self.structure[i-1] in ['c']
                if i+1 < len(self.structure):
                    assert self.structure[i+1] in ['c']
            elif el == 'li':
                if i >0:
                    assert self.structure[i-1] in ['ncc','s']
                if i+1 < len(self.structure):
                    assert self.structure[i+1] in ['ncc','s']
            else:
                raise Exception('Bad element in structure')
            
            
