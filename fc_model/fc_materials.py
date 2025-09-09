# Material property type codes per group
from typing import Dict, List, Literal, TypedDict, Union

from .fc_data import FCData


FCMaterialPropertiesTypeLiteral = Literal[
    "elasticity", # Упругость и вязкоупругость
    "common", # Общие свойства
    "thermal",  # Температурные свойства
    "geomechanic", # Геомеханика
    "plasticity", # Пластичность
    "hardening",  # Упрочнение
    "creep", # Позучесть
    "preload", # Преднагружение
    "strength", # Прочность
    "swelling" # Распухание
]


FC_MATERIAL_PROPERTY_NAMES_KEYS: Dict[FCMaterialPropertiesTypeLiteral, Dict[Union[str, int], Union[str, int]]] = {
    "elasticity": {
        0: "YOUNG_MODULE",         # HOOK
        1: "POISSON_RATIO",        # HOOK
        2: "SHEAR_MODULUS",        # HOOK, MURNAGHAN
        3: "BULK_MODULUS",         # HOOK, MURNAGHAN
        4: "MU",                   # BLATZ_KO
        5: "ALPHA",                # BLATZ_KO
        6: "BETA",                 # BLATZ_KO
        7: "LAME_MODULE",          # MURNAGHAN
        8: "C3",                   # MURNAGHAN
        9: "C4",                   # MURNAGHAN
        10: "C5",                  # MURNAGHAN
        16: "E_T",                 # HOOK_TRANSVERSAL_ISOTROPIC
        17: "E_L",                 # HOOK_TRANSVERSAL_ISOTROPIC
        18: "PR_T",                # HOOK_TRANSVERSAL_ISOTROPIC
        19: "PR_TL",               # HOOK_TRANSVERSAL_ISOTROPIC
        20: "G_TL",                # HOOK_TRANSVERSAL_ISOTROPIC
        21: "G12",                 # HOOK_ORTHOTROPIC
        22: "G23",                 # HOOK_ORTHOTROPIC
        23: "G13",                 # HOOK_ORTHOTROPIC
        24: "PRXY",                # HOOK_ORTHOTROPIC
        25: "PRYZ",                # HOOK_ORTHOTROPIC
        26: "PRXZ",                # HOOK_ORTHOTROPIC
        27: "C1",                  # COMPR_MOONEY
        28: "C2",                  # COMPR_MOONEY
        29: "D",                   # COMPR_MOONEY
        79: "VP",                  # HOOK
        80: "VS",                  # HOOK
        82: "C_1111",              # ANISOTROPIC
        83: "C_1112",              # ANISOTROPIC
        84: "C_1113",              # ANISOTROPIC
        85: "C_1122",              # ANISOTROPIC
        86: "C_1123",              # ANISOTROPIC
        87: "C_1133",              # ANISOTROPIC
        88: "C_1212",              # ANISOTROPIC
        89: "C_1213",              # ANISOTROPIC
        90: "C_1222",              # ANISOTROPIC
        91: "C_1223",              # ANISOTROPIC
        92: "C_1233",              # ANISOTROPIC
        93: "C_1313",              # ANISOTROPIC
        94: "C_1322",              # ANISOTROPIC
        95: "C_1323",              # ANISOTROPIC
        96: "C_1333",              # ANISOTROPIC
        97: "C_2222",              # ANISOTROPIC
        98: "C_2223",              # ANISOTROPIC
        99: "C_2233",              # ANISOTROPIC
        100: "C_2323",             # ANISOTROPIC
        101: "C_2333",             # ANISOTROPIC
        102: "C_3333",             # ANISOTROPIC
    },
    "common": {
        0: "DENSITY",                        # USUAL
        1: "STRUCTURAL_DAMPING_RATIO",       # USUAL
        2: "MASS_DAMPING_RATIO",             # USUAL
        3: "STIFFNESS_DAMPING_RATIO",        # USUAL
    },
    "thermal": {
        0: "COEF_LIN_EXPANSION",             # ISOTROPIC
        1: "COEF_THERMAL_CONDUCTIVITY",      # ISOTROPIC
        5: "COEF_THERMAL_CONDUCTIVITY_XX",   # ORTHOTROPIC
        9: "COEF_THERMAL_CONDUCTIVITY_YY",   # ORTHOTROPIC
        13: "COEF_THERMAL_CONDUCTIVITY_ZZ",  # ORTHOTROPIC
        14: "COEF_LIN_EXPANSION_X",          # ORTHOTROPIC
        15: "COEF_LIN_EXPANSION_Y",          # ORTHOTROPIC
        16: "COEF_LIN_EXPANSION_Z",          # ORTHOTROPIC
        17: "COEF_THERMAL_CONDUCTIVITY_T",   # TRANSVERSAL_ISOTROPIC
        18: "COEF_THERMAL_CONDUCTIVITY_L",   # TRANSVERSAL_ISOTROPIC
        19: "COEF_LIN_EXPANSION_T",          # TRANSVERSAL_ISOTROPIC
        20: "COEF_LIN_EXPANSION_L",          # TRANSVERSAL_ISOTROPIC
    },
    "geomechanic": {
        0: "PERMEABILITY",                   # BIOT_ISOTROPIC
        1: "FLUID_VISCOSITY",               # ISOTROPIC
        2: "POROSITY",                      # ISOTROPIC
        3: "FLUID_BULK_MODULUS",            # ISOTROPIC
        4: "SOLID_BULK_MODULUS",            # ISOTROPIC
        5: "BIOT_ALPHA",                    # BIOT_ISOTROPIC
        6: "PERMEABILITY_XX",               # BIOT_ORTHOTROPIC
        7: "PERMEABILITY_XY",               # BIOT_ORTHOTROPIC
        8: "PERMEABILITY_XZ",               # BIOT_ORTHOTROPIC
        9: "PERMEABILITY_YX",               # BIOT_ORTHOTROPIC
        10: "PERMEABILITY_YY",              # BIOT_ORTHOTROPIC
        11: "PERMEABILITY_YZ",              # BIOT_ORTHOTROPIC
        12: "PERMEABILITY_ZX",              # BIOT_ORTHOTROPIC
        13: "PERMEABILITY_ZY",              # BIOT_ORTHOTROPIC
        14: "PERMEABILITY_ZZ",              # BIOT_ORTHOTROPIC
        15: "PERMEABILITY_T",               # BIOT_TRANSVERSAL_ISOTROPIC
        16: "PERMEABILITY_TT",              # BIOT_TRANSVERSAL_ISOTROPIC
        17: "PERMEABILITY_TL",              # BIOT_TRANSVERSAL_ISOTROPIC
        18: "PERMEABILITY_L",               # BIOT_TRANSVERSAL_ISOTROPIC
        19: "FLUID_DENSITY",                # ISOTROPIC
        20: "BIOT_MODULUS",                 # ISOTROPIC
        21: "BIOT_ALPHA_X",                 # BIOT_ORTHOTROPIC
        22: "BIOT_ALPHA_Y",                 # BIOT_ORTHOTROPIC
        23: "BIOT_ALPHA_Z",                 # BIOT_ORTHOTROPIC
        24: "BIOT_ALPHA_T",                 # BIOT_TRANSVERSAL_ISOTROPIC
        25: "BIOT_ALPHA_L",                 # BIOT_TRANSVERSAL_ISOTROPIC
    },
    "plasticity": {
        0: "YIELD_STRENGTH",                 # MISES
        5: "YIELD_STRENGTH_COMPR",           # DRUCKER_PRAGER, MOHR_COULOMB
        7: "COHESION",                       # DRUCKER_PRAGER, MOHR_COULOMB
        8: "INTERNAL_FRICTION_ANGLE",        # DRUCKER_PRAGER, MOHR_COULOMB
        9: "DILATANCY_ANGLE",                # MOHR_COULOMB
        21: "DPC_A",                         # DRUCKER_PRAGER_CREEP
        22: "DPC_N",                         # DRUCKER_PRAGER_CREEP
        23: "DPC_M",                         # DRUCKER_PRAGER_CREEP
    },
    "hardening": {
        1: "TENSILE_STRAIN",                # LINEAR
        2: "E_TAN",                         # LINEAR
        3: "HARDENING",                     # MULTILINEAR
        6: "TENSILE_STRAIN_COMPR",          # LINEAR
        10: "E_TAN_COMPR",                  # LINEAR
        11: "HARDENING_COMPR",              # MULTILINEAR
        41: "HARDENING_COHES",              # MULTILINEAR
    },
    "creep": {
        38: "C1",                           # NORTON
        39: "C2",                           # NORTON
        40: "C3",                           # NORTON
    },
    "preload": {
        0: "STRESS_XX",                     # INITIAL
        1: "STRESS_YY",                     # INITIAL
        2: "STRESS_ZZ",                     # INITIAL
        3: "STRESS_XY",                     # INITIAL
        4: "STRESS_YZ",                     # INITIAL
        5: "STRESS_XZ",                     # INITIAL
        6: "STRAIN_XX",                     # INITIAL
        7: "STRAIN_YY",                     # INITIAL
        8: "STRAIN_ZZ",                     # INITIAL
        9: "STRAIN_XY",                     # INITIAL
        10: "STRAIN_YZ",                    # INITIAL
        11: "STRAIN_XZ",                    # INITIAL
        12: "PSI_XX",                       # INITIAL
        13: "PSI_YY",                       # INITIAL
        14: "PSI_ZZ",                       # INITIAL
        15: "PSI_XY",                       # INITIAL
        16: "PSI_YZ",                       # INITIAL
        17: "PSI_XZ",                       # INITIAL
        18: "PSI_YX",                       # INITIAL
        19: "PSI_ZY",                       # INITIAL
        20: "PSI_ZX",                       # INITIAL
        21: "GRADIENT_XX",                  # INITIAL
        22: "GRADIENT_YY",                  # INITIAL
        23: "GRADIENT_ZZ",                  # INITIAL
        24: "GRADIENT_XY",                  # INITIAL
        25: "GRADIENT_YZ",                  # INITIAL
        26: "GRADIENT_XZ",                  # INITIAL
        27: "GRADIENT_YX",                  # INITIAL
        28: "GRADIENT_ZY",                  # INITIAL
        29: "GRADIENT_ZX",                  # INITIAL
        30: "PLASTIC_STRAIN_XX",            # INITIAL
        31: "PLASTIC_STRAIN_YY",            # INITIAL
        32: "PLASTIC_STRAIN_ZZ",            # INITIAL
        33: "PLASTIC_STRAIN_XY",            # INITIAL
        34: "PLASTIC_STRAIN_YZ",            # INITIAL
        35: "PLASTIC_STRAIN_XZ",            # INITIAL
        36: "FINGER_STRAIN_XX",             # INITIAL
        37: "FINGER_STRAIN_YY",             # INITIAL
        38: "FINGER_STRAIN_ZZ",             # INITIAL
        39: "FINGER_STRAIN_XY",             # INITIAL
        40: "FINGER_STRAIN_YZ",             # INITIAL
        41: "FINGER_STRAIN_XZ",             # INITIAL
        42: "PLASTIC_STRAIN_MISES",         # INITIAL
        43: "THERMAL_STRESS_XX",            # INITIAL
        44: "THERMAL_STRESS_YY",            # INITIAL
        45: "THERMAL_STRESS_ZZ",            # INITIAL
        46: "THERMAL_STRESS_XY",            # INITIAL
        47: "THERMAL_STRESS_YZ",            # INITIAL
        48: "THERMAL_STRESS_XZ",            # INITIAL
    },
    "strength": {
        0: "TENSILE_STRENGTH",              # ISOTROPIC
        1: "TENSILE_STRENGTH_COMPR",        # ISOTROPIC
    },
    "swelling": {}
}

FC_MATERIAL_PROPERTY_NAMES_CODES: Dict[FCMaterialPropertiesTypeLiteral, Dict[Union[str, int], Union[str, int]]] = {
    group: {key: code for code, key in mapping.items()} for group, mapping in FC_MATERIAL_PROPERTY_NAMES_KEYS.items()
}

FC_MATERIAL_PROPERTY_TYPES_KEYS: Dict[FCMaterialPropertiesTypeLiteral, Dict[Union[str, int], Union[str, int]]] = {
    "elasticity": {
        0: "HOOK",
        1: "HOOK_ORTHOTROPIC",
        2: "HOOK_TRANSVERSAL_ISOTROPIC",
        3: "BLATZ_KO",
        4: "MURNAGHAN",
        11: "COMPR_MOONEY",
        20: "NEO_HOOK",
        21: "ANISOTROPIC",
    },
    "common": {0: "USUAL"},
    "thermal": {0: "ISOTROPIC", 1: "ORTHOTROPIC", 2: "TRANSVERSAL_ISOTROPIC"},
    "geomechanic": {
        0: "BIOT_ISOTROPIC",
        1: "BIOT_ORTHOTROPIC",
        2: "BIOT_TRANSVERSAL_ISOTROPIC",
    },
    "plasticity": {0: "MISES", 1: "DRUCKER_PRAGER", 4: "DRUCKER_PRAGER_CREEP", 9: "MOHR_COULOMB"},
    "hardening": {0: "LINEAR", 1: "MULTILINEAR"},
    "creep": {0: "NORTON"},
    "preload": {0: "INITIAL"},
    "strength": {0: "ISOTROPIC"},
    "swelling": {}
}

FC_MATERIAL_PROPERTY_TYPES_CODES: Dict[FCMaterialPropertiesTypeLiteral, Dict[Union[str, int], Union[str, int]]] = {
    group: {name: code for code, name in mapping.items()} for group, mapping in FC_MATERIAL_PROPERTY_TYPES_KEYS.items()
}


class FCSrcMaterialProperty(TypedDict):
    const_dep: List[Union[str, List[str]]]
    const_dep_size: List[int]
    const_names: List[Union[int, str]]
    const_types: List[Union[int, str, List[int]]]
    constants: List[str]
    type: Union[int, str]


class FCMaterialProperty:
    """
    Определяет одно физическое свойство материала (e.g., плотность, модуль Юнга).
    """
    type: Union[str, int]   # Берется из MATERIAL_PROPERTY_TYPES
    name: Union[str, int]  # Берется из CONST_NAME_MAP
    data: FCData  # Описание зависимости свойства

    def __init__(
        self,
        type: Union[str, int],
        name: Union[str, int],
        data: FCData
    ):
        """
        Инициализация свойства материала.
        :param type: Тип свойства (строка, например "HOOK")
        :param name: Имя свойства (строка, например "YOUNG_MODULE")
        :param data: Значение свойства (FCData)
        """
        self.type = type
        self.name = name
        self.data = data

    def __str__(self) -> str:
        return (
            f"FCMaterialProperty(type={self.type}, name={self.name}, data={self.data})"
        )

    def __repr__(self) -> str:
        return (
            f"<FCMaterialProperty type={self.type!r} name={self.name!r}>"
        )

class FCSrcMaterialBase(TypedDict):
    id: int
    name: str


class FCSrcMaterial(FCSrcMaterialBase, total=False):
    common: List[FCSrcMaterialProperty]
    elasticity: List[FCSrcMaterialProperty]
    geomechanic: List[FCSrcMaterialProperty]
    plasticity: List[FCSrcMaterialProperty]
    hardening: List[FCSrcMaterialProperty]
    creep: List[FCSrcMaterialProperty]
    preload: List[FCSrcMaterialProperty]
    strength: List[FCSrcMaterialProperty]
    thermal: List[FCSrcMaterialProperty]
    swelling: List[FCSrcMaterialProperty]


class FCMaterial:
    """
    Определяет материал и набор его физических свойств.
    """
    id: int  # Уникальный идентификатор материала
    name: str  # Имя материала
    properties: Dict[FCMaterialPropertiesTypeLiteral, List[List[FCMaterialProperty]]]  # Словарь, где свойства сгруппированы по типам

    def __init__(self, src_material:FCSrcMaterial):
        self.id = src_material['id']
        self.name = src_material['name']
        self.properties: Dict[FCMaterialPropertiesTypeLiteral, List[List[FCMaterialProperty]]] = {}

        # Источник: только группы верхнего уровня из FCSrcMaterial
        property_groups: Dict[FCMaterialPropertiesTypeLiteral, List[FCSrcMaterialProperty]] = {}

        for property_group_name in FC_MATERIAL_PROPERTY_NAMES_KEYS.keys():
            arr: List[FCSrcMaterialProperty] = src_material.get(property_group_name) # type: ignore
            if arr:
                property_groups[property_group_name] = arr

        for property_group_name, src_properties in property_groups.items():
            self.properties[property_group_name] = [] 
            for src_property in src_properties:
                type_code = src_property.get("type", 0)
                type_key = FC_MATERIAL_PROPERTY_TYPES_KEYS[property_group_name].get(type_code, type_code)  

                some_property_group: List[FCMaterialProperty] = []

                self.properties[property_group_name].append(some_property_group)

                for i, constants in enumerate(src_property.get("constants", [])):

                    name_code = src_property["const_names"][i]
                    name_key = FC_MATERIAL_PROPERTY_NAMES_KEYS[property_group_name].get(name_code, name_code)  

                    prop = FCMaterialProperty(
                        name=name_key,
                        type=type_key,
                        data=FCData(
                            constants,
                            src_property["const_types"][i],
                            src_property["const_dep"][i]
                        )
                    )
                    some_property_group.append(prop) 

    def dump(self) -> FCSrcMaterial:
        material_src: FCSrcMaterial = {"id": self.id, "name": self.name}

        for property_group_name, property_groups in self.properties.items():

            if property_group_name not in material_src:
                material_src[property_group_name] = []

            for some_property_group in property_groups:

                src_some_property_group: FCSrcMaterialProperty = {
                        "const_dep": [],
                        "const_dep_size": [],
                        "const_names": [],
                        "const_types": [],
                        "constants": [],
                        "type": 0,
                    }
                material_src[property_group_name].append(src_some_property_group) 

                for property in some_property_group:

                    constants, const_types, const_dep = property.data.dump()

                    type_code = FC_MATERIAL_PROPERTY_TYPES_CODES[property_group_name].get(property.type, property.type)
                    name_code = FC_MATERIAL_PROPERTY_NAMES_CODES[property_group_name].get(property.name, property.name)

                    src_some_property_group['type'] = int(type_code)
                    src_some_property_group['const_dep'].append(const_dep)
                    src_some_property_group['const_dep_size'].append(len(property.data))
                    src_some_property_group['const_names'].append(name_code)
                    src_some_property_group['const_types'].append(const_types)
                    src_some_property_group['constants'].append(constants)

        return material_src


    def __str__(self) -> str:
        return (
            f"FCMaterial(id={self.id}, name='{self.name}', properties={self.properties})"
        )

    def __repr__(self) -> str:
        return (
            f"<FCMaterial {self.id!r} {self.name!r}>"
        )