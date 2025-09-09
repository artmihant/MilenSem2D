from typing import Dict, Union, cast
from typing import List, Optional, TypedDict
from numpy import dtype

from .fc_data import FCData
from .fc_value import FCValue


FC_LOADS_TYPES_KEYS = {
    # Нагрузки на грань
    1: 'FaceDeadStress',                    # Давление на грань
    3: 'FaceTrackingStress',                # Следящее давление на грань
    11: 'FaceHeatFlux',                     # Тепловой поток на грани
    13: 'FaceConvection',                   # Конвекция на грани
    15: 'FaceRadiation',                    # Излучение на грани
    19: 'FaceAbsorbingBC',                  # Поглощающее ГУ на грани
    21: 'ShellHeatfluxTopBottom',           # Тепловой поток верх и низ
    22: 'ShellHeatfluxTop',                 # Тепловой поток верх
    23: 'ShellHeatfluxBottom',              # Тепловой поток низ
    24: 'ShellConvectionTopBottom',         # Конвекция верх и низ
    25: 'ShellConvectionTop',               # Конвекция верх
    26: 'ShellConvectionBottom',            # Конвекция низ
    35: 'FaceDistributedForce',             # Распределенная сила на грань
    36: 'FaceEquivalentForce',              # Равнодействующая сила на грань
    37: 'FaceTrackingDistributedForce',     # Следящая распределенная сила на грань
    38: 'FaceTrackingEquivalentForce',      # Следящая равнодействующая сила на грань
    39: 'FaceFluidFlux',                    # Поток жидкости через грань

    # Нагрузки на ребро
    2: 'SegmentDeadStress',                 # Давление на ребро
    4: 'SegmentTrackingStress',             # Следящее давление на ребро
    12: 'SegmentHeatFlux',                  # Тепловой поток на ребре
    14: 'SegmentConvection',                # Конвективный теплообмен на ребре
    16: 'SegmentRadiation',                 # Излучение на ребре
    20: 'SegmentAbsorbingBC',               # Поглощающее ГУ на ребре
    31: 'SegmentDistributedForce',          # Распределенная сила на ребро
    32: 'SegmentEquivalentForce',           # Равнодействующая сила на ребро
    33: 'SegmentTrackingDistributedForce',  # Следящая распределенная сила на ребро
    34: 'SegmentTrackingEquivalentForce',   # Следящая равнодействующая сила на ребро
    40: 'SegmentFluidFlux',                 # Поток жидкости через ребро

    # Нагрузки на узел
    5: 'NodeForce',                         # Узловая сила
    18: 'HeatSource',                       # Узловой источник тепла
    28: 'NodeHeatFlux',                     # Узловой тепловой поток
    29: 'NodeConvection',                   # Узловая конвекция
    30: 'NodeRadiation',                    # Узловое излучение
    41: 'NodeFluidFlux',                    # Поток жидкости через узел
    43: 'FluidSource',                      # Узловой источник жидкости

    # Нагрузки на элемент
    17: 'VolumeHeatSource',                 # Объемный источник тепла
    42: 'VolumeFluidSource',                # Объемный источник жидкости
    44: 'VolumeGravityMassForce',           # Гравитация
}

FC_LOADS_TYPES_CODES: Dict[str, int] = {code: key for key, code in FC_LOADS_TYPES_KEYS.items()}


FC_RESTRAINT_FLAGS_KEYS = {
    0: 'EmptyRestraint',            # Отсутствует закрепление. Применяется в массивах вместе с остальными вариантами.
    1: 'Displacement',              # ГУ по перемещениям и поворотам для узлов. Длина массива равна 6.
    2: 'Velocity',                  # ГУ по скоростям и скоростям поворотов для узлов. Длина массива равна 6
    3: 'Temperature',               # ГУ по температуре для узлов. Длина массива равна 1.
    4: 'TemperatureTop',            # Температура на верхней поверхности. Для узлов оболочечных элементов.
    5: 'TemperatureBottom',         # Температура на нижней поверхности. Для узлов оболочечных элементов.
    6: 'TemperatureMiddle',         # Температура для узлов на срединной поверхности оболочечного элемента.
    7: 'TemperatureGradient',       # Градиент температуры для узлов оболочечного элемента.
    9: 'Acceleration',              # ГУ по ускорениям и угловым ускорениям для узлов. Длина массива равна 6.
    10: 'PorePressure',             # ГУ по поровому давлению для узлов. Длина массива равна 1.
    12: 'DirectionDisplacement',    # ГУ по направлению по перемещениям. Применяется к граням элементов. Длина массива равна 1
    13: 'DirectionVelocity',        # ГУ по направлению по скоростям. Применяется к граням элементов. Длина массива равна 1
    14: 'DirectionAcceleration',    # ГУ по направлению по ускорениям. Применяется к граням элементов. Длина массива равна 1
    15: 'VolumeAngularVelocity',    # ГУ по угловым скоростям. Применяется к элементам. Длина массива равна 3.
    16: 'Fluence',                  # ГУ по флюенсу (дозе). Длина массива равна 1.
}

FC_RESTRAINT_FLAGS_CODES: Dict[str, int] = {code: key for key, code in FC_RESTRAINT_FLAGS_KEYS.items()}


FC_INITIAL_SET_TYPES_KEYS = {
    0: "Displacement",
    1: "Velocity",
    2: "AngularVelocity",
    3: "Temperature",
    4: "PorePressure"
}

FC_INITIAL_SET_TYPES_CODES: Dict[str, int] = {code: key for key, code in FC_INITIAL_SET_TYPES_KEYS.items()}


class FCSrcLoadStrict(TypedDict):
    apply_to: str
    apply_to_size: int
    name: str
    id: int
    type: int
    data: List[str]
    dep_var_num: List[Union[str, List[str]]]
    dep_var_size: List[int]
    dependency_type: List[Union[int, List[int]]]

class FCSrcLoad(FCSrcLoadStrict, total=False):
    cs: int


class FCSrcRestraintStrict(TypedDict):
    apply_to: str
    apply_to_size: int
    name: str
    id: int
    data: List[str]
    dep_var_num: List[Union[str, List[str]]]
    dep_var_size: List[int]
    dependency_type: List[Union[int, str, List[int]]]
    flag: List[int]

class FCSrcRestraint(FCSrcRestraintStrict, total=False):
    cs: int


class FCSrcInitialSetStrict(TypedDict):
    apply_to: str
    apply_to_size: int
    id: int
    data: List[str]
    dep_var_num: List[Union[str, List[str]]]
    dep_var_size: List[int]
    dependency_type: List[Union[int, str, List[int]]]
    flag: List[int]
    type: int


class FCSrcInitialSet(FCSrcInitialSetStrict, total=False):
    cs: int


class FCLoad:
    id: int
    name: str
    type: str

    apply: FCValue
    cs_id: int
    data: List[FCData]

    def __init__(self, src_load: FCSrcLoad):

        self.id = src_load['id']
        self.name = src_load['name']

        self.cs_id = src_load.get('cs', 0)

        self.apply = FCValue(src_load['apply_to'], dtype('int32'))
        self.apply.resize(src_load.get('apply_to_size', 0))
        if len(self.apply) != src_load.get('apply_to_size', 0):
            raise ValueError(
                f"Load(id={self.id}) apply_to_size mismatch: {len(self.apply)} != {src_load.get('apply_to_size', 0)}"
            )
        self.type = FC_LOADS_TYPES_KEYS[src_load['type']]
        self.data: List[FCData] = []

        if 'data' in src_load:
            dep_types_all = src_load.get("dependency_type", [])
            dep_vars_all = src_load.get('dep_var_num', [])
            for i, data in enumerate(src_load["data"]):
                dep_type_i = dep_types_all[i] if i < len(dep_types_all) else 0
                dep_var_i = dep_vars_all[i] if i < len(dep_vars_all) else ""
                self.data.append(FCData(
                    data,
                    dep_type_i,
                    dep_var_i
                ))
            # consistency for dependency arrays lengths
            if len(dep_types_all) and len(dep_types_all) < len(self.data):
                raise ValueError("dependency_type shorter than data components")
            if len(dep_vars_all) and len(dep_vars_all) < len(self.data):
                raise ValueError("dep_var_num shorter than data components")

    def dump(self) -> FCSrcLoad:

        load_src: FCSrcLoad = {
            'id': self.id,
            'name': self.name,
            'type': FC_LOADS_TYPES_CODES[self.type],
            'apply_to': self.apply.dump(),
            'apply_to_size': len(self.apply),
            'data': [],
            'dependency_type': [],
            'dep_var_num': [],
            'dep_var_size': [],
        }

        if self.cs_id:
            load_src['cs'] = self.cs_id

        for data in self.data:
            src_data, src_types, src_deps = data.dump()
            load_src['data'].append(src_data)
            dep_types: Union[int, str, List[int]] = src_types 
            if isinstance(dep_types, str):
                raise TypeError("dep_types должен быть int или List[int], а не str")
            dep_vars: Union[str, List[str]] = src_deps 
            load_src['dependency_type'].append(dep_types)
            load_src['dep_var_num'].append(dep_vars)
            load_src['dep_var_size'].append(len(data))

        return load_src

    def __str__(self) -> str:
        return (
            f"FCLoad(id={self.id}, name='{self.name}', type={self.type}, "
            f"cs_id={self.cs_id}, apply_to_size={len(self.apply)}, data_count={len(self.data)})"
        )

    def __repr__(self) -> str:
        return f"<FCLoad {self.id!r} {self.name} {self.type}>"



class FCRestraint:
    id: int
    name: str

    apply: FCValue
    cs_id: int
    data: List[FCData]
    flags: List[str]

    def __init__(self, src_restraint:FCSrcRestraint):

        self.id = src_restraint['id']
        self.name = src_restraint['name']
        self.cs_id = src_restraint.get('cs', 0)

        self.apply = FCValue(src_restraint['apply_to'], dtype('int32'))
        self.apply.resize(src_restraint.get('apply_to_size', 0))
        if len(self.apply) != src_restraint.get('apply_to_size', 0):
            raise ValueError(
                f"Restraint(id={self.id}) apply_to_size mismatch: {len(self.apply)} != {src_restraint.get('apply_to_size', 0)}"
            )

        self.data: List[FCData] = []

        if 'data' in src_restraint:
            dep_types_all = src_restraint.get("dependency_type", [])
            dep_vars_all = src_restraint.get('dep_var_num', [])
            for i, data in enumerate(src_restraint["data"]):
                dep_type_i = dep_types_all[i] if i < len(dep_types_all) else 0
                dep_var_i = dep_vars_all[i] if i < len(dep_vars_all) else ""
                self.data.append(FCData(
                    data,
                    dep_type_i,
                    dep_var_i
                ))
            if len(dep_types_all) and len(dep_types_all) < len(self.data):
                raise ValueError("dependency_type shorter than data components")
            if len(dep_vars_all) and len(dep_vars_all) < len(self.data):
                raise ValueError("dep_var_num shorter than data components")
        self.flags = [FC_RESTRAINT_FLAGS_KEYS[code] for code in src_restraint['flag']]


    def dump(self) -> FCSrcRestraint:

        src_restraint: FCSrcRestraint = {
            'id': self.id,
            'name': self.name,
            'apply_to': self.apply.dump(),
            'apply_to_size': len(self.apply),
            'data': [],
            'dependency_type': [],
            'dep_var_num': [],
            'dep_var_size': [],
            'flag': []
        }

        if self.cs_id:
            src_restraint['cs'] = self.cs_id


        for data in self.data:
            src_data, src_types, src_deps = data.dump()
            src_restraint['data'].append(src_data)
            dep_types: Union[int, str, List[int]] = src_types  
            dep_vars: Union[str, List[str]] = src_deps 
            src_restraint['dependency_type'].append(dep_types)
            src_restraint['dep_var_num'].append(dep_vars)
            src_restraint['dep_var_size'].append(len(data))

        src_restraint['flag'] = [FC_RESTRAINT_FLAGS_CODES[key] for key in self.flags] 

        return src_restraint

    def __str__(self) -> str:
        return (
            f"FCRestraint(id={self.id}, name='{self.name}', apply={self.apply}, cs_id={self.cs_id}, "
            f"data={self.data}, flags={self.flags})"
        )

    def __repr__(self) -> str:
        return (
            f"<FCRestraint {self.id!r} {self.name} {self.flags}>"
        )



class FCInitialSet:
    id: int
    apply: FCValue
    cs_id: int
    data: List[FCData]
    flags: List[str]
    type: str

    def __init__(self, src_initial_set:FCSrcInitialSet):

        self.id = src_initial_set['id']
        self.cs_id = src_initial_set.get('cs', 0)

        self.apply = FCValue(src_initial_set['apply_to'], dtype('int32'))
        self.apply.resize(src_initial_set.get('apply_to_size', 0))
        if len(self.apply) != src_initial_set.get('apply_to_size', 0):
            raise ValueError(
                f"InitialSet(id={self.id}) apply_to_size mismatch: {len(self.apply)} != {src_initial_set.get('apply_to_size', 0)}"
            )
        
        self.data: List[FCData] = []

        if 'data' in src_initial_set:
            dep_types_all = src_initial_set.get("dependency_type", [])
            dep_vars_all = src_initial_set.get('dep_var_num', [])
            for i, data in enumerate(src_initial_set["data"]):
                dep_type_i = dep_types_all[i] if i < len(dep_types_all) else 0
                dep_var_i = dep_vars_all[i] if i < len(dep_vars_all) else ""
                self.data.append(FCData(
                    data,
                    dep_type_i,
                    dep_var_i
                ))
            if len(dep_types_all) and len(dep_types_all) < len(self.data):
                raise ValueError("dependency_type shorter than data components")
            if len(dep_vars_all) and len(dep_vars_all) < len(self.data):
                raise ValueError("dep_var_num shorter than data components")

        self.flags = [FC_RESTRAINT_FLAGS_KEYS[code] for code in src_initial_set['flag']]

        self.type = FC_INITIAL_SET_TYPES_KEYS[src_initial_set['type']]

    def dump(self) -> FCSrcInitialSet:

        src_initial_set: FCSrcInitialSet = {
            'id': self.id,
            'apply_to': self.apply.dump(),
            'apply_to_size': len(self.apply),
            'data': [],
            'dependency_type': [],
            'dep_var_num': [],
            'dep_var_size': [],
            'flag': [FC_RESTRAINT_FLAGS_CODES[key] for key in self.flags],
            'type': FC_INITIAL_SET_TYPES_CODES[self.type]
        }

        if self.cs_id:
            src_initial_set['cs'] = self.cs_id

        for data in self.data:
            src_data, src_types, src_deps = data.dump()
            src_initial_set['data'].append(src_data)
            dep_types: Union[int, str, List[int]] = src_types
            dep_vars: Union[str, List[str]] = src_deps
            src_initial_set['dependency_type'].append(dep_types)
            src_initial_set['dep_var_num'].append(dep_vars)
            src_initial_set['dep_var_size'].append(len(data))

        return src_initial_set


    def __str__(self) -> str:
        return (
            f"FCInitialSet(id={self.id}, apply={self.apply}, data={self.data}, "
            f"flags={self.flags}, type={self.type}, cs_id={self.cs_id})"
        )

    def __repr__(self) -> str:
        return (
            f"<FCInitialSet {self.id!r} {self.type} {self.flags}>"
        )
