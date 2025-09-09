from __future__ import annotations
import json

from typing import TypedDict, Optional, Dict, Any, List

from .fc_blocks import FCBlock
from .fc_conditions import FC_INITIAL_SET_TYPES_CODES, FC_INITIAL_SET_TYPES_KEYS, \
    FC_LOADS_TYPES_CODES, FC_LOADS_TYPES_KEYS, FC_RESTRAINT_FLAGS_CODES, FC_RESTRAINT_FLAGS_KEYS, FCInitialSet, FCRestraint, FCLoad
from .fc_constraint import FCConstraint
from .fc_coordinate_system import FCCoordinateSystem
from .fc_data import FC_DEPENDENCY_TYPES_CODES, FC_DEPENDENCY_TYPES_KEYS, FCData, FCDependencyColumn
from .fc_materials import FC_MATERIAL_PROPERTY_NAMES_CODES, FC_MATERIAL_PROPERTY_NAMES_KEYS, FC_MATERIAL_PROPERTY_TYPES_CODES, FC_MATERIAL_PROPERTY_TYPES_KEYS, FCMaterial, FCMaterialProperty
from .fc_mesh import FCMesh
from .fc_elements import FC_ELEMENT_TYPES_KEYID, FC_ELEMENT_TYPES_KEYNAME, FCElement, FCElementType
from .fc_property_tables import FCPropertyTable
from .fc_receivers import FCReceiver
from .fc_set import FCSet
from .fc_value import FCValue


class FCHeader(TypedDict):
    binary: bool
    description: str
    version: int
    types: dict[str, int]


class FCSettings(TypedDict, total=False):
    type: str  # Тип расчета ("static", "dynamic", "eigenfrequencies", "buckling", "spectrum", "harmonic", "effectiveprops")
    dimensions: str  # Размерность задачи ("2D", "3D")
    plane_state: str  # Вид расчетов для "2D" ("p-stress", "p-strain", "axisym_x", "axisym_y")
    permission_write: bool  # Разрешить запись на диск при нехватке ОЗУ
    periodic_bc: bool  # Периодические ГУ для расчета эффективных свойств
    finite_deformations: bool  # Включить конечные деформации
    elasticity: bool  # Включить упругость
    plasticity: bool  # Включить пластичность
    heat_transfer: bool  # Включить теплопроводность
    porefluid_transfer: bool  # Включить пьезопроводность
    slm: bool  # Включить аддитивное производство
    incompressibility: bool  # Включить несжимаемость
    preload: bool  # Преднагруженная модель
    lumpmass: bool  # Использовать лумпингованую диагональную матрицу масс
    radiation_among_surfaces: bool  # Включить лучистый теплообмен
    linear_solver: Dict[str, Any]  # Настройки линейного решателя
    nonlinear_solver: Dict[str, Any]  # Настройки нелинейного решателя
    damping: Dict[str, Any]  # Настройки демпфирования
    thermal_gap_settings: Dict[str, Any]  # Настройки решателя для газового зазора
    eigen_solver: Dict[str, Any]  # Настройки решателя собственных частот
    dynamics: Dict[str, Any]  # Настройки динамического расчета
    statics: Dict[str, Any]  # Настройки статического расчета
    harmonic: Dict[str, Any]  # Настройки гармонического расчета
    output: Dict[str, Any]  # Настройки вывода данных
    test_opts: Dict[str, Any]  # Настройки отладочной версии ядра



class FCModel:
    """
    Основной класс для представления, загрузки и сохранения модели в формате Fidesys Case (.fc).

    Представляет собой контейнер для всех сущностей модели: узлов, элементов,
    материалов, нагрузок, закреплений и т.д.

    Атрибуты:
        header (FCHeader): Заголовок файла.
        coordinate_systems (Dict[int, FCCoordinateSystem]): Коллекция систем координат.
        elems (FCMesh): Контейнер сетки и элементов.
        blocks (Dict[int, FCBlock]): Коллекция блоков, связывающих элементы с материалами.
        materials (Dict[int, FCMaterial]): Коллекция материалов и их физических свойств.
        loads (List[FCLoad]): Список нагрузок, приложенных к модели.
        restraints (List[FCRestraint]): Список закреплений (ограничений).
        ... и другие коллекции сущностей.

    Пример использования:
    
    # Создание или загрузка модели
    model = FCModel() # Создать пустую модель
    # model = FCModel(filepath="path/to/model.fc") # Загрузить из файла

    # ... (добавление узлов, элементов, материалов)

    # Сохранение модели
    model.save("new_model.fc")
    """

    header: FCHeader = {
        "binary": True,
        "description": "Fidesys Case Format",
        "types": {"char": 1, "short_int": 2, "int": 4, "double": 8, },
        "version": 3
    }

    coordinate_systems: Dict[int, FCCoordinateSystem]

    mesh: FCMesh

    blocks: Dict[int, FCBlock]
    property_tables: Dict[int, FCPropertyTable]
    materials: Dict[int, FCMaterial]

    loads: List[FCLoad]
    restraints: List[FCRestraint]
    initial_sets: List[FCInitialSet]

    contact_constraints: List[FCConstraint]
    coupling_constraints: List[FCConstraint]
    periodic_constraints: List[FCConstraint]

    receivers: List[FCReceiver]

    nodesets: Dict[int, FCSet]
    sidesets: Dict[int, FCSet]

    settings: Dict[str, Any]

    def __init__(self, filepath: Optional[str] = None) -> None:
        """
        Инициализирует объект FCModel.

        Если указан `filepath`, модель будет загружена из этого файла.
        В противном случае будет создана пустая модель с инициализированными коллекциями.

        Args:
            filepath (str, optional): Путь к файлу .fc для загрузки. Defaults to None.
        """
        
        # Инициализация всех коллекций как пустых
        self.coordinate_systems = {}

        self.mesh = FCMesh()

        self.blocks = {}
        self.property_tables = {}
        self.materials = {}
        
        self.loads: List[FCLoad] = []
        self.restraints: List[FCRestraint] = []
        self.initial_sets: List[FCInitialSet] = []

        self.contact_constraints: List[FCConstraint] = []
        self.coupling_constraints: List[FCConstraint] = []
        self.periodic_constraints: List[FCConstraint] = []
        self.receivers: List[FCReceiver] = []

        self.nodesets = {}
        self.sidesets = {}

        self.settings = {}

        if filepath:
            with open(filepath, "r") as f:
                src_data = json.load(f)

            self.src_data = src_data
            self._decode_header(src_data)
            self._decode_blocks(src_data)
            self._decode_coordinate_systems(src_data)
            self._decode_contact_constraints(src_data)
            self._decode_coupling_constraints(src_data)
            self._decode_periodic_constraints(src_data)
            self._decode_mesh(src_data)
            self._decode_settings(src_data)
            self._decode_materials(src_data)
            self._decode_restraints(src_data)
            self._decode_initial_sets(src_data)
            self._decode_loads(src_data)
            self._decode_receivers(src_data)
            self._decode_property_tables(src_data)
            self._decode_sets(src_data)


    def save(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(self.dump(), f, indent=4)


    def dump(self) -> Dict[str, Any]:
        """
        Сохраняет текущее состояние модели в файл формата .fc.

        Собирает данные из всех коллекций (узлы, элементы, материалы и т.д.),
        кодирует их в нужный формат (JSON с base64 для бинарных данных)
        и записывает в указанный файл.

        Args:
            filepath (str): Путь к файлу, в который будет сохранена модель.
        """

        output_data: Dict[str, Any] = {}

        self._encode_blocks(output_data)
        self._encode_contact_constraints(output_data)
        self._encode_coordinate_systems(output_data)
        self._encode_coupling_constraints(output_data)
        self._encode_periodic_constraints(output_data)
        self._encode_header(output_data)
        self._encode_loads(output_data)
        self._encode_materials(output_data)
        self._encode_mesh(output_data)
        self._encode_receivers(output_data)
        self._encode_restraints(output_data)
        self._encode_initial_sets(output_data)
        self._encode_settings(output_data)
        self._encode_property_tables(output_data)
        self._encode_sets(output_data)

        return output_data

    def _decode_header(self, input_data: Dict[str, Any]) -> None:
        header_data = input_data.get('header')
        if header_data is not None:
            self.header = header_data

    def _encode_header(self, output_data: Dict[str, Any]) -> None:
        output_data['header'] = self.header

    def _decode_blocks(self, input_data: Dict[str, Any]) -> None:
        self.blocks = {}
        for src in input_data.get('blocks', []):
            blk = FCBlock(src)
            self.blocks[blk.id] = blk

    def _encode_blocks(self, output_data: Dict[str, Any]) -> None:
        if self.blocks:
            output_data['blocks'] = [blk.dump() for blk in self.blocks.values()]

    def _decode_coordinate_systems(self, input_data: Dict[str, Any]) -> None:
        self.coordinate_systems = {}
        for src in input_data.get('coordinate_systems', []):
            cs = FCCoordinateSystem(src)
            self.coordinate_systems[cs.id] = cs

    def _encode_coordinate_systems(self, output_data: Dict[str, Any]) -> None:
        if self.coordinate_systems:
            output_data['coordinate_systems'] = [cs.dump() for cs in self.coordinate_systems.values()]


    def _decode_contact_constraints(self, input_data: Dict[str, Any]) -> None:
        for cc_src in input_data.get('contact_constraints', []):
            self.contact_constraints.append(FCConstraint(cc_src))


    def _encode_contact_constraints(self, output_data: Dict[str, Any]) -> None:
        if self.contact_constraints:
            output_data['contact_constraints'] = []
            for cc in self.contact_constraints:
                output_data['contact_constraints'].append(cc.dump())

    def _decode_coupling_constraints(self, input_data: Dict[str, Any]) -> None:
        for cc_src in input_data.get('coupling_constraints', []):
            self.coupling_constraints.append(FCConstraint(cc_src))


    def _encode_coupling_constraints(self, output_data: Dict[str, Any]) -> None:
        if self.coupling_constraints:
            output_data['coupling_constraints'] = []
            for cc in self.coupling_constraints:
                output_data['coupling_constraints'].append(cc.dump())


    def _decode_periodic_constraints(self, input_data: Dict[str, Any]) -> None:

        for cc_src in input_data.get('periodic_constraints', []):
            self.periodic_constraints.append(FCConstraint(cc_src))


    def _encode_periodic_constraints(self, output_data: Dict[str, Any]) -> None:
        if self.periodic_constraints:
            output_data['periodic_constraints'] = []

            for cc in self.periodic_constraints:
                output_data['periodic_constraints'].append(cc.dump())


    def _decode_sets(self, src_data: Dict[str, Any]) -> None:
        if 'sets' in src_data:
            self.nodesets = {}
            for ns_src in src_data['sets'].get('nodesets', []):
                ns = FCSet(ns_src)
                self.nodesets[ns.id] = ns
            self.sidesets = {}
            for ss_src in src_data['sets'].get('sidesets', []):
                ss = FCSet(ss_src)
                self.sidesets[ss.id] = ss


    def _encode_sets(self, src_data: Dict[str, Any]) -> None:
        if not (self.nodesets or self.sidesets):
            return
        src_data['sets'] = {}
        if self.nodesets:
            src_data['sets']['nodesets'] = [ns.dump() for ns in self.nodesets.values()]
        if self.sidesets:
            src_data['sets']['sidesets'] = [ss.dump() for ss in self.sidesets.values()]


    def _decode_mesh(self, src_data: Dict[str, Any]) -> None:
        self.mesh.decode(src_data['mesh'])

    def _encode_mesh(self, src_data: Dict[str, Any]) -> None:
        src_data['mesh'] = self.mesh.encode()


    def _decode_settings(self, src_data: Dict[str, Any]) -> None:
        self.settings = src_data.get('settings', {})

    def _encode_settings(self, src_data: Dict[str, Any]) -> None:
        src_data['settings'] = self.settings

    def _decode_property_tables(self, src_data: Dict[str, Any]) -> None:
        self.property_tables = {}
        for pt_src in src_data.get('property_tables', []):
            pt = FCPropertyTable(pt_src)
            self.property_tables[pt.id] = pt

    def _encode_property_tables(self, src_data: Dict[str, Any]) -> None:
        if self.property_tables:
            src_data['property_tables'] = [pt.dump() for pt in self.property_tables.values()]


    def _decode_materials(self, src_data: Dict[str, Any]) -> None:
        self.materials = {}
        for src_material in src_data.get('materials', []):
            material = FCMaterial(src_material)
            self.materials[material.id] = material


    def _encode_materials(self, src_data: Dict[str, Any]) -> None:
        if self.materials:
            src_data['materials'] = [mat.dump() for mat in self.materials.values()]


    def _decode_loads(self, input_data: Dict[str, Any]) -> None:
        for src_load in input_data.get('loads', []):
            self.loads.append(FCLoad(src_load))


    def _encode_loads(self, output_data: Dict[str, Any]) -> None:
        if self.loads:
            output_data['loads'] = []
            for load in self.loads:
                output_data['loads'].append(load.dump())

    def _decode_restraints(self, input_data: Dict[str, Any]) -> None:
        for src_restraint in input_data.get('restraints', []):
            self.restraints.append(FCRestraint(src_restraint))

    def _encode_restraints(self, output_data: Dict[str, Any]) -> None:
        if self.restraints:
            output_data['restraints'] = []
            for restraint in self.restraints:
                output_data['restraints'].append(restraint.dump())

    def _decode_initial_sets(self, input_data: Dict[str, Any]) -> None:
        for src_initial_set in input_data.get('initial_sets', []):
            self.initial_sets.append(FCInitialSet(src_initial_set))

    def _encode_initial_sets(self, output_data: Dict[str, Any]) -> None:
        if self.initial_sets:
            output_data['initial_sets'] = []
            for initial_set in self.initial_sets:
                output_data['initial_sets'].append(initial_set.dump())

    def _decode_receivers(self, input_data: Dict[str, Any]) -> None:
        for src_receiver in input_data.get('receivers', []):
            self.receivers.append(FCReceiver(src_receiver))

    def _encode_receivers(self, output_data: Dict[str, Any]) -> None:
        if self.receivers:
            output_data['receivers'] = []
            for receiver in self.receivers:
                output_data['receivers'].append(receiver.dump())



__all__ = [
    'FCModel',
    'FCMesh', 'FCBlock', 'FCPropertyTable', 'FCCoordinateSystem', 'FCConstraint',
    'FCElement', 'FCElementType', 'FCMaterial', 'FCLoad', 'FCRestraint', 'FCInitialSet',
    'FCReceiver', 'FCSet', 'FCDependencyColumn', 'FCValue', 'FCData', 'FCHeader',
    'FCMaterialProperty',
    'FC_DEPENDENCY_TYPES_KEYS', 'FC_DEPENDENCY_TYPES_CODES',
    'FC_INITIAL_SET_TYPES_CODES', 'FC_INITIAL_SET_TYPES_KEYS',
    'FC_LOADS_TYPES_CODES', 'FC_LOADS_TYPES_KEYS',
    'FC_MATERIAL_PROPERTY_NAMES_CODES', 'FC_MATERIAL_PROPERTY_NAMES_KEYS',
    'FC_MATERIAL_PROPERTY_TYPES_CODES', 'FC_MATERIAL_PROPERTY_TYPES_KEYS',
    'FC_RESTRAINT_FLAGS_CODES', 'FC_RESTRAINT_FLAGS_KEYS',
    'FC_ELEMENT_TYPES_KEYID', 'FC_ELEMENT_TYPES_KEYNAME',
]
