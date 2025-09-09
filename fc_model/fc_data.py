# Dependency (const_types) enumeration mapping
from typing import Dict, List, Tuple, Union

from numpy import dtype, float64, generic
from numpy.typing import NDArray

from .fc_value import FCValue

# Dependency types (FCData)
FC_DEPENDENCY_TYPES_KEYS: Dict[int, str] = {
    0: "CONSTANT",
    1: "TABULAR_X",
    2: "TABULAR_Y",
    3: "TABULAR_Z",
    4: "TABULAR_TIME",
    5: "TABULAR_TEMPERATURE",
    6: "FORMULA",
    7: "TABULAR_FREQUENCY",
    8: "TABULAR_STRAIN",
    10: "TABULAR_ELEMENT_ID",
    11: "TABULAR_NODE_ID",
    12: "TABULAR_MODE_ID",
}

FC_DEPENDENCY_TYPES_CODES: Dict[str, int] = {v: k for k, v in FC_DEPENDENCY_TYPES_KEYS.items()}


class FCDependencyColumn:
    type: str  # Форма задания зависимости - значение из DEPENDENCY_TYPES
    value: FCValue

    def __init__(self, type: str, value: FCValue):
        self.type = type
        self.value = value

    def __str__(self) -> str:
        return f"FCDependencyColumn(type={self.type}, value={self.value})"

    def __repr__(self) -> str:
        return f"<FCDependencyColumn type={self.type!r}>"


class FCData:
    """
    Определяет зависимость свойства от внешних факторов (температура, координаты, etc.).
    """
    type: Union[int, str] # -1 - таблица, иное число - константа

    value: FCValue  # Данные для зависимости (e.g., массив ID узлов)
    table: List[FCDependencyColumn]

    def __init__(self, data: Union[NDArray[generic], str], dep_type: Union[List[int], int, str], dep_data: Union[List[NDArray[generic]], List[str], str, None]):

        if isinstance(dep_type, list) and isinstance(dep_data, list):
            self.value = FCValue(data, dtype(float64))
            self.type = -1
            if len(dep_type) != len(dep_data):
                raise ValueError("FCData: dep_type and dep_data lists must have equal lengths")
            self.table = [FCDependencyColumn(
                type = FC_DEPENDENCY_TYPES_KEYS[deps_type],
                value = FCValue(dep_data[j], dtype(float64))
            ) for j, deps_type in enumerate(dep_type)]

        elif (isinstance(dep_type, int) or isinstance(dep_type, str)) and isinstance(dep_data, str):
            self.value = FCValue(data, dtype(float64), 'formula' if dep_type == 6 else 'array')
            self.type = dep_type
            self.table = []
        else:
            raise ValueError("Invalid dependency data")

    def dump(self) -> Union[Tuple[str, List[int], List[str]], Tuple[str, Union[int, str], str]]:
        if self.type == -1:
            return self.value.dump(), [FC_DEPENDENCY_TYPES_CODES[deps.type] for deps in self.table], [deps.value.dump() for deps in self.table]
        else:
            return self.value.dump(), self.type, ""

    def __len__(self) -> int:
        if not len(self.table):
            return 0
        return len(self.table[0].value)

    def __str__(self) -> str:
        return (
            f"FCData(type={self.type}, value={self.value}, table={self.table})"
        )

    def __repr__(self) -> str:
        return (
            f"<FCData {self.type!r}({len(self)}))>"
        )
