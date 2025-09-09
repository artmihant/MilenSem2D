
from typing import Any, TypedDict, Dict


class FCSrcPropertyTable(TypedDict):
    id: int
    type: int
    properties: Dict[str, Any]
    additional_properties: Dict[str, Any]


class FCPropertyTable:
    id: int
    type: int
    properties: Dict[str, Any]
    additional_properties: Dict[str, Any]

    def __init__(self, src_data: FCSrcPropertyTable):
        self.id = src_data['id']
        self.type = src_data.get('type', 0)  # тип может отсутствовать
        self.properties = src_data.get('properties', {})
        self.additional_properties = src_data.get('additional_properties', {})

    def dump(self) -> FCSrcPropertyTable:
        return {
            "id": self.id,
            "type": self.type,
            "properties": self.properties,
            "additional_properties": self.additional_properties
        }

    def __str__(self) -> str:
        return (
            f"FCPropertyTable(id={self.id}, type={self.type}, properties={self.properties}, additional_properties={self.additional_properties})"
        )

    def __repr__(self) -> str:
        return (
            f"<FCPropertyTable {self.id!r} {self.type!r}>"
        )
