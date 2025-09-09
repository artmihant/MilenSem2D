from typing import TypedDict, List

from numpy import dtype, int32
from .fc_value import FCValue


class FCSrcReceiver(TypedDict):
    id: int
    name: str
    apply_to: str
    apply_to_size: int
    dofs: List[int]
    type: int


class FCReceiver:
    id: int
    apply: FCValue
    dofs: List[int]
    name: str
    type: int

    def __init__(self, src_data: FCSrcReceiver):
        self.apply = FCValue(src_data['apply_to'], dtype(int32))
        self.apply.resize(src_data['apply_to_size'])
        self.id = src_data['id']
        self.name = src_data['name']
        self.dofs = src_data['dofs']
        self.type = src_data['type']

    def dump(self) -> FCSrcReceiver:
        return {
            "apply_to": self.apply.dump(),
            "apply_to_size": len(self.apply),
            "id": self.id,
            "name": self.name,
            "dofs": self.dofs,
            "type": self.type
        }

    def __str__(self) -> str:
        return (
            f"FCReceiver(id={self.id}, name='{self.name}', dofs={self.dofs}, type={self.type}, apply_to_size={len(self.apply)})"
        )

    def __repr__(self) -> str:
        return (
            f"<FCReceiver {self.id!r} {self.name!r}>"
        )
