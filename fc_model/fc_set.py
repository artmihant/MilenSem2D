from typing import TypedDict

from numpy import dtype, int32

from .fc_value import FCValue


class FCSrcSet(TypedDict):
    id: int
    name: str
    apply_to: str
    apply_to_size: int


class FCSet:
    id: int
    apply: FCValue
    name: str

    def __init__(self, src_data: FCSrcSet):
        self.apply = FCValue(src_data['apply_to'], dtype(int32))
        self.apply.resize(src_data['apply_to_size'])
        self.id = src_data['id']
        self.name = src_data['name']

    def dump(self) -> FCSrcSet:
        return {
            "apply_to": self.apply.dump(),
            "apply_to_size": len(self.apply),
            "id": self.id,
            "name": self.name
        }

    def __str__(self) -> str:
        return f"FCSet(id={self.id}, name='{self.name}', apply_to_size={len(self.apply)})"

    def __repr__(self) -> str:
        return f"<FCSet{'['+self.name+']' if self.name else ''} {self.id!r} {self.apply.__repr__()}>"
