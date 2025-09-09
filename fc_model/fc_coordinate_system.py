from typing import TypedDict
from numpy.typing import NDArray

from numpy import dtype, float64

from .fc_value import decode, encode



class FCSrcCoordinateSystem(TypedDict):
    id: int
    type: str
    name: str
    origin: str
    dir1: str
    dir2: str


class FCCoordinateSystem:
    id: int
    type: str
    name: str
    origin: NDArray[float64]
    dir1: NDArray[float64]
    dir2: NDArray[float64]


    def __init__(
        self,
        src_data: FCSrcCoordinateSystem
    ):
        self.id = src_data['id']
        self.type = src_data['type']
        self.name = src_data['name']
        self.origin = decode(src_data['origin'], dtype(float64))
        self.dir1 = decode(src_data['dir1'], dtype(float64))
        self.dir2 = decode(src_data['dir2'], dtype(float64))


    def dump(self) -> FCSrcCoordinateSystem:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "origin": encode(self.origin),
            "dir1": encode(self.dir1),
            "dir2": encode(self.dir2)
        }

    def __str__(self) -> str:
        return (
            f"FCCoordinateSystem(id={self.id}, type='{self.type}', name='{self.name}', "
            f"origin={self.origin}, dir1={self.dir1}, dir2={self.dir2})"
        )

    def __repr__(self) -> str:
        return f"<FCCoordinateSystem id={self.id!r} type={self.type!r} name={self.name!r}>"
