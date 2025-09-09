from typing import Any, TypedDict, Dict, Union

from numpy import dtype, int32

from .fc_value import FCValue


class FCSrcConstraint(TypedDict):
    id: int
    name: str
    type: Union[int, str]
    master: str
    master_size: int
    slave: str
    slave_size: int


class FCConstraint:
    id: int
    name: str
    type: Union[int, str]
    master: FCValue
    slave: FCValue
    properties: Dict[str, Any]

    def __init__(
        self,
        src_data: FCSrcConstraint
    ):
        self.id = src_data['id']
        self.name = src_data['name']
        self.type = src_data['type']
        
        self.master = FCValue(src_data['master'], dtype(int32))
        self.master.resize(src_data['master_size'])
        self.slave = FCValue(src_data['slave'], dtype(int32))
        self.slave.resize(src_data['slave_size'])
        
        self.properties = {
                    key: src_data[key] for key in src_data #type:ignore
                    if key not in FCSrcConstraint.__annotations__.keys()}
        

    def dump(self) -> FCSrcConstraint:

        src_constraint: FCSrcConstraint = {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "master": self.master.dump(),
            "slave": self.slave.dump(),
            "master_size": len(self.master),
            "slave_size": len(self.slave),
        }

        for key in self.properties:
            src_constraint[key] = self.properties[key] #type:ignore

        return src_constraint


    def __str__(self) -> str:
        return (
            f"FCConstraint(id={self.id}, name='{self.name}', type={self.type}, "
            f"master={self.master}, slave={self.slave}, properties={self.properties})"
        )

    def __repr__(self) -> str:
        return f"<FCConstraint {self.id!r} {self.name!r} {self.type!r}>"