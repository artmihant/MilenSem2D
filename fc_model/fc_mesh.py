from typing import Iterator, List, Dict, Literal, TypedDict, Union
import numpy as np
from numpy.typing import NDArray

from fc_model.fc_elements import FC_ELEMENT_TYPES_KEYID, FC_ELEMENT_TYPES_KEYNAME, FCElement, FCElementTypeLiteral

from .fc_value import decode, encode


class FCSrcMesh(TypedDict):
    elem_blocks: str
    elem_orders: str
    elem_parent_ids: str
    elem_types: str
    elemids: str
    elems: str
    elems_count: int
    nids: str
    nodes: str
    nodes_count: int


class FCMesh:
    """
    Контейнер для хранения всех элементов модели, сгруппированных по типам.

    Внутри коллекции элементы хранятся в словаре `elements`,
    где ключами являются строковые имена типов элементов (e.g., 'HEX8', 'TETRA4'),
    а значениями — словари `Dict[int, FCElement]` соответствующего типа.

    Этот класс также управляет общей кодировкой и декодировкой всего набора
    элементов в/из формата .fc.
    """

    nodes_ids: NDArray[np.int32]
    nodes_xyz: NDArray[np.float64]

    elements: Dict[FCElementTypeLiteral, Dict[int, FCElement]]


    def __init__(self) -> None:

        self.nodes_ids = np.array([], dtype=np.int32)
        self.nodes_xyz = np.array([], dtype=np.float64)

        self.elements = {}


    def decode(self, src_mesh: FCSrcMesh) -> None:

        self.nodes_ids = decode(src_mesh['nids'], np.dtype('int32'))
        nodes_raw = decode(src_mesh['nodes'], np.dtype('float64'))
        if nodes_raw.size % 3 != 0:
            raise ValueError(f"mesh.nodes length must be divisible by 3, got {nodes_raw.size}")
        self.nodes_xyz = nodes_raw.reshape(-1, 3)

        # basic consistency: nodes_count must match ids/xyz lengths
        if src_mesh['nodes_count'] != len(self.nodes_ids):
            raise ValueError(
                f"nodes_count mismatch: header {src_mesh['nodes_count']} vs decoded {len(self.nodes_ids)}"
            )
        if self.nodes_xyz.shape[0] != len(self.nodes_ids):
            raise ValueError(
                f"nodes xyz count mismatch: {self.nodes_xyz.shape[0]} rows vs {len(self.nodes_ids)} ids"
            )

        elem_blocks: NDArray[np.int32] = decode(src_mesh['elem_blocks'])
        elem_orders: NDArray[np.int32] = decode(src_mesh['elem_orders'])
        elem_parent_ids: NDArray[np.int32] = decode(src_mesh['elem_parent_ids'])
        elem_types: NDArray[np.uint8] = decode(src_mesh['elem_types'], np.dtype(np.uint8))
        elem_ids: NDArray[np.int32] = decode(src_mesh['elemids'])
        elem_nodes: NDArray[np.int32] = decode(src_mesh['elems'])

        # basic consistency: elems_count must match arrays' lengths
        elems_count = src_mesh['elems_count']
        for name, arr in (
            ("elemids", elem_ids),
            ("elem_blocks", elem_blocks),
            ("elem_orders", elem_orders),
            ("elem_parent_ids", elem_parent_ids),
            ("elem_types", elem_types),
        ):
            if len(arr) != elems_count:
                raise ValueError(f"{name} length {len(arr)} != elems_count {elems_count}")

        elem_sizes = np.vectorize(lambda t: FC_ELEMENT_TYPES_KEYID[t]['nodes'])(elem_types)
        total_nodes = int(np.sum(elem_sizes))
        if len(elem_nodes) != total_nodes:
            raise ValueError(
                f"elems (flattened nodes) length {len(elem_nodes)} != expected {total_nodes} from elem_types"
            )
        elem_offsets = [0, *np.cumsum(elem_sizes)]

        for i, eid in enumerate(elem_ids):
            fc_type_name = FC_ELEMENT_TYPES_KEYID[elem_types[i]]['name']
            eid = int(eid)
            if fc_type_name not in self.elements:
                self.elements[fc_type_name] = {}

            element = FCElement({
                'id': eid,
                'type': fc_type_name,
                'nodes': elem_nodes[elem_offsets[i]:elem_offsets[i+1]].tolist(),
                'parent_id': elem_parent_ids[i],
                'block': elem_blocks[i],
                'order': elem_orders[i],
            })

            self.elements[fc_type_name][eid] = element


    def encode(self) -> FCSrcMesh:

        elems_count = len(self)

        elem_ids: NDArray[np.int32] = np.zeros(elems_count, np.int32)
        elem_blocks: NDArray[np.int32] = np.zeros(elems_count, np.int32)
        elem_orders: NDArray[np.int32] = np.zeros(elems_count, np.int32)
        elem_parent_ids: NDArray[np.int32] = np.zeros(elems_count, np.int32)
        elem_types: NDArray[np.int8] = np.zeros(elems_count, np.int8)

        # basic consistency: nodes arrays
        if self.nodes_xyz.ndim != 2 or self.nodes_xyz.shape[1] != 3:
            raise ValueError("nodes must be a 2D array of shape (N,3)")
        if len(self.nodes_ids) != self.nodes_xyz.shape[0]:
            raise ValueError(
                f"nodes ids length {len(self.nodes_ids)} != nodes xyz rows {self.nodes_xyz.shape[0]}"
            )

        for i, elem in enumerate(self):
            elem_ids[i] = elem.id
            elem_blocks[i] = elem.block
            elem_parent_ids[i] = elem.parent_id
            elem_orders[i] = elem.order
            elem_types[i] = FC_ELEMENT_TYPES_KEYNAME[elem.type]['fc_id']

        # basic consistency: each element nodes count must match its type definition
        expected_nodes_total = 0
        for elem in self:
            expected_nodes_total += FC_ELEMENT_TYPES_KEYNAME[elem.type]['nodes']
        elem_nodes: NDArray[np.int32] = np.array(self.nodes_list, np.int32)
        if len(elem_nodes) != expected_nodes_total:
            raise ValueError(
                f"flattened nodes length {len(elem_nodes)} != expected {expected_nodes_total} from element types"
            )

        src_mesh: FCSrcMesh = {
            "elem_blocks": encode(elem_blocks),
            "elem_orders": encode(elem_orders),
            "elem_parent_ids": encode(elem_parent_ids),
            "elem_types": encode(elem_types),
            "elemids": encode(elem_ids),
            "elems": encode(elem_nodes),
            "elems_count": elems_count,
            "nids": encode(self.nodes_ids), 
            "nodes": encode(self.nodes_xyz),
            "nodes_count": len(self.nodes_ids)           
        }

        return src_mesh


    def __len__(self) -> int:
        return sum([len(self.elements[typename]) for typename in self.elements])


    def __bool__(self) -> bool:
        return len(self) > 0


    def __iter__(self) -> Iterator[FCElement]:
        for typename in self.elements:
            for elem in self.elements[typename].values():
                yield elem


    def __contains__(self, key: Union[int, FCElementTypeLiteral]) -> bool:
        for tp in self.elements:
            if key in self.elements[tp]:
                return True
        return False


    def __getitem__(self, key: Union[int, FCElementTypeLiteral]) -> Union[Dict[int, FCElement], FCElement]:
        if isinstance(key, str):
            return self.elements[key]
        elif isinstance(key, int):
            for typename in self.elements:
                if key in self.elements[typename]:
                    return self.elements[typename][key]
        raise KeyError(f'{key}')


    def __setitem__(self, key: int, item: FCElement) -> None:

        if item.type not in self.elements:
            self.elements[item.type] = {}
        self.elements[item.type][item.id] = item


    @property
    def nodes_list(self) -> List[int]:
        return [node for elem in self for node in elem.nodes]


    def compress(self) -> Dict[int, int]:
        index_map = {elem.id: i + 1 for i, elem in enumerate(self)}
        self.reindex(index_map)
        return index_map


    def reindex(self, index_map: Dict[int, int]) -> None:
        for typename in list(self.elements.keys()):
            new_bucket: Dict[int, FCElement] = {}
            for elem in self.elements[typename].values():
                if elem.id in index_map:
                    elem.id = index_map[elem.id]
                new_bucket[elem.id] = elem
            self.elements[typename] = new_bucket


    @property
    def max_id(self) -> int:
        max_id = 0
        for tp in self.elements:
            if self.elements[tp]:
                local_max = max(self.elements[tp].keys())
                if max_id < local_max:
                    max_id = local_max
        return max_id


    def add(self, item: FCElement) -> int:
        if item.type not in self.elements:
            self.elements[item.type] = {}
        if item.id in self or item.id < 1:
            item.id = self.max_id + 1
        self.elements[item.type][item.id] = item
        return item.id


    def __str__(self) -> str:
        return (
            f"FCMesh(nodes_ids={self.nodes_ids}, nodes_xyz={self.nodes_xyz}, elements={self.elements})"
        )


    def __repr__(self) -> str:
        return (
            f"<FCMesh nodes:{len(self.nodes_ids)} elements:{len(self)}>"
        )
