from base64 import b64decode, b64encode
import binascii
from typing import Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

FCValueTypeLiteral = Literal['formula', 'array', 'null']

def isBase64(sb: str) -> bool:
    """Проверяет, является ли строка корректной base64 (строгая проверка)."""
    try:
        if isinstance(sb, str):
            sb_bytes = bytes(sb, 'ascii')
        elif isinstance(sb, bytes):
            sb_bytes = sb
        else:
            raise ValueError("Argument must be string or bytes")

        # Длина base64 строки должна быть кратна 4
        if len(sb_bytes) % 4 != 0:
            return False

        decoded = b64decode(sb_bytes, validate=True)
        return b64encode(decoded) == sb_bytes
    except (TypeError, binascii.Error, ValueError):
        return False


from typing import TypeVar, Generic
import numpy as np
from numpy.typing import NDArray

T = TypeVar('T', bound=np.generic)

def decode(src: str, dtype: Optional[np.dtype[T]] = None) -> NDArray[T]:
    """Декодирует строку base64 в numpy массив с заданным типом данных."""
    if src == '':
        return np.array([], dtype=dtype if dtype else np.dtype('int32')) 
    data = b64decode(src, validate=True)
    return np.frombuffer(data, dtype if dtype else np.dtype('int32'))

def encode(data: NDArray[np.generic]) -> str:
    """Кодирует numpy массив в строку base64."""
    return b64encode(data.tobytes()).decode()

class FCValue:

    type: FCValueTypeLiteral = 'null'
    data: Union[NDArray[np.generic], str]

    def __init__(self, src_data: Union[NDArray[np.generic], str], dtype:np.dtype[np.generic] = np.dtype('int32'), value_type: FCValueTypeLiteral='array'):

        if isinstance(src_data, str):

            if value_type == 'array':

                if src_data == '':
                    self.data = np.array([], dtype=dtype)
                    self.type = 'null'
                elif isBase64(src_data):
                    # Строгое распознавание base64 прошло — дополнительно проверим кратность буфера типу
                    raw = b64decode(src_data, validate=True)
                    if len(raw) % dtype.itemsize != 0:
                        # Не соответствует типу — трактуем как формулу
                        self.data = src_data
                        self.type = 'formula'
                    else:
                        self.data = np.frombuffer(raw, dtype)
                        self.type = 'array'
                else:
                    self.data = src_data
                    self.type = 'formula'

            elif value_type == 'null':
                self.data = np.array([], dtype=dtype)
                self.type = 'null'
            elif value_type == 'formula':
                self.data = src_data
                self.type = 'formula'

        else:
            self.data = src_data
            self.type = 'array'

    def resize(self, size: int) -> None:
        if isinstance(self.data, np.ndarray) and size > 0 and self.data.size % size == 0:
            self.data = self.data.reshape(size, -1)

    def dump(self) -> str:
        if isinstance(self.data, np.ndarray):
            return encode(self.data)
        else:
            return self.data

    def __len__(self) -> int:
        if self.type == 'array':
            return len(self.data)
        else:
            return 0

    def __str__(self) -> str:
        return f"FCValue(type={self.type}, data={self.data})"

    def __repr__(self) -> str:
        data_str = ''
        if isinstance(self.data, np.ndarray):
            data_str = f'NDArray<{self.data.dtype}>{self.data.shape}'
        else:
            data_str = self.data

        return f"<FCValue {data_str}>"
