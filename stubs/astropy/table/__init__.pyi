from typing import Any, Iterable, Mapping, Sized, List, Union
import abc
from astropy.table.row import Row

class Table(Iterable[Row], Sized, metaclass=abc.ABCMeta):
    def copy(self): ...
    colnames: List[str]
    def __getitem__(self, key: Union[int, str]) -> Row: ...
