import abc
from typing import Any, Iterable, List, Mapping, Sized, Union

from astropy.table.row import Row

class Table(Iterable[Row], Sized, metaclass=abc.ABCMeta):
    def copy(self): ...
    colnames: List[str]
    def __getitem__(self, key: Union[int, str]) -> Row: ...
