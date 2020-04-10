import abc
from typing import Iterable, List, Sized, Union

from astropy.table.row import Row

class Table(Iterable[Row], Sized, metaclass=abc.ABCMeta):
    def copy(self) -> Table: ...
    colnames: List[str]
    def __getitem__(self, key: Union[int, str]) -> Row: ...
    def remove_rows(self, rows: List[int]) -> None: ...
