import abc
from typing import Any, Iterable, Union

class Row(Iterable[Any], metaclass=abc.ABCMeta):
    def __getitem__(self, key: Union[int, str]) -> Any: ...
