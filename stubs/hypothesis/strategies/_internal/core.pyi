from typing import Any, AnyStr, Callable, Hashable, List, Pattern, Tuple, Union
from hypothesis.strategies._internal.strategies import Ex, SearchStrategy
from hypothesis.utils.conventions import InferType

UniqueBy = Union[Callable[[Ex], Hashable], Tuple[Callable[[Ex], Hashable], ...]]

def builds(
    *callable_and_args: Union[Callable[..., Ex], SearchStrategy[Any]],
    **kwargs: Union[SearchStrategy[Any], InferType]
) -> SearchStrategy[Ex]:
    pass

def from_regex(
    regex: Union[AnyStr, Pattern[AnyStr]], *, fullmatch: bool = False
) -> SearchStrategy[AnyStr]:
    pass

def integers(min_value: int = None, max_value: int = None) -> SearchStrategy[int]:
    pass

def lists(
    elements: SearchStrategy[Ex],
    *,
    min_size: int = 0,
    max_size: int = None,
    unique_by: UniqueBy = None,
    unique: bool = False
) -> SearchStrategy[List[Ex]]:
    pass
