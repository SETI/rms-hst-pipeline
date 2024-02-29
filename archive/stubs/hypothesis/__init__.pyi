from typing import Any, Callable, Union

from hypothesis.strategies._internal.strategies import SearchStrategy
from hypothesis.utils.conventions import InferType

def assume(condition: Any) -> bool:
    pass

def given(
    *_given_arguments: Union[SearchStrategy, InferType],
    **_given_kwargs: Union[SearchStrategy, InferType]
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    pass
