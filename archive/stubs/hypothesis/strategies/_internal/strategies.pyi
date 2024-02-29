from typing import Callable, Generic, TypeVar

Ex = TypeVar("Ex", covariant=True)
T = TypeVar("T")

class SearchStrategy(Generic[Ex]):
    def map(self, pack: Callable[[Ex], T]) -> "SearchStrategy[T]":
        pass
