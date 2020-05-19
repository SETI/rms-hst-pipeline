from typing import Any, Callable, Dict, List, TypeVar
import abc


class Lookup(abc.ABC):
    """
    A strategy for looking up a key value.  Subclasses may look in a
    single set of cards from a file, or may look at multiple files, or
    do something completely different.
    """

    @abc.abstractmethod
    def __getitem__(self, key: str) -> Any:
        pass

    def keys(self, keys: List[str]) -> List[Any]:
        """Default implementation is to look up each key separated."""
        return list(map(self.__getitem__, keys))


class DictLookup(Lookup):
    """
    Look up a key value in a set of card dictionaries from one file.
    """

    def __init__(self, card_dicts: List[Dict[str, Any]]) -> None:
        self.card_dicts = card_dicts

    def __getitem__(self, key: str) -> Any:
        return self.card_dicts[0][key]


R = TypeVar("R")
D = TypeVar("D")


def _multi_lookup(f: Callable[[D], R], args: List[D]) -> R:
    """
    Runs a function over a list, returning the first successful result
    or raising the exception on the last try if there are no
    successes.
    """
    first_args = args[:-1]
    last_arg = args[-1]
    for arg in first_args:
        try:
            return f(arg)
        except KeyError:
            continue
    return f(last_arg)


class TripleDictLookup(Lookup):
    """
    Look up a key value over sets of card dictionaries from three
    files.
    """

    def __init__(
        self,
        card_dicts: List[Dict[str, Any]],
        raw_card_dicts: List[Dict[str, Any]],
        shm_card_dicts: List[Dict[str, Any]],
    ) -> None:
        self.card_dicts = card_dicts
        self.raw_card_dicts = raw_card_dicts
        self.shm_card_dicts = shm_card_dicts

    def __getitem__(self, key: str) -> Any:
        """Returns the first successful lookup."""

        def find_key(d: List[Dict[str, Any]]) -> Any:
            return d[0][key]

        return _multi_lookup(
            find_key, [self.card_dicts, self.raw_card_dicts, self.shm_card_dicts]
        )

    def keys(self, keys: List[str]) -> List[Any]:
        """Returns the first successful lookup."""

        def find_keys(d: List[Dict[str, Any]]) -> List[Any]:
            return DictLookup(d).keys(keys)

        return _multi_lookup(
            find_keys, [self.card_dicts, self.raw_card_dicts, self.shm_card_dicts]
        )
