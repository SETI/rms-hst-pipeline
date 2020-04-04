from hypothesis.strategies._internal.core import builds, from_regex, integers, lists
from hypothesis.strategies._internal.strategies import SearchStrategy

def composite(func):
    return func

__all__ = ["builds", "from_regex", "integers", "lists", "SearchStrategy"]
