from typing import Optional, Tuple

from pdart.labels.Lookup import Lookup


class LabelError(Exception):
    """
    An Exception intended to be chained: provides the LIDVID of the
    bundle, collection, or product whose label could not be created,
    and optionally the name of the file.
    """

    def __init__(
        self,
        lidvid: str,
        filename: Optional[str] = None,
        lookups: Tuple[Lookup, Lookup, Lookup] = None,
    ) -> None:
        self.lidvid = lidvid
        self.filename = filename
        self.lookups = lookups

    def __repr__(self) -> str:
        res = f"LabelError({self.lidvid!r}"
        if self.filename:
            res = res + f", {self.filename!r}"
        if self.lookups:
            res = res + f", {self.lookups}"
        res = res + ")"

        return res
