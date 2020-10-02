from typing import Optional


class LabelError(Exception):
    """
    An Exception intended to be chained: provides the LIDVID of the
    bundle, collection, or product whose label could not be created,
    and optionally the name of the file.
    """

    def __init__(self, lidvid: str, filename: Optional[str] = None) -> None:
        self.lidvid = lidvid
        self.filename = filename

    def __repr__(self) -> str:
        if self.filename:
            return f"LabelError({self.lidvid!r}, {self.filename!r})"
        else:
            return f"LabelError({self.lidvid!r})"
