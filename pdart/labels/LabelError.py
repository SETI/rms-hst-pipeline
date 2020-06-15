from typing import Optional


class LabelError(Exception):
    """
    An Exception intended to be chained: provides the LIDVID of the
    bundle, collection, or product whose label could not be created,
    and optionally the name of the file.
    """

    def __init__(
        self, prev_msg: str, lidvid: str, filename: Optional[str] = None
    ) -> None:
        self.prev_msg = prev_msg
        self.lidvid = lidvid
        self.filename = filename

    def __str__(self) -> str:
        if self.filename:
            return f"{self.lidvid}/{self.filename}: {self.prev_msg}"
        else:
            return f"{self.lidvid}: {self.prev_msg}"
