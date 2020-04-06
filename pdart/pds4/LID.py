"""Representation of a PDS4 LID."""
import functools
import re

from typing import List


@functools.total_ordering
class LID(object):
    """Representation of a PDS4 LID."""

    def __init__(self, lid_str: str) -> None:
        """
        Create a LID object from a string, raising an exception if
        the LID string is malformed.
        """

        ids = lid_str.split(":")

        # Check requirements
        assert len(lid_str) <= 255
        assert len(ids) in [4, 5, 6]
        assert ids[0] == "urn"
        assert ids[1] == "nasa"
        assert ids[2] == "pds"
        allowed_chars_re = r"\A[-._a-z0-9]+\Z"
        for id_ in ids:
            assert re.match(allowed_chars_re, id_), id_

        # Assign the Id fields
        self.lid = lid_str
        self.bundle_id = ids[3]

        # ...so this indexing of ids is safe
        self.collection_id = ids[4] if len(ids) > 4 else None
        self.product_id = ids[5] if len(ids) > 5 else None

    @staticmethod
    def create_from_parts(parts: List[str]) -> "LID":
        parts_len = len(parts)
        assert parts_len in [1, 2, 3], parts
        if parts_len == 1:
            return LID(f"urn:nasa:pds:{parts[0]}")
        elif parts_len == 2:
            return LID(f"urn:nasa:pds:{parts[0]}:{parts[1]}")
        else:
            return LID(f"urn:nasa:pds:{parts[0]}:{parts[1]}:{parts[2]}")

    def parts(self) -> List[str]:
        ids = [self.bundle_id, self.collection_id, self.product_id]
        return [p for p in ids if p is not None]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, LID):
            return NotImplemented
        return str(self) < str(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LID):
            return NotImplemented
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(self.lid)

    def __str__(self) -> str:
        return self.lid

    def __repr__(self) -> str:
        return f"LID({self.lid!r})"

    def is_product_lid(self) -> bool:
        """Return True iff the LID is a product LID."""
        return self.product_id is not None

    def is_collection_lid(self) -> bool:
        """Return True iff the LID is a collection LID."""
        return self.collection_id is not None and self.product_id is None

    def is_bundle_lid(self) -> bool:
        """Return True iff the LID is a bundle LID."""
        return self.collection_id is None

    def parent_lid(self) -> "LID":
        """
        Return a LID object for the object's parent.  Throw an error
        iff the object is a bundle LID.
        """
        if self.is_bundle_lid():
            raise ValueError(f"bundle LID {self.lid!r} has no parent LID")
        else:
            parts = self.lid.split(":")
            return LID(":".join(parts[:-1]))

    def to_browse_lid(self) -> "LID":
        """
        Convert a LID within a data collection into the corresponding
        LID in the browse collection.
        """
        assert self.collection_id, "to_browse_lid() -> None: Can't call on bundle LID"
        collection_id_parts = self.collection_id.split("_")
        assert (
            collection_id_parts[0] == "data"
        ), f"to_browse_lid: Only legal within data_ collections; had {self}"
        collection_id_parts[0] = "browse"
        browse_collection_id = "_".join(collection_id_parts)

        lid_parts = self.lid.split(":")
        lid_parts[4] = browse_collection_id
        browse_collection_lid = ":".join(lid_parts)
        return LID(browse_collection_lid)

    def to_shm_lid(self) -> "LID":
        """
        Convert a product LID into the corresponding LID for a SHM file.
        """
        assert self.collection_id, "to_shm_lid(): Can't call on bundle LID"
        collection_id_parts = self.collection_id.split("_")
        assert (
            collection_id_parts[0] == "data"
        ), f"to_shm_lid: Only legal within data_ collections; had {self}"
        # replace the suffix
        collection_id_parts[2] = "shm"
        shm_collection_id = "_".join(collection_id_parts)

        lid_parts = self.lid.split(":")
        lid_parts[4] = shm_collection_id
        shm_product_lid = ":".join(lid_parts)
        return LID(shm_product_lid)

    def extend_lid(self, segment: str) -> "LID":
        """
        Create a new LID by extending this one with  another segment.
        """
        ps = self.parts()
        ps.append(segment)
        return LID.create_from_parts(ps)
