"""Representation of a PDS4 LID."""
from typing import TYPE_CHECKING
import re

if TYPE_CHECKING:
    from typing import List, Optional


class LID(object):
    """Representation of a PDS4 LID."""

    def __init__(self, str):
        # type: (str) -> None
        """
        Create a LID object from a string, raising an exception if
        the LID string is malformed.
        """

        ids = str.split(':')

        # Check requirements
        assert len(str) <= 255
        assert len(ids) in [4, 5, 6]
        assert ids[0] == 'urn'
        assert ids[1] == 'nasa'
        assert ids[2] == 'pds'
        allowed_chars_re = r'\A[-._a-z0-9]+\Z'
        for id in ids:
            assert re.match(allowed_chars_re, id), id

        # Assign the Id fields
        self.lid = str
        self.bundle_id = ids[3]

        # ...so this indexing of ids is safe
        self.collection_id = ids[4] if len(ids) > 4 else None
        self.product_id = ids[5] if len(ids) > 5 else None

    @staticmethod
    def create_from_parts(parts):
        # type: (List[str]) -> LID
        parts_len = len(parts)
        assert parts_len in [1, 2, 3], parts
        if parts_len == 1:
            return LID('urn:nasa:pds:%s' % parts[0])
        elif parts_len == 2:
            b, c = parts
            return LID('urn:nasa:pds:%s:%s' % (b, c))
        else:
            b, c, p = parts
            return LID('urn:nasa:pds:%s:%s:%s' % (b, c, p))

    def parts(self):
        # type: () -> List[str]
        ids = [self.bundle_id, self.collection_id, self.product_id]
        return [p for p in ids if p is not None]

    def __cmp__(self, other):
        if other is None:
            return 1
        return cmp(str(self), str(other))

    def __hash__(self):
        return hash(self.lid)

    def __str__(self):
        return self.lid

    def __repr__(self):
        return 'LID(%r)' % self.lid

    def is_product_lid(self):
        # type: () -> bool
        """Return True iff the LID is a product LID."""
        return self.product_id is not None

    def is_collection_lid(self):
        # type: () -> bool
        """Return True iff the LID is a collection LID."""
        return self.collection_id is not None and self.product_id is None

    def is_bundle_lid(self):
        # type: () -> bool
        """Return True iff the LID is a bundle LID."""
        return self.bundle_id is not None and self.collection_id is None

    def parent_lid(self):
        # type: () -> LID
        """
        Return a LID object for the object's parent.  Throw an error
        iff the object is a bundle LID.
        """
        if self.is_bundle_lid():
            raise ValueError('bundle LID %r has no parent LID' %
                             self.lid)
        else:
            parts = self.lid.split(':')
            return LID(':'.join(parts[:-1]))

    def to_browse_lid(self):
        # type: () -> LID
        """
        Convert a LID within a data collection into the corresponding
        LID in the browse collection.
        """
        assert self.collection_id, 'to_browse_lid(): Can\'t call on bundle LID'
        collection_id_parts = self.collection_id.split('_')
        assert collection_id_parts[0] == 'data', \
            'to_browse_lid: Only legal within data_ collections; had %s' % self
        collection_id_parts[0] = 'browse'
        browse_collection_id = '_'.join(collection_id_parts)

        lid_parts = self.lid.split(':')
        lid_parts[4] = browse_collection_id
        browse_collection_lid = ':'.join(lid_parts)
        return LID(browse_collection_lid)

    def to_shm_lid(self):
        # type: () -> LID
        """
        Convert a product LID into the corresponding LID for a SHM file.
        """
        assert self.collection_id, 'to_shm_lid(): Can\'t call on bundle LID'
        collection_id_parts = self.collection_id.split('_')
        assert collection_id_parts[0] == 'data', \
            'to_shm_lid: Only legal within data_ collections; had %s' % self
        # replace the suffix
        collection_id_parts[2] = 'shm'
        shm_collection_id  = '_'.join(collection_id_parts)

        lid_parts = self.lid.split(':')
        lid_parts[4] = shm_collection_id
        shm_product_lid = ':'.join(lid_parts)
        return LID(shm_product_lid)

    def extend_lid(self, segment):
        # type: (str) -> LID
        """
        Create a new LID by extending this one with  another segment.
        """
        ps = self.parts()
        ps.append(segment)
        return LID.create_from_parts(ps)
