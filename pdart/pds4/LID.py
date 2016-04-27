import re


class LID(object):
    """Representation of a PDS4 LID."""

    def __init__(self, str):
        """
        Create a LID object from a string, throwing an exception if
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
            assert re.match(allowed_chars_re, id)

        # Assign the Id fields
        self.lid = str
        self.bundle_id = ids[3]

        # Now we modify ids to include possibly missing Ids...
        while len(ids) < 6:
            ids.append(None)

        # ...so this indexing of ids is safe
        self.collection_id = ids[4]
        self.product_id = ids[5]

    def __eq__(self, other):
        return self.lid == other.lid

    def __ne__(self, other):
        return self.lid != other.lid

    def __str__(self):
        return self.lid

    def __repr__(self):
        return 'LID(%r)' % self.lid

    def is_product_lid(self):
        """Return True iff the LID is a product LID."""
        return self.product_id is not None

    def is_collection_lid(self):
        """Return True iff the LID is a collection LID."""
        return self.collection_id is not None and self.product_id is None

    def is_bundle_lid(self):
        """Return True iff the LID is a bundle LID."""
        return self.bundle_id is not None and self.collection_id is None

    def parent_lid(self):
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