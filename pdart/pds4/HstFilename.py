"""Functionality to extract data from HST filenames."""
import re

from fs.path import basename


class HstFilename(object):
    """
    A wrapper around the name of an HST file with functionality to extract
    data from the filename.
    """

    def __init__(self, filename):
        # type: (unicode) -> None
        self.filename = filename
        assert len(basename(filename)) > 6, \
            'Filename must be at least six characters long'
        basename2 = basename(filename)
        assert basename2[0].lower() in 'iju', \
            ('First char of filename %s must be i, j, or u.' % str(basename2))

    def __str__(self):
        return self.filename.__str__()

    def __repr__(self):
        return 'HstFilename(%r)' % self.filename

    def _basename(self):
        # type: () -> unicode
        return basename(self.filename)

    def instrument_name(self):
        # type: () -> str
        """
        Return the instrument name determined by the first character
        of the filename.
        """
        filename = self._basename()
        i = filename[0].lower()
        assert i in 'iju', ('First char of filename %s must be i, j, or u.'
                            % str(filename))
        if i == 'i':
            return 'wfc3'
        elif i == 'j':
            return 'acs'
        elif i == 'u':
            return 'wfpc2'
        else:
            raise Exception('First char of filename must be i, j, or u.')

    def hst_internal_proposal_id(self):
        # type: () -> str
        """
        Return the HST proposal ID determined by the three characters
        after the first of the filename.
        """
        return str(self._basename()[1:4].lower())

    def suffix(self):
        # type: () -> str
        """
        Return the suffix of the filename, that is all characters
        after the first underscore up to the period before the 'fits'
        extension.
        """
        return str(re.match(r'\A[^_]+_([^\.]+)\..*\Z',
                            self._basename()).group(1))

    def visit(self):
        # type: () -> str
        """
        Return the visit id determined by the two characters after the
        first four of the filename.
        """
        return str(self._basename()[4:6].lower())
