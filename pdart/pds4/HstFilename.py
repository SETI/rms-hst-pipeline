import os.path
import re


class HstFilename(object):
    def __init__(self, filename):
        self.filename = filename
        assert len(os.path.basename(filename)) > 6, \
            'Filename must be at least six characters long'
        basename = os.path.basename(filename)
        assert basename[0].lower() in 'iju', \
            ('First char of filename %s must be i, j, or u.' % str(basename))

    def __str__(self):
        return self.filename.__str__()

    def __repr__(self):
        return 'HstFilename(%r)' % self.filename

    def _basename(self):
        return os.path.basename(self.filename)

    def instrument_name(self):
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
        return self._basename()[1:4].lower()

    def suffix(self):
        return re.match(r'\A[^_]+_([^\.]+)\..*\Z',
                        self._basename()).group(1)

    def visit(self):
        return self._basename()[4:6].lower()
