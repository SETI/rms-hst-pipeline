"""Functionality to extract data from HST filenames."""
import re

from fs.path import basename

from pdart.pipeline.SuffixInfo import (  # type: ignore
    ACCEPTED_LETTER_CODES,
    INSTRUMENT_FROM_LETTER_CODE,
)


class HstFilename(object):
    """
    A wrapper around the name of an HST file with functionality to extract
    data from the filename.
    """

    def __init__(self, filename: str) -> None:
        self.filename = filename
        assert (
            len(basename(filename)) > 6
        ), "Filename must be at least six characters long"
        basename2 = basename(filename)
        assert (
            basename2[0].lower() in ACCEPTED_LETTER_CODES
        ), f"First char of filename {basename2!r} must be in {ACCEPTED_LETTER_CODES!r}."

    def __str__(self) -> str:
        return self.filename.__str__()

    def __repr__(self) -> str:
        return f"HstFilename({self.filename!r})"

    def _basename(self) -> str:
        return basename(self.filename)

    def instrument_name(self) -> str:
        """
        Return the instrument name determined by the first character
        of the filename.
        """
        filename = self._basename()
        i = filename[0].lower()
        assert (
            i in ACCEPTED_LETTER_CODES
        ), f"First char of filename {filename!r} must be in {ACCEPTED_LETTER_CODES!r}."
        try:
            return INSTRUMENT_FROM_LETTER_CODE[i]
        except KeyError:
            raise Exception(
                f"First char of filename {filename!r} must be in {ACCEPTED_LETTER_CODES!r}."
            )

    def hst_internal_proposal_id(self) -> str:
        """
        Return the HST proposal ID determined by the three characters
        after the first of the filename.
        """
        return str(self._basename()[1:4].lower())

    def rootname(self) -> str:
        """
        Return the "rootname" of the filename, that is all characters
        before the underscore of the suffix.  This is used in
        association tables.
        """
        match = re.match(r"\A([^_]+)_.*\Z", self._basename())
        assert match
        return str(match.group(1)).lower()

    def suffix(self) -> str:
        """
        Return the suffix of the filename, that is all characters
        after the first underscore up to the period before the 'fits'
        extension.
        """
        match = re.match(r"\A[^_]+_([^.]+)\..*\Z", self._basename())
        assert match
        return str(match.group(1)).lower()

    def visit(self) -> str:
        """
        Return the visit id determined by the two characters after the
        first four of the filename.
        """
        return str(self._basename()[4:6].lower())
