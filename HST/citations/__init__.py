##########################################################################################
# Citation_Information class for HST pipeline
##########################################################################################

import re
import textwrap
from xml.sax.saxutils import escape

from .citation_information_from_apt import citation_information_from_apt
from .citation_information_from_pro import citation_information_from_pro
from .fix_title import fix_title
from .fix_authors import fix_authors
from .fix_abstract import fix_abstract

YEARS_FOR_CYCLE = {  # YEARS_FOR_CYCLE[cycle_number] = (start_year, end_year)
    1: (1991, 1992),  # https://books.google.com/books?id=iy7HKCO9vO0C
    2: (1992, 1993),
    3: (
        1993,
        1993,
    ),  # https://www.google.com/books/edition/Hubble_Space_Telescope/xnPvAAAAMAAJ
    # SM1 = December 1993
    4: (
        1994,
        1995,
    ),  # http://s92034.eos-intl.net/elibsql11_S92034_Documents/04_Call%20for%20Proposals.pdf
    5: (1995, 1996),
    6: (
        1996,
        1997,
    ),  # http://s92034.eos-intl.net/elibsql11_S92034_Documents/02_Call%20for%20proposals.pdf
    # SM2 = February 1997
    7: (1997, 1999),  # https://mars.nasa.gov/MPF/mpf/hst.html
    # SM3a = December 1999
    8: (
        1999,
        2000,
    ),  # http://s92034.eos-intl.net/elibsql11_S92034_Documents/02_Phase%20I%20Proposal%20Instructions.pdf
    9: (2000, 2001),
    10: (2001, 2002),
    # SM3b = March 2002
    11: (2002, 2003),  # https://books.google.com/books?id=B3HvAAAAMAAJ&pg=RA13-PA1
    12: (
        2003,
        2004,
    ),  # https://sci.esa.int/web/hubble/-/33851-begin-of-cycle-12-observations
    13: (2004, 2005),
    14: (2005, 2006),  # https://sci.esa.int/web/hubble/-/36784-hubble-status-report
    15: (2006, 2007),
    16: (2008, 2009),
    17: (
        2009,
        2010,
    ),  # http://documents.stsci.edu/hst/proposing/documents/cp_cy17/1_General_Information3.html
    # SM4 = May 2009
    18: (2010, 2011),
    19: (2011, 2012),
    20: (2012, 2013),
    21: (2013, 2014),
    22: (2014, 2015),
    23: (2015, 2016),
    24: (2016, 2017),
    25: (2017, 2018),
    26: (2018, 2019),
    27: (2019, 2020),
    28: (2020, 2021),
    29: (2021, 2022),
}

# Servicing Missions:
# SM1 = December 1993
# SM2 = February 1997
# SM3a = December 1999
# SM3b = March 2002
# SM4 = May 2009

# PUBLICATION_YEARS[proposal_number] = publication_year
# Can be edited manually to override publication_year errors inferred from the
# .PRO or .APT files.
PUBLICATION_YEARS = {
    5466: 1995,
    6840: 1996,
    6880: 1998,
    9391: 2002,
}

##########################################################################################
##########################################################################################

class Citation_Information:
    """This class encapsulates the information needed for the PDS4 Citation_Information
    object, as extracted from either an APT file or a .PRO file.

    The function create_from_file takes two inputs:
        - name of the file (which must end in .apt or .pro)
        - version ID of the HST pipeline, which is included in the returned information

    The returned Citation_Information object has the following properties, which return
    attributes appropriate to the PDS4 Citation_Information object.
        author_list
        editor_list
        publication_year
        keywords (currently always an empty list)
        description (the full bibliographic citation)
    Values are not "escaped" for XML!

    Important note about years:

    The publication year in the citation is supposed to be the year of the last
    observation in the program. Unfortunately, this cannot be reliably determined from the
    .apt or .pro files. This routine implements a number of work-arounds to provide this
    information:

    1. Call method set_publication_year to set the correct year. This is the preferred
       method but requires an external procedure to determine the year.
    2. Hard-wire a value into the global dictionary PUBLICATION_YEARS.
    3. Otherwise, the end year of the cycle will be used.
    """

    PIPELINE_VERSION = ''   # Global definition

    def __init__(self, filename, propno, category, cycle, authors, title,
                       submission_year, timing_year, abstract=''):

        self.filename = filename
        self.propno = propno
        self.category = category
        self.cycle = cycle
        self.authors = authors
        self.title = title
        self.abstract = abstract

        # Latest submission year of the proposal if known, otherwise zero
        self.submission_year = submission_year

        # Latest year defined by a BEFORE/AFTER/BETWEEN limit, otherwise zero
        self.timing_year = timing_year  # defined by BEFORE/AFTER/BETWEEN

        # Year set externally via a call to set_publication_year
        self.pub_year = 0

    def write(self, filename):
        """Write the citation info into a text file."""

        with open(filename, 'w', encoding='UTF-8') as f:
            f.write(self.filename)
            f.write('\n')
            f.write(str(self.propno))
            f.write('\n')
            f.write(self.category)
            f.write('\n')
            f.write(', '.join(self.authors))
            f.write('\n')
            f.write(self.title)
            f.write('\n')
            if isinstance(self.abstract, list):
                for line in self.abstract:
                    f.write(str(line))
                    f.write('\n')
            else:
                f.write(self.abstract)
                f.write('\n')
            f.write(str(self.submission_year))
            f.write('\n')
            f.write(str(self.timing_year))
            f.write('\n')
            f.write(str(self.pub_year))
            f.write('\n')

    @staticmethod
    def read(filename):
        """Read the citation info from a text file."""

        with open(filename, encoding='UTF-8') as f:
            fname = f.readline().rstrip()
            propno = int(f.readline())
            category = f.readline().rstrip()
            authors = f.readline().rstrip().split(', ')
            title = f.readline().rstrip()
            abstract = f.readline().rstrip()
            submission_year = int(f.readline())
            timing_year = int(f.readline())
            pub_year = int(f.readline())

        citation = Citation_Information(fname, propno, category, cycle, authors, title,
                                        submission_year, timing_year, abstract)
        citation.set_publication_year(pub_year)
        return citation

    @staticmethod
    def create_from_file(filename, pipeline_version=None):

        # Read the file and extract the key info
        filename_lc = filename.lower()
        if filename_lc.endswith('.apt'):
            info = citation_information_from_apt(filename)
        elif filename_lc.endswith('.pro'):
            info = citation_information_from_pro(filename)
        elif filename_lc.endswith('.txt'):
            info = citation_information_from_text(filename)
        else:
            raise ValueError('unrecognized file format: ' + filename)

        (propno, category, cycle, authors, title,
         submission_year, timing_year, abstract) = info

        # Cleanup
        title = fix_title(title)
        authors = fix_authors(authors)
        abstract = fix_abstract(abstract)

        return Citation_Information(filename, propno, category, cycle, authors, title,
                                    submission_year, timing_year, abstract)

    @property
    def author_list(self):
        """Value of author_list attribute for use in XML.
        """

        return ', '.join(self.authors)

    @property
    def editor_list(self):
        """Value of editor_list attribute for use in XML.
        """

        editor = 'RMS Node MAST-to-PDS4 Pipeline'
        if self.PIPELINE_VERSION:
            editor = editor + ' v. ' + str(self.PIPELINE_VERSION)

        return editor

    @property
    def publication_year(self):
        """Value of publication_year attribute for use in XML.
        """

        # Return the publication year if it has been specifically provided
        if self.pub_year:
            return str(self.pub_year)

        # Return a hard-wired value if available
        if self.propno in PUBLICATION_YEARS:
            return str(PUBLICATION_YEARS[self.propno])

        # Otherwise, use the end year of the cycle
        return str(YEARS_FOR_CYCLE[self.cycle][1])

    @property
    def keywords(self):
        """Keywords for use in XML; currently always an empty list.
        """

        return []

    @property
    def description(self):
        """The complete citation description, for use in XML.
        """

        return (self.author_list
                + ', "'
                + self.title
                + '", HST Cycle '
                + str(self.cycle)
                + ' Program '
                + str(self.propno)
                + ', '
                + self.publication_year
                + '.')

    def abstract_formatted(self, width=80, indent=0, escaped=False):
        """Abstract as a list of strings, formatted to insert into an XML file.
        """

        formatted = []
        for paragraph in self.abstract:
            if not paragraph:
                formatted.append('')
                continue

            recs = textwrap.wrap(paragraph, width - indent)
            paragraph = '\n'.join(recs)

            if indent:
                paragraph = textwrap.indent(paragraph, indent * ' ')

            paragraph = escape(paragraph)
            formatted += paragraph.splitlines()

        return formatted

    def set_publication_year(self, year):
        """Set the publication year that will appear in the citation."""

        self.pub_year = year

    def __str__(self):
        return self.description

    def __repr__(self):
        return 'Citation_Information(' + self.description + ')'

    @staticmethod
    def set_pipeline_version(version):
        """Set the version of the pipeline, to be used in the returned editor_list. Note
        that this is a class method.
        """

        Citation_Information.PIPELINE_VERSION = version

    @staticmethod
    def create_test_citation_information():
        """For testing."""
        info = Citation_Information('{filename}', 99999, '{category}', 1,
                                    ['{author_1}', '{author_2}'], '{title}',
                                    2001, 2001,
                                    ['{abstract_line_1}', '{abstract_line_2}'])
        info.set_publication_year(2021)
        return info

##########################################################################################
