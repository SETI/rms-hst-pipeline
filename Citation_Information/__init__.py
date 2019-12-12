################################################################################
# Citation_Information class for MAST -> PDS4 pipeline
#
# Mark Showalter, December 12, 2019
################################################################################

from Citation_Information_from_apt import Citation_Information_from_apt
from Citation_Information_from_pro import Citation_Information_from_pro

class Citation_Information:
    """This class encapsulates the information needed for the PDS4
    Citation_Information object, as extracted from either an APT file or a
    .PRO file.

    The constructor takes two inputs:
        - name of the file (which must end in .apt or .pro)
        - version ID of the MAST -> PDS4 pipeline, which is included in the
          returned information

    The returned Citation_Information object has the following attributes, all
    strings, which can be accessed directly:
        author_list
        editor_list
        publication_year
        keyword (currently always a blank string)
        description (the full bibliographic citation)

    Note that the publication_year should actually be the year of the last
    observation in the program. Unfortunately, this cannot be reliably
    determined from the APT or PRO files. The routine provides a year that will
    usually be correct. However, if the correct year can be determined in some
    other way, method replace_year can be used to update the year in the
    object.
    """

    def __init__(self, filename, pipeline_version=None):

        # Save the filename because why not
        self.filename = filename

        # Read the file and extract the key info
        filename_lc = filename.lower()
        if filename_lc.endswith('.apt'):
            info = Citation_Information_from_apt(filename)
        elif filename_lc.endswith('.pro'):
            info = Citation_Information_from_pro(filename)
        else:
            raise ValueError('unrecognized file format: ' + filename)

        (self.propno,
         self.category,
         self.cycle,
         self.authors,
         self.title,
         self.year) = info

        # Construct needed fields for Citation_Information object
        self.author_list = ', '.join(self.authors)

        editor = 'RMS Node MAST-to-PDS4 Pipeline'
        if pipeline_version:
            editor = editor + ' v. ' + str(pipeline_version)
        self.editor_list = [editor]

        self.publication_year = str(self.year)

        self.keyword = ''

        self.description = (
            self.author_list + 
            ', "' + self.title +
            '", HST Cycle ' + str(self.cycle) +
            ' Program ' + self.category + ' ' + str(self.propno) +
            ', ' + str(self.year) + '.')

    def replace_year(self, year):
        """Replaces the year that appears in the citation."""

        self.year = int(year)
        self.publication_year = str(year)
        self.description = self.description[:-5] + self.publication_year + '.'

    def __str__(self):
        return self.description

    def __repr__(self):
        return 'Citation_Information(' + self.description + ')'

################################################################################

