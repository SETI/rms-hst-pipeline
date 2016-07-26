from contextlib import closing
import os.path
import sys

from pdart.pds4.Product import *
from pdart.pds4labels.FileContentsLabelReduction import *
from pdart.pds4labels.HstParametersReduction import *
from pdart.pds4labels.ObservingSystem import *
from pdart.pds4labels.TargetIdentificationLabelReduction import *
from pdart.pds4labels.TimeCoordinatesLabelReduction import *
from pdart.reductions.BadFitsFileReduction import *
from pdart.reductions.CompositeReduction import *
from pdart.reductions.Reduction import *
from pdart.xml.Schema import *
from pdart.xml.Templates import *

make_label = interpret_document_template(
    """<?xml version="1.0" encoding="utf-8"?>
<?xml-model href="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1500.sch"
            schematypens="http://purl.oclc.org/dsdl/schematron"?>
<Product_Observational
   xmlns="http://pds.nasa.gov/pds4/pds/v1"
   xmlns:hst="http://pds.nasa.gov/pds4/hst/v0"
   xmlns:pds="http://pds.nasa.gov/pds4/pds/v1"
   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
   xsi:schemaLocation="http://pds.nasa.gov/pds4/pds/v1
                       http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1500.xsd">
  <Identification_Area>
    <logical_identifier><NODE name="lid" /></logical_identifier>
    <version_id>0.1</version_id>
    <title>This collection contains the <NODE name="suffix" /> \
image obtained the HST Observing Program <NODE name="proposal_id" />\
.</title>
    <information_model_version>1.6.0.0</information_model_version>
    <product_class>Product_Observational</product_class>
    <Modification_History>
      <Modification_Detail>
        <modification_date>2016-04-20</modification_date>
        <version_id>0.1</version_id>
        <description>PDS4 version-in-development of the product</description>
      </Modification_Detail>
    </Modification_History>
  </Identification_Area>
  <Observation_Area>
    <NODE name="Time_Coordinates" />
    <Investigation_Area>
      <name><NODE name="Investigation_Area_name" /></name>
      <type>Individual Investigation</type>
      <Internal_Reference>
        <lidvid_reference><NODE name="investigation_lidvid" />\
</lidvid_reference>
        <reference_type>data_to_investigation</reference_type>
      </Internal_Reference>
    </Investigation_Area>
    <NODE name="Observing_System" />
    <NODE name="Target_Identification" />
    <Mission_Area><NODE name="HST" /></Mission_Area>
  </Observation_Area>
  <File_Area_Observational>
    <File>
      <file_name><NODE name="file_name" /></file_name>
    </File>
    <FRAGMENT name="file_contents" />
  </File_Area_Observational>
</Product_Observational>""")


def mk_Investigation_Area_name(proposal_id):
    return 'HST observing program %d' % proposal_id


def mk_Investigation_Area_lidvid(proposal_id):
    return 'urn:nasa:pds:context:investigation:investigation.hst_%05d::1.0' % \
        proposal_id


class ProductLabelReduction(BadFitsFileReduction):
    """
    Reduce a product to the label of its first (presumably only) FITS
    file, write the label into the archive and return the label.  If
    the FITS file is bad, return None.
    """
    def __init__(self, verify=False):
        base_reduction = CompositeReduction(
            [FileContentsLabelReduction(),
             TargetIdentificationLabelReduction(),
             HstParametersReduction(),
             TimeCoordinatesLabelReduction()])
        BadFitsFileReduction.__init__(self, base_reduction)
        self.verify = verify

    def _nones(self):
        """
        Return a tuple of Nones of the same length as the list of
        reductions composed (used as a failure value when the FITS
        file can't be read).
        """
        return len(self.base_reduction.reductions) * (None,)

    def bad_fits_file_reduction(self, file):
        return self._nones()

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        # string
        product = Product(archive, lid)
        file_name = os.path.basename(product.absolute_filepath())

        instrument = product.collection().instrument()
        suffix = product.collection().suffix()

        proposal_id = product.bundle().proposal_id()
        investigation_area_name = mk_Investigation_Area_name(proposal_id)
        investigation_area_lidvid = mk_Investigation_Area_lidvid(proposal_id)

        reduced_fits_file = get_reduced_fits_files()[0]

        # on a bad FITS file, return None
        if reduced_fits_file == self._nones():
            return None

        (file_contents,
         target_identification,
         hst,
         time_coordinates) = reduced_fits_file

        label = make_label({
                'lid': str(lid),
                'suffix': suffix.upper(),
                'proposal_id': str(proposal_id),
                'Investigation_Area_name': investigation_area_name,
                'investigation_lidvid': investigation_area_lidvid,
                'Observing_System': observing_system(instrument),
                'file_name': file_name,
                'Time_Coordinates': time_coordinates,
                'Target_Identification': target_identification,
                'HST': hst((lid.product_id, instrument)),
                'file_contents': file_contents
                }).toxml()

        label_fp = Product(archive, lid).label_filepath()
        with open(label_fp, 'w') as f:
            f.write(label)

        if self.verify:
            verify_label_or_throw(label)

        return label


def make_product_label(product, verify):
    """
    Create the label text for this :class:`Product`.  If the FITS file
    is bad, raise an exception.  If verify is True, verify the label
    against its XML and Schematron schemas.  Raise an exception if
    either fails.
    """
    label = DefaultReductionRunner().run_product(ProductLabelReduction(verify),
                                                 product)
    if label is None:
        raise Exception('Bad FITS file')

    return label


def _make_header_dictionary(lid, hdu_index, cursor):
    cursor.execute("""SELECT keyword, value FROM cards
                      WHERE product=? AND hdu_index=?""",
                   (lid, hdu_index))
    return dict(cursor.fetchall())


def _make_header_dictionaries(lid, hdu_count, cursor):
    return [_make_header_dictionary(lid, i, cursor) for i in range(hdu_count)]


def make_db_product_label(conn, lid, verify):
    """
    Create the label text for the product having this :class:'LID'
    using the database connection.  If verify is True, verify the
    label against its XML and Schematron schemas.  Raise an exception
    if either fails.
    """
    with closing(conn.cursor()) as cursor:
        cursor.execute(
            """SELECT filename, label_filepath, collection,
                      product_id, hdu_count
               FROM products WHERE product=?""",
            (lid,))
        (file_name, label_fp, collection,
         product_id, hdu_count) = cursor.fetchone()

        cursor.execute("""SELECT bundle, instrument, suffix
                          FROM collections WHERE collection=?""",
                       (collection,))
        (bundle, instrument, suffix) = cursor.fetchone()

        cursor.execute('SELECT proposal_id FROM bundles WHERE bundle=?',
                       (bundle,))
        (proposal_id,) = cursor.fetchone()

        headers = _make_header_dictionaries(lid, hdu_count, cursor)

    label = make_label({
            'lid': str(lid),
            'proposal_id': str(proposal_id),
            'suffix': suffix,
            'file_name': file_name,
            'file_contents': get_db_file_contents(headers, conn, lid),
            'Investigation_Area_name': mk_Investigation_Area_name(proposal_id),
            'investigation_lidvid': mk_Investigation_Area_lidvid(proposal_id),
            'Observing_System': observing_system(instrument),
            'Time_Coordinates': get_db_time_coordinates(headers, conn, lid),
            'Target_Identification': get_db_target(headers, conn, lid),
            'HST': get_db_hst_parameters(headers, conn, lid,
                                         instrument, product_id)
            }).toxml()
    with open(label_fp, 'w') as f:
        f.write(label)

    print 'product label for', lid
    sys.stdout.flush()

    if verify:
        verify_label_or_throw(label)

    return label
