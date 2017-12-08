import unittest

from fs.path import basename, join

from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_db.FitsFileDB import card_dictionaries, populate_from_fits_file
from pdart.new_db.SqlAlchTables import File, FitsFile
from pdart.new_labels.HstParameters import *
from pdart.xml.Pretty import pretty_print


class Test_HstParameters(unittest.TestCase):
    # TODO Write cases for other two instruments

    # TODO get_gain_mode_id() is failing on this input file.  Investigate why.

    def test_get_acs_parameters(self):
        # type: () -> None
        db = create_bundle_db_in_memory()
        db.create_tables()
        archive = '/Users/spaceman/Desktop/Archive'

        fits_product_lidvid = \
            'urn:nasa:pds:hst_13012:data_acs_raw:jbz504eoq_raw::2'
        os_filepath = join(
            archive,
            'hst_13012/data_acs_raw/visit_04/jbz504eoq_raw.fits')

        populate_from_fits_file(db,
                                os_filepath,
                                fits_product_lidvid)

        file_basename = basename(os_filepath)

        fits_file = db.session.query(FitsFile).filter(
            File.product_lidvid == fits_product_lidvid).filter(
            File.basename == file_basename).one()

        hdu_count = fits_file.hdu_count

        card_dicts = card_dictionaries(db.session,
                                       fits_product_lidvid,
                                       hdu_count)

        nb = get_hst_parameters(card_dicts, u'acs', fits_product_lidvid)
        doc = xml.dom.getDOMImplementation().createDocument(None, None, None)
        str = nb(doc).toxml()
        str = pretty_print(str)
        print str
        self.assertEqual(_expected, str)


_expected = '\n'.join([
    '<?xml version="1.0"?>',
    '<hst:HST>',
    '  <hst:Parameters_General>',
    '    <hst:stsci_group_id>### placeholder for stsci_group_id '
    '###</hst:stsci_group_id>',
    '    <hst:hst_proposal_id>13012</hst:hst_proposal_id>',
    '    <hst:hst_pi_name>Lamy, Laurent </hst:hst_pi_name>',
    '    <hst:hst_target_name>URANUS</hst:hst_target_name>',
    '    <hst:aperture_type>SBC</hst:aperture_type>',
    '    <hst:exposure_duration>270.0</hst:exposure_duration>',
    '    <hst:exposure_type>NORMAL</hst:exposure_type>',
    '    <hst:filter_name>F165LP+N/A</hst:filter_name>',
    '    <hst:fine_guidance_system_lock_type>FINE</hst'
    ':fine_guidance_system_lock_type>',
    '    <hst:gyroscope_mode>### placeholder for gyroscope_mode '
    '###</hst:gyroscope_mode>',
    '    <hst:instrument_mode_id>ACCUM</hst:instrument_mode_id>',
    '    <hst:moving_target_flag>true</hst:moving_target_flag>',
    '  </hst:Parameters_General>',
    '  <hst:Parameters_ACS>',
    '    <hst:detector_id>SBC</hst:detector_id>',
    '    <hst:gain_mode_id>### placeholder for gain_mode_id '
    '###</hst:gain_mode_id>',
    '    <hst:observation_type>IMAGING</hst:observation_type>',
    '    <hst:repeat_exposure_count>0</hst:repeat_exposure_count>',
    '    <hst:subarray_flag>### placeholder for subarray_flag ###'
    '</hst:subarray_flag>',
    '  </hst:Parameters_ACS>',
    '</hst:HST>',
    ''])  # type: str
# I'm jumping through hoops here just to get PEP8 compliance.
