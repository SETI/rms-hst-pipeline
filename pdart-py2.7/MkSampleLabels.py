from fs.path import basename, splitext

from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_db.FitsFileDB import populate_database_from_fits_file
from pdart.new_labels.FitsProductLabel import make_fits_product_label
from pdart.pds4.LIDVID import LIDVID


def runGeneric(fits_product_lidvid_str, fits_filepath):
    # type: (str, str) -> unicode
    """
    Given a LIDVID and a FITS file, build a sample label for an
    instrument.
    """
    lidvid = LIDVID(fits_product_lidvid_str)

    def parent_lidvid(lidvid):
        # just propagates the VID
        lid = lidvid.lid().parent_lid()
        vid = lidvid.vid()
        return LIDVID.create_from_lid_and_vid(lid, vid)

    coll_lidvid = parent_lidvid(lidvid)
    bundle_lidvid = parent_lidvid(coll_lidvid)

    bundle_db = create_bundle_db_in_memory()
    bundle_db.create_tables()
    bundle_db.create_bundle(str(bundle_lidvid))
    bundle_db.create_non_document_collection(str(coll_lidvid),
                                             str(bundle_lidvid))
    bundle_db.create_fits_product(fits_product_lidvid_str,
                                  str(coll_lidvid))
    populate_database_from_fits_file(bundle_db,
                                     fits_filepath,
                                     fits_product_lidvid_str)
    file_basename = basename(fits_filepath)

    return make_fits_product_label(bundle_db, fits_product_lidvid_str,
                                   file_basename, True)


def runTracedGeneric(instrument, fits_product_lidvid_str, fits_filepath):
    # type: (str, str, str) -> None
    """
    Build a sample label for an instrument, write it to the
    filesystem, and print a message.
    """
    label = runGeneric(fits_product_lidvid_str, fits_filepath)
    base = splitext(basename(fits_filepath))[0]
    label_filepath = '/Users/spaceman/' + base + '.xml'
    with open(label_filepath, 'w') as f:
        f.write(str(label))
    print 'ran for', LIDVID(fits_product_lidvid_str), 'on', instrument


def runForACS():
    # type: () -> None
    """Build a sample label for ACS"""
    runTracedGeneric(
        'acs',
        'urn:nasa:pds:hst_10534:data_wfpc2_d0m:u9c10101m_d0m::1.0',
        '/Users/spaceman/Desktop/Archive/hst_10534/data_wfpc2_d0m/visit_01'
        '/u9c10101m_d0m.fits')


def runForWFC3():
    # type: () -> None
    """Build a sample label for WFC3"""
    runTracedGeneric(
        'wfc3',
        'urn:nasa:pds:hst_11536:data_wfc3_raw:ib6m02d9q_raw::1.0',
        '/Users/spaceman/Desktop/Archive/hst_11536'
        '/data_wfc3_raw/visit_02/ib6m02d9q_raw.fits')


def runForWFPC2():
    # type: () -> None
    """Build a sample label for WFPC2"""
    runTracedGeneric(
        'wfpc2',
        'urn:nasa:pds:hst_09059:data_acs_raw:j6gp01lzq_raw::1.0',
        '/Users/spaceman/Desktop/Archive/hst_09059/data_acs_raw'
        '/visit_01/j6gp01lzq_raw.fits')


def run():
    # type: () -> None
    runForACS()
    runForWFC3()
    runForWFPC2()


if __name__ == '__main__':
    run()
