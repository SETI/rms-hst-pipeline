import tempfile

from fs.path import basename, splitext

from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_db.FitsFileDB import populate_database_from_fits_file
from pdart.new_labels.FitsProductLabel import make_fits_product_label
from pdart.pds4.LIDVID import LIDVID

def runForGeneric(instrument, fits_product_lidvid_str, filepath):
    lidvid = LIDVID(fits_product_lidvid_str)

    if False:
        from subprocess import call
        call(['cp', filepath, '/Users/spaceman/'])
        print 'ran for', lidvid, 'on', instrument
        return
        
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
                                     filepath,
                                     fits_product_lidvid_str)
    file_basename = basename(filepath)

    label = make_fits_product_label(bundle_db, fits_product_lidvid_str,
                                  file_basename, True)

    base = splitext(file_basename)[0]
    label_filepath = '/Users/spaceman/' + base + '.xml'
    with open(label_filepath, 'w') as f:
        f.write(label)
    print 'ran for', lidvid, 'on', instrument

def runForACS():
    runForGeneric('acs',
                  'urn:nasa:pds:hst_10534:data_wfpc2_d0m:u9c10101m_d0m::1.0', 
                  '/Users/spaceman/Desktop/Archive/hst_10534/data_wfpc2_d0m/visit_01/u9c10101m_d0m.fits')

def runForWFC3():
    runForGeneric('wfc3',
                  'urn:nasa:pds:hst_11536:data_wfc3_raw:ib6m02d9q_raw::1.0',
                  '/Users/spaceman/Desktop/Archive/hst_11536/data_wfc3_raw/visit_02/ib6m02d9q_raw.fits')

def runForWFPC2():
    runForGeneric('wfpc2',
                  'urn:nasa:pds:hst_09059:data_acs_raw:j6gp01lzq_raw::1.0',
                  '/Users/spaceman/Desktop/Archive/hst_09059/data_acs_raw/visit_01/j6gp01lzq_raw.fits')
    

def run():
    runForACS()
    runForWFC3()
    runForWFPC2()


if __name__ == '__main__':
    run()
