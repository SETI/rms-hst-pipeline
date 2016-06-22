import os.path
import sys

from pdart.exceptions.Combinators import *
from pdart.pds4.Archive import *
from pdart.pds4.LID import *
from pdart.pds4.Product import *
from pdart.pds4labels.ProductLabel import *


def make_product_from_filepath(product_filepath):
    (visit_dir, fits_filepath) = os.path.split(product_filepath)
    (product_segment, product_ext) = os.path.splitext(fits_filepath)
    (collection_dir, visit_segment) = os.path.split(visit_dir)
    (bundle_dir, collection_segment) = os.path.split(collection_dir)
    (archive_dir, bundle_segment) = os.path.split(bundle_dir)
    archive = Archive(archive_dir)
    lid_str = 'urn:nasa:pds:%s:%s:%s' % \
        (bundle_segment, collection_segment, product_segment)
    sys.stdout.flush()
    lid = LID(lid_str)
    return Product(archive, lid)


def usage():
    sys.stderr.write('usage: python MakeLabel.py <product FITS file>\n')
    sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) is not 2:
        usage()

    product_filepath = sys.argv[1]
    if os.path.splitext(product_filepath)[1] != '.fits':
        usage()

    product = make_product_from_filepath(product_filepath)

    def run():
        label = make_product_label(product, False)
        failures = xml_schema_failures(None, label)
        if failures is not None:
            print label
            raise Exception('XML schema validation errors: ' + failures)
        failures = schematron_failures(None, label)
        if failures is not None:
            print label
            raise Exception('Schematron validation errors: ' + failures)

    raise_verbosely(run)
