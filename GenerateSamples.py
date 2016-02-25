import datetime
import os
import os.path

import BundleLabelMaker
import CollectionLabelMaker
import FileArchives
import LabelMaker
import LID
import Product
import ProductLabelMaker


def make_samples(dst_dir, product_lid):
    """
    Create bundle, collection, and product labels given the product's
    LID.  Use the default archive.  Create the destination directory
    if it doesn't exist.
    """
    if os.path.isdir(dst_dir):
        pass
    elif os.path.isfile(dst_dir):
        print ('The destination "directory" %s already exists but is a file.' %
               dst_dir)
        sys.exit(1)
    else:
        os.makedirs(dst_dir)

    assert os.path.isdir(dst_dir)

    def label_checks(filepath):
        return LabelMaker.xml_schema_check(filepath) and \
            LabelMaker.schematron_check(filepath)

    archive = FileArchives.get_any_archive()
    product = Product.Product(archive, product_lid)
    product_lm = ProductLabelMaker.ProductLabelMaker(product)
    product_filepath = os.path.join(dst_dir, 'product.xml')
    product_lm.create_default_xml_file(product_filepath)
    assert label_checks(product_filepath)

    collection = product.collection()
    collection_lm = CollectionLabelMaker.CollectionLabelMaker(collection)
    collection_filepath = os.path.join(dst_dir, 'collection.xml')
    collection_lm.create_default_xml_file(collection_filepath)
    assert label_checks(collection_filepath)

    bundle = collection.bundle()
    bundle_lm = BundleLabelMaker.BundleLabelMaker(bundle)
    bundle_filepath = os.path.join(dst_dir, 'bundle.xml')
    bundle_lm.create_default_xml_file(bundle_filepath)
    assert label_checks(bundle_filepath)


if __name__ == '__main__':
    now = datetime.datetime.now()
    today = now.strftime('%Y-%m-%d')
    dst_dir = '/Users/spaceman/SampleGeneratedLabels-%s' % today
    make_samples(dst_dir,
                 LID.LID('urn:nasa:pds:hst_09059:data_acs_raw:visit_01'))