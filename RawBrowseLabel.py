"""
SCRIPT: Grab the first raw data product, create a product label for
its corresponding browse product, and print the label.  Run the XML
Schema and Schematron tests on it; if it fails, print the failures.
Otherwise, print a cheery message.
"""
from pdart.pds4.Archives import *
from pdart.pds4labels.BrowseProductLabel import *
from pdart.reductions.Reduction import *
from pdart.xml.Schema import *


def get_product(archive):
    for bundle in archive.bundles():
        for collection in bundle.collections():
            if collection.prefix() == 'data' and collection.suffix() == 'raw':
                for product in collection.products():
                    return product
    return None

if __name__ == '__main__':
    archive = get_any_archive()
    product = get_product(archive)
    reduction = BrowseProductLabelReduction()
    runner = DefaultReductionRunner()
    label = runner.run_product(reduction, product)
    print label
    failures = xml_schema_failures(None, label) and \
        schematron_failures(None, label)
    if failures is None:
        print 'PERFECT!!!'
    else:
        print failures
