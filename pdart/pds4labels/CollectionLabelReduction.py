import io
import sys

from pdart.pds4.Collection import *
from pdart.pds4labels.CollectionLabelXml import *
from pdart.reductions.Reduction import *
from pdart.xml.Schema import *


class CollectionLabelReduction(Reduction):
    def __init__(self, verify=False):
        Reduction.__init__(self)
        self.verify = verify

    """
    Reduction of a :class:`pdart.pds4.Collection` to its PDS4 label as
    a string.
    """
    def reduce_collection(self, archive, lid, get_reduced_products):
        collection = Collection(archive, lid)
        suffix = collection.suffix()
        proposal_id = collection.bundle().proposal_id()
        inventory_name = collection.inventory_name()

        dict = {'lid': interpret_text(str(lid)),
                'suffix': interpret_text(suffix.upper()),
                'proposal_id': interpret_text(str(proposal_id)),
                'Citation_Information': placeholder_citation_information,
                'inventory_name': interpret_text(inventory_name)
                }
        label = make_label(dict).toxml()
        label_fp = Collection(archive, lid).label_filepath()

        inventory_filepath = collection.inventory_filepath()
        with io.open(inventory_filepath, 'w', newline='') as f:
            f.write(make_collection_inventory(collection))

        with open(label_fp, 'w') as f:
            f.write(label)

        if self.verify:
            verify_label_or_throw(label)

        return label


def make_collection_label(collection, verify):
    """
    Create the label text for this :class:`pdart.pds4.Collection`.  If
    verify is True, verify the label against its XML and Schematron
    schemas.  Raise an exception if either fails.
    """
    return DefaultReductionRunner().run_collection(
        CollectionLabelReduction(verify), collection)


def make_collection_inventory(collection):
    lines = [u'P,%s\r\n' % str(product.lid)
             for product in collection.products()]
    return ''.join(lines)


def make_collection_label_and_inventory(collection):
    """
    Create the label and inventory for a collection and write to the disk.
    """
    inventory_filepath = collection.inventory_filepath()
    with io.open(inventory_filepath, 'w', newline='') as f:
        f.write(make_collection_inventory(collection))

    label_filepath = collection.label_filepath()
    with io.open(label_filepath, 'w') as f:
        f.write(make_collection_label(collection, True))
