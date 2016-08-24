"""
Functionality to build a bundle label using a
:class:`~pdart.reductions.Reduction.Reduction`.
"""
import sys

from pdart.pds4.Bundle import *
from pdart.pds4labels.BundleLabelDB import *
from pdart.pds4labels.BundleLabelXml import *
from pdart.reductions.Reduction import *


class BundleLabelReduction(Reduction):
    def __init__(self, verify=False):
        Reduction.__init__(self)
        self.verify = verify

    """
    Reduction of a :class:`pdart.pds4.Bundle` to its PDS4 label as a
    string.
    """
    def reduce_bundle(self, archive, lid, get_reduced_collections):
        reduced_collections = get_reduced_collections()
        dict = {'lid': interpret_text(str(lid)),
                'proposal_id': interpret_text(str(Bundle(archive,
                                                         lid).proposal_id())),
                'Citation_Information': placeholder_citation_information,
                'Bundle_Member_Entries':
                    combine_nodes_into_fragment(reduced_collections)
                }
        label = make_label(dict).toxml()
        label_fp = Bundle(archive, lid).label_filepath()
        with open(label_fp, 'w') as f:
            f.write(label)

        if self.verify:
            verify_label_or_throw(label)

        return label

    def reduce_collection(self, archive, lid, get_reduced_products):
        dict = {'lid': interpret_text(str(lid))}
        return make_bundle_entry_member(dict)


def make_bundle_label(bundle, verify):
    """
    Create the label text for this :class:`Bundle`.  If verify is
    True, verify the label against its XML and Schematron schemas.
    Raise an exception if either fails.
    """
    return DefaultReductionRunner().run_bundle(BundleLabelReduction(verify),
                                               bundle)
