"""
Functionality to build a product label using a
:class:`~pdart.reductions.Reduction.Reduction`.
"""
import os.path
import sys

from pdart.pds4.Product import *
from pdart.pds4labels.FileContentsReduction import *
from pdart.pds4labels.HstParametersReduction import *
from pdart.pds4labels.ObservingSystem import *
from pdart.pds4labels.ProductLabelXml import *
from pdart.pds4labels.TargetIdentificationReduction import *
from pdart.pds4labels.TimeCoordinatesReduction import *
from pdart.reductions.BadFitsFileReduction import *
from pdart.reductions.CompositeReduction import *
from pdart.reductions.Reduction import *
from pdart.xml.Pretty import *
from pdart.xml.Schema import *


class ProductLabelReduction(BadFitsFileReduction):
    """
    Reduce a product to the label of its first (presumably only) FITS
    file, write the label into the archive and return the label.  If
    the FITS file is bad, return None.
    """
    def __init__(self, verify=False):
        base_reduction = CompositeReduction(
            [FileContentsReduction(),
             TargetIdentificationReduction(),
             HstParametersReduction(),
             TimeCoordinatesReduction()])
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
        label = pretty_print(label)

        label_fp = Product(archive, lid).label_filepath()
        with open(label_fp, 'w') as f:
            f.write(label)

        if self.verify:
            verify_label_or_raise(label)

        return label


def make_product_label(product, verify):
    """
    Create the label text for this :class:`~pdart.pds4.Product`.  If
    the FITS file is bad, raise an exception.  If verify is True,
    verify the label against its XML and Schematron schemas.  Raise an
    exception if either fails.
    """
    label = DefaultReductionRunner().run_product(ProductLabelReduction(verify),
                                                 product)
    if label is None:
        raise Exception('Bad FITS file')

    return label
