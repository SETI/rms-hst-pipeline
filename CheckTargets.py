from pdart.exceptions.Combinators import *
from pdart.pds4.Archives import *
from pdart.pds4labels.TargetIdentificationLabelReduction import *
from pdart.reductions.Reduction import *


class CheckTargetsReduction(Reduction):
    """
    When run on an archive, print a the product LIDs with their
    calculated targets.
    """
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        get_reduced_collections()

    def reduce_collection(self, archive, lid, get_reduced_products):
        get_reduced_products()

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        targ_info = get_reduced_fits_files()[0]
        if targ_info['type'] != 'NO_TARGNAME':
            print (lid.lid, targ_info)

    def reduce_fits_file(self, file, get_reduced_hdus):
        try:
            reduced_hdus = get_reduced_hdus()
            targname = reduced_hdus[0]
            if targname is None:
                return {'type': 'NO_TARGNAME'}
            target = targname_to_target(targname)
            if target is None:
                return {'type': 'UNRESOLVED', 'targname': targname}
            else:
                return {'type': 'RESOLVED',
                        'targname': targname,
                        'target': target[0]}
        except IOError:
            # couldn't read the file
            return {'type': 'READ_ERROR'}

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        if n == 0:
            return get_reduced_header_unit()
        else:
            pass

    def reduce_header_unit(self, n, header_unit):
        if n == 0:
            try:
                targname = header_unit['TARGNAME']
                return targname
            except KeyError:
                return None


if __name__ == '__main__':
    reduction = CheckTargetsReduction()
    archive = get_any_archive()
    raise_verbosely(lambda: run_reduction(reduction, archive))
