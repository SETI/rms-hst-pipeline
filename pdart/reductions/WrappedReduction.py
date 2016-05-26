from pdart.reductions.Reduction import *


class WrappedReduction(Reduction):
    """
    Wraps a reduction in another reduction.  Useful for selectively
    overriding methods.
    """
    def __init__(self, base_reduction):
        assert base_reduction
        assert isinstance(base_reduction, Reduction)
        self.base_reduction = base_reduction

    def reduce_archive(self, archive_root, get_reduced_bundles):
        return self.base_reduction.reduce_archive(archive_root,
                                                  get_reduced_bundles)

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        return self.base_reduction.reduce_bundle(archive,
                                                 lid,
                                                 get_reduced_collections)

    def reduce_collection(self, archive, lid, get_reduced_products):
        return self.base_reduction.reduce_collection(archive,
                                                     lid,
                                                     get_reduced_products)

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        return self.base_reduction.reduce_product(archive,
                                                  lid,
                                                  get_reduced_fits_files)

    def reduce_fits_file(self, file, get_reduced_hdus):
        return self.base_reduction.reduce_fits_file(file, get_reduced_hdus)

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        return self.base_reduction.reduce_hdu(self, n, hdu,
                                              get_reduced_header_unit,
                                              get_reduced_data_unit)

    def reduce_header_unit(n, header_unit):
        return self.base_reduction.reduce_header_unit(n, header_unit)

    def reduce_data_unit(n, data_unit):
        return self.base_reduction.reduce_data_unit(n, data_unit)
