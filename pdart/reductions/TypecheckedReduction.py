from pdart.reductions.Reduction import *


class Typechecks(object):
    def check_is_archive_reduction(self, obj):
        pass

    def check_is_bundle_reduction(self, obj):
        pass

    def check_is_collection_reduction(self, obj):
        pass

    def check_is_product_reduction(self, obj):
        pass

    def check_is_fits_file_reduction(self, obj):
        pass

    def check_is_hdu_reduction(self, obj):
        pass

    def check_is_header_unit_reduction(self, obj):
        pass

    def check_is_data_unit_reduction(self, obj):
        pass


def check_thunk(check, thunk):
    def checked_thunk():
        res = thunk()
        check(res)
        return res
    return checked_thunk


def check_list_thunk(check, thunk):
    def checked_thunk():
        res = thunk()
        assert isinstance(res, list)
        for r in res:
            check(r)
        return res
    return checked_thunk


class TypecheckedReduction(Reduction):
    def __init__(self, typechecks, base_reduction):
        assert typechecks
        self.typechecks = typechecks
        assert base_reduction
        self.base_reduction = base_reduction

    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles_tc = \
            check_list_thunk(self.typechecks.check_is_bundle_reduction,
                             get_reduced_bundles)
        res = self.base_reduction.reduce_archive(archive_root,
                                                 get_reduced_bundles_tc)
        self.typechecks.check_is_archive_reduction(res)
        return res

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        get_reduced_collections_tc = \
            check_list_thunk(self.typechecks.check_is_collection_reduction,
                             get_reduced_collections)
        res = self.base_reduction.reduce_bundle(archive,
                                                lid,
                                                get_reduced_collections_tc)
        self.typechecks.check_is_bundle_reduction(res)
        return res

    def reduce_collection(self, archive, lid, get_reduced_products):
        get_reduced_products_tc = \
            check_list_thunk(self.typechecks.check_is_product_reduction,
                             get_reduced_products)
        res = self.base_reduction.reduce_collection(archive,
                                                    lid,
                                                    get_reduced_products_tc)
        self.typechecks.check_is_collection_reduction(res)
        return res

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        get_reduced_fits_files_tc = \
            check_list_thunk(self.typechecks.check_is_fits_file_reduction,
                             get_reduced_fits_files)
        res = self.base_reduction.reduce_product(archive,
                                                 lid,
                                                 get_reduced_fits_files_tc)
        self.typechecks.check_is_product_reduction(res)
        return res

    def reduce_fits_file(self, file, get_reduced_hdus):
        get_reduced_hdus_tc = \
            check_list_thunk(self.typechecks.check_is_hdu_reduction,
                             get_reduced_hdus)
        res = self.base_reduction.reduce_fits_file(file,
                                                   get_reduced_hdus_tc)
        self.typechecks.check_is_fits_file_reduction(res)
        return res

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        get_reduced_header_unit_tc = \
            check_thunk(self.typechecks.check_is_header_unit_reduction,
                        get_reduced_header_unit)
        get_reduced_data_unit_tc = \
            check_thunk(self.typechecks.check_is_data_unit_reduction,
                        get_reduced_data_unit)
        res = self.base_reduction.reduce_hdu(n,
                                             hdu,
                                             get_reduced_header_unit_tc,
                                             get_reduced_data_unit_tc)
        self.typechecks.check_is_hdu_reduction(res)
        return res

    def reduce_header_unit(self, n, header_unit):
        res = self.base_reduction.reduce_header_unit(n,
                                                     header_unit)
        self.typechecks.check_is_header_unit_reduction(res)
        return res

    def reduce_data_unit(self, n, data_unit):
        res = self.base_reduction.reduce_data_unit(n,
                                                   data_unit)
        self.typechecks.check_is_data_unit_reduction(res)
        return res


class TypecheckedReductionRunner(object):
    def __init__(self, typechecks, base_runner):
        assert typechecks
        self.typechecks = typechecks
        assert base_runner
        self.base_runner = base_runner

    def run_archive(self, reduction, archive):
        res = self.base_runner.run_archive(reduction, archive)
        self.typechecks.check_is_archive_reduction(res)
        return res

    def run_bundle(self, reduction, bundle):
        res = self.base_runner.run_bundle(reduction, bundle)
        self.typechecks.check_is_bundle_reduction(res)
        return res

    def run_collection(self, reduction, collection):
        res = self.base_runner.run_collection(reduction, collection)
        self.typechecks.check_is_collection_reduction(res)
        return res

    def run_product(self, reduction, product):
        res = self.base_runner.run_product(reduction, product)
        self.typechecks.check_is_product_reduction(res)
        return res

    def run_fits_file(self, reduction, file):
        res = self.base_runner.run_fits_file(reduction, file)
        self.typechecks.check_is_fits_file_reduction(res)
        return res

    def run_hdu(self, reduction, n, hdu):
        res = self.base_runner.run_hdu(reduction, n, hdu)
        self.typechecks.check_is_hdu_reduction(res)
        return res

    def run_header_unit(self, reduction, n, header_unit):
        res = self.base_runner.run_header_unit(reduction,
                                               n,
                                               header_unit)
        self.typechecks.check_is_header_unit_reduction(res)
        return res

    def run_data_unit(self, reduction, n, data_unit):
        res = self.base_runner.run_data_unit(reduction, n, data_unit)
        self.typechecks.check_is_data_unit_reduction(res)
        return res
