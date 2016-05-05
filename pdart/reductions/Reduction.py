import abc
import pyfits

from pdart.exceptions.Combinators import parallel_list


class Reduction(object):
    """
    A collection of methods to reduce PDS4 and FITS structure into a
    new form.
    """
    def reduce_archive(self, archive_root, get_reduced_bundles):
        pass

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        pass

    def reduce_collection(self, archive, lid, get_reduced_products):
        pass

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        pass

    def reduce_fits_file(self, file, get_reduced_hdus):
        pass

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        pass

    def reduce_header_unit(self, n, header_unit):
        pass

    def reduce_data_unit(self, n, data_unit):
        pass


def reduction_type_documentation(dict):
    """
    Return a string showing the types of the methods of Reduction.
    The dictionary argument gives the types of the reductions.
    """
    format_str = """archive reduces to {archive}
bundle reduces to {bundle}
collection reduces to {collection}
product reduces to {product}
fits_file reduces to {fits_file}
hdu reduces to {hdu}
header_unit reduces to {header_unit}
data_unit reduces to {data_unit}

reduce_archive(
    archive_root: str,
    get_reduced_bundles: () -> [{bundle}])
    ): {archive}

reduce_bundle(
    archive: Archive,
    lid: LID,
    get_reduced_collections: () -> [{collection}])
    ): {bundle}

reduce_collection(
    archive: Archive,
    lid: LID,
    get_reduced_products: () -> [{product}])
    ): {collection}

reduce_product(
    archive: Archive,
    lid: LID,
    get_reduced_fits_files: () -> [{fits_file}])
    ): {product}

reduce_fits_file(
    file: string,
    get_reduced_hdus: () -> [{hdu}])
    ): {fits_file}

reduce_hdu(
    n: int,
    hdu: hdu,
    get_reduced_header_unit: () -> {header_unit},
    get_reduced_data_unit: () -> {data_unit})
    : {hdu}

reduce_header_unit(
    n: int,
    header_unit: header_unit)
    ): {header_unit}

reduce_data_unit(
    n: int,
    data_unit: data_unit)
    ): {data_unit}"""
    return format_str.format(**dict)


class ReductionRunner(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def run_archive(self, reduction, archive):
        pass

    @abc.abstractmethod
    def run_bundle(self, reduction, bundle):
        pass

    @abc.abstractmethod
    def run_collection(self, reduction, collection):
        pass

    @abc.abstractmethod
    def run_product(self, reduction, product):
        pass

    @abc.abstractmethod
    def run_fits_file(self, reduction, file):
        pass

    @abc.abstractmethod
    def run_hdu(self, reduction, n, hdu):
        pass

    @abc.abstractmethod
    def run_header_unit(self, reduction, n, header_unit):
        pass

    @abc.abstractmethod
    def run_data_unit(self, reduction, n, data_unit):
        pass


class DefaultReductionRunner(object):
    """
    An algorithm to recursively reduce PDS4 and FITS structures
    according to a :class:`Reduction` instance.

    You don't have to understand how this works to use it.
    """
    def run_archive(self, reduction, archive):
        def get_reduced_bundles():
            bundles = list(archive.bundles())

            def make_thunk(bundle):
                def thunk():
                    return self.run_bundle(reduction, bundle)
                return thunk

            return parallel_list('run_archive',
                                 [make_thunk(bundle) for bundle in bundles])

        return reduction.reduce_archive(archive.root, get_reduced_bundles)

    def run_bundle(self, reduction, bundle):
        def get_reduced_collections():
            collections = list(bundle.collections())

            def make_thunk(collection):
                def thunk():
                    return self.run_collection(reduction, collection)
                return thunk

            return parallel_list('run_bundle', [make_thunk(collection)
                                                for collection in collections])

        return reduction.reduce_bundle(bundle.archive, bundle.lid,
                                       get_reduced_collections)

    def run_collection(self, reduction, collection):
        def get_reduced_products():
            products = list(collection.products())

            def make_thunk(product):
                def thunk():
                    return self.run_product(reduction, product)
                return thunk
            return parallel_list('run_collection',
                                 [make_thunk(product) for product in products])

        return reduction.reduce_collection(collection.archive,
                                           collection.lid,
                                           get_reduced_products)

    def run_product(self, reduction, product):
        def get_reduced_fits_files():
            files = list(product.files())

            def make_thunk(file):
                def thunk():
                    return self.run_fits_file(reduction, file)
                return thunk
            return parallel_list('run_product',
                                 [make_thunk(file) for file in files])

        return reduction.reduce_product(product.archive, product.lid,
                                        get_reduced_fits_files)

    def run_fits_file(self, reduction, file):
        def get_reduced_hdus():
            fits = pyfits.open(file.full_filepath())

            def build_thunk(n, hdu):
                def thunk():
                    return self.run_hdu(reduction, n, hdu)
                return thunk

            try:
                return parallel_list('run_fits_file',
                                     [build_thunk(n, hdu)
                                      for n, hdu in enumerate(fits)])
            finally:
                fits.close()

        return reduction.reduce_fits_file(file, get_reduced_hdus)

    def run_hdu(self, reduction, n, hdu):
        def get_reduced_header_unit():
            return self.run_header_unit(reduction, n, hdu.header)

        def get_reduced_data_unit():
            return self.run_data_unit(reduction, n, hdu.data)

        return reduction.reduce_hdu(n,
                                    hdu,
                                    get_reduced_header_unit,
                                    get_reduced_data_unit)

    def run_header_unit(self, reduction, n, header_unit):
        return reduction.reduce_header_unit(n, header_unit)

    def run_data_unit(self, reduction, n, data_unit):
        return reduction.reduce_data_unit(n, data_unit)


def run_reduction(reduction, archive):
    """
    Run a :class:`Reduction` on an :class:`Archive` using the default
    recursion.
    """
    return DefaultReductionRunner().run_archive(reduction, archive)
