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

    def reduce_header_unit(self, n, get_header_unit):
        pass

    def reduce_data_unit(self, n, get_data_unit):
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
    ): {product}

reduce_hdu(
    n: int,
    hdu: hdu,
    get_reduced_header_unit: () -> {header_unit},
    get_reduced_data_unit: () -> {data_unit})
    : {hdu}

reduce_header_unit(
    n: int,
    get_header_unit: () -> header_unit)
    ): {header_unit}

reduce_data_unit(
    n: int,
    get_data_unit: () -> data_unit)
    ): {data_unit}"""
    return format_str.format(**dict)


class ReductionRunner(object):
    """
    An algorithm to recursively reduce PDS4 and FITS structures
    according to a :class:`Reduction` instance.

    You don't have to understand how this works to use it.
    """
    def run_archive(self, reduction, archive):
        def get_reduced_bundles():
            bundles = list(archive.bundles())
            return parallel_list('run_archive',
                                 [lambda: self.run_bundle(reduction, bundle)
                                  for bundle in bundles])

        return reduction.reduce_archive(archive.root, get_reduced_bundles)

    def run_bundle(self, reduction, bundle):
        def get_reduced_collections():
            collections = list(bundle.collections())
            return parallel_list('run_bundle',
                                 [lambda: self.run_collection(reduction,
                                                              collection)
                                  for collection in collections])

        return reduction.reduce_bundle(bundle.archive, bundle.lid,
                                       get_reduced_collections)

    def run_collection(self, reduction, collection):
        def get_reduced_products():
            products = list(collection.products())
            return parallel_list('run_collection',
                                 [lambda: self.run_product(reduction, product)
                                  for product in products])

        return reduction.reduce_collection(collection.archive,
                                           collection.lid,
                                           get_reduced_products)

    def run_product(self, reduction, product):
        def get_reduced_fits_files():
            files = list(product.files())
            return parallel_list('run_product',
                                 [lambda: self.run_fits_file(reduction, file)
                                  for file in files])

        return reduction.reduce_product(product.archive, product.lid,
                                        get_reduced_fits_files)

    def run_fits_file(self, reduction, file):
        def get_reduced_hdus():
            fits = pyfits.open(file.full_filepath())

            # We have to jump through some hoops to make sure that in
            # building the parallel list we return, we capture the n
            # and hdu *values* not the *variables* that hold them.  In
            # the obvious (but wrong) implementation I first wrote,
            #
            # [lambda: self.run_hdu(reduction, n, hdu) for n, hdu in
            # enumerate(fits)]
            #
            # it captured the n and hdu *variables* and as a result,
            # all of the resulting list got the same n and hdu (the
            # last ones, since the lambdas didn't get run until after
            # the enumeration was done).  If we pass n and hdu as
            # arguments, their current *value* is used rather than just
            # capturing the variable.
            #
            # Ignore this if it makes your head hurt.
            def build_thunk(n, hdu):
                """
                Return a no-argument function (a thunk) that runs
                self.run_hdu() with the given n and hdu values.
                """
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
            return reduction.reduce_header_unit(n, lambda: hdu.header)

        def get_reduced_data_unit():
            return reduction.reduce_data_unit(n, lambda: hdu.data)

        return reduction.reduce_hdu(n, hdu,
                                    get_reduced_header_unit,
                                    get_reduced_data_unit)

    def run_header_unit(self, reduction, n, hu):
        return reduction.reduce_header_unit(n, get_header_unit)

    def run_data_unit(self, reduction, n, du):
        return reduction.reduce_data_unit(n, get_data_unit)


def run_reduction(reduction, archive):
    """
    Run a :class:`Reduction` on an :class:`Archive` using the default
    recursion.
    """
    return ReductionRunner().run_archive(reduction, archive)
