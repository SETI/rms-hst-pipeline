import abc
import itertools
import pyfits
import traceback

import FileArchives
import pdart.exceptions.ExceptionInfo
import pdart.exceptions.Result

class Runner(object):
    # Note the coding pattern in these methods: decompose, recurse,
    # and recompose.  Given a structured object, we first *decompose*
    # it by getting an iterator to its children.  Next we *recurse*
    # through the children, reducing each one.  Then we *recompose*
    # the (reduced) structure by building it from its reduced
    # children.

    # We pass generators around instead of lists so that you can fully
    # control the computation from the recomp object.

    # For instance, you might choose not to recurse below a certain
    # level.  Opening FITS files is particularly expensive in time, so
    # if you don't need that level of detail, you can just not use the
    # generate_hds() iterator.

    # Or you might choose to reduce objects in parallel, collecting
    # the combined history possibly containing multiple exceptions.

    def run_archive(self, reducer, archive):
        bundles = archive.bundles()
        bundles_ = itertools.imap(lambda(x): self.run_bundle(reducer, x),
                                  bundles)
        return reducer.reduce_archive_gen(archive.root, bundles_)

    def run_bundle(self, reducer, bundle):
        colls = bundle.collections()
        colls_ = itertools.imap(lambda(x): self.run_collection(reducer, x),
                                colls)
        return reducer.reduce_bundle_gen(bundle.archive, bundle.lid, colls_)

    def run_collection(self, reducer, collection):
        products = collection.products()
        products_ = itertools.imap(lambda(x): self.run_product(reducer, x),
                                   products)
        return reducer.reduce_collection_gen(collection.archive,
                                             collection.lid,
                                             products_)

    def run_product(self, reducer, product):
        fits_files = product.files()
        fits_files_ = itertools.imap(lambda(x): self.run_fits(reducer, x),
                                     fits_files)
        return reducer.reduce_product_gen(product.archive, product.lid,
                                          fits_files_)

    def run_fits(self, reducer, file):
        def generate_hdus():
            try:
                fits = pyfits.open(file.full_filepath())
                try:
                    for n, hdu in enumerate(fits):
                        yield (n, hdu)
                finally:
                    fits.close()
            except IOError:
                yield None

        hdus = generate_hdus()

        def wrap(x):
            if x:
                self.run_hdu(reducer, x)
            else:
                return None

        hdus_ = itertools.imap(wrap, hdus)
        return reducer.reduce_fits_gen(file, hdus_)

    def run_hdu(self, reducer, (n, hdu)):
        def hu(): yield hdu.header

        def du(): yield hdu.data

        hu_ = itertools.imap(lambda(x): self.run_header_unit(reducer, n, x),
                             hu())
        du_ = itertools.imap(lambda(x): self.run_data_unit(reducer, n, x),
                             du())

        return reducer.reduce_hdu_gen(n, hdu, hu_, du_)

    def run_header_unit(self, reducer, n, hu):
        # No decomposition or recursion needed since there's nothing below
        return reducer.reduce_header_unit(n, hu)

    def run_data_unit(self, reducer, n, du):
        # No decomposition or recursion needed since there's nothing below
        return reducer.reduce_data_unit(n, du)


class GenReducer(object):
    # Note that you can ignore the generators completely to avoid
    # recursing deeper.
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def reduce_archive_gen(self, root, bundles_): pass

    @abc.abstractmethod
    def reduce_bundle_gen(self, archive, lid, collections_): pass

    @abc.abstractmethod
    def reduce_collection_gen(self, archive, lid, products_): pass

    @abc.abstractmethod
    def reduce_product_gen(self, archive, lid, fits_files_): pass

    @abc.abstractmethod
    def reduce_fits_gen(self, fits, hdus_): pass

    @abc.abstractmethod
    def reduce_hdu_gen(self, n, hdu, hu_, du_): pass

    @abc.abstractmethod
    def reduce_header_unit(self, n, hu): pass

    @abc.abstractmethod
    def reduce_data_unit(self, n, du): pass


class NullGenReducer(GenReducer):
    def reduce_archive_gen(self, root, bundles_):
        pass

    def reduce_bundle_gen(self, archive, lid, collections_):
        pass

    def reduce_collection_gen(self, archive, lid, products_):
        pass

    def reduce_product_gen(self, archive, lid, fits_files_):
        pass

    def reduce_fits_gen(self, fits, hdus_):
        pass

    def reduce_hdu_gen(self, n, hdu, hu_, du_):
        pass

    def reduce_header_unit(self, n, hu):
        pass

    def reduce_data_unit(self, n, du):
        pass


class Reducer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def reduce_archive(self, root, bundles): pass

    @abc.abstractmethod
    def reduce_bundle(self, archive, lid, collections): pass

    @abc.abstractmethod
    def reduce_collection(self, archive, lid, products): pass

    @abc.abstractmethod
    def reduce_product(self, archive, lid, fits_files): pass

    @abc.abstractmethod
    def reduce_fits(self, fits, hdus): pass

    @abc.abstractmethod
    def reduce_hdu(self, n, hdu, hu, du): pass

    @abc.abstractmethod
    def reduce_header_unit(self, n, hu): pass

    @abc.abstractmethod
    def reduce_data_unit(self, n, du): pass


class NullReducer(Reducer):
    def reduce_archive(self, root, bundles):
        pass

    def reduce_bundle(self, archive, lid, collections):
        pass

    def reduce_collection(self, archive, lid, products):
        pass

    def reduce_product(self, archive, lid, fits_files):
        pass

    def reduce_fits(self, fits, hdus):
        pass

    def reduce_hdu(self, n, hdu, hu, du):
        pass

    def reduce_header_unit(self, n, hu):
        pass

    def reduce_data_unit(self, n, du):
        pass

    def __str__(self):
        return 'NullReducer'

    def __repr__(self):
        return 'NullReducer'


class CompositeReducer(Reducer):
    def __init__(self, reducers):
        assert reducers
        self.reducers = reducers

    def reduce_archive(self, root, bundles_list):
        return [r.reduce_archive(root, bundles)
                for (r, bundles)
                in checked_zip(self.reducers, zip(*bundles_list))]

    def reduce_bundle(self, archive, lid, collections_list):
        return [r.reduce_bundle(archive, lid, collections)
                for (r, collections)
                in checked_zip(self.reducers, zip(*collections_list))]

    def reduce_collection(self, archive, lid, products_list):
        return [r.reduce_collection(archive, lid, products)
                for (r, products)
                in checked_zip(self.reducers, zip(*products_list))]

    def reduce_product(self, archive, lid, fits_files_list):
        return [r.reduce_product(archive, lid, files)
                for (r, files)
                in checked_zip(self.reducers, zip(*fits_files_list))]

    def reduce_fits(self, fits, hdus_list):
        return [r.reduce_fits(fits, hdus)
                for (r, hdus)
                in checked_zip(self.reducers, zip(*hdus_list))]

    def reduce_hdu(self, n, hdu, hu_list, du_list):
        return [r.reduce_hdu(n, hdu, hu, du)
                for (r, hu, du)
                in checked_zip(self.reducers, hu_list, du_list)]

    def reduce_header_unit(self, n, hu):
        return [r.reduce_header_unit(n, hu) for r in self.reducers]

    def reduce_data_unit(self, n, du):
        return [r.reduce_data_unit(n, du) for r in self.reducers]

    def __str__(self):
        return 'CompositeReducer(%s)' % str(self.reducers)


def checked_zip(*args):
    # Check that all the arguments are the same length
    argSet = set([len(arg) for arg in args])
    assert len(argSet) == 1, str(args)
    return zip(*args)


class ReducerAdapter(GenReducer):
    def __init__(self, reducer):
        assert reducer
        self.reducer = reducer

    def reduce_archive_gen(self, root, bundles_):
        return self.reducer.reduce_archive(root, list(bundles_))

    def reduce_bundle_gen(self, archive, lid, collections_):
        return self.reducer.reduce_bundle(archive, lid, list(collections_))

    def reduce_collection_gen(self, archive, lid, products_):
        return self.reducer.reduce_collection(archive, lid, list(products_))

    def reduce_product_gen(self, archive, lid, fits_files_):
        return self.reducer.reduce_product(archive, lid, list(fits_files_))

    def reduce_fits_gen(self, fits, hdus_):
        return self.reducer.reduce_fits(fits, list(hdus_))

    def reduce_hdu_gen(self, n, hdu, hu_, du_):
        return self.reducer.reduce_hdu(n, hdu, hu_.next(), du_.next())

    def reduce_header_unit(self, n, hu):
        return self.reducer.reduce_header_unit(n, hu)

    def reduce_data_unit(self, n, du):
        return self.reducer.reduce_data_unit(n, du)

    def __str__(self):
        return 'ReducerAdapter(%s)' % str(self.reducer)

############################################################
# testing
############################################################


class CheckFitsReducer(NullGenReducer):
    # Files map to code
    def reduce_archive_gen(self, root, bundles_):
        list(bundles_)
        print 'archive', root

    def reduce_bundle_gen(self, archive, lid, collections_):
        list(collections_)
        print 'bundle', lid

    def reduce_collection_gen(self, archive, lid, products_):
        list(products_)

    def reduce_product_gen(self, archive, lid, files_):
        print 'product', lid
        msgs = [msg for msg in list(files_) if msg is not None]
        msg_count = len(msgs)
        if msg_count is 0:
            return

        print 'Product \'%s\' has FITS error(s): %s' % (lid, ' '.join(msgs))

    def reduce_fits_gen(self, fits, hdus_):
        try:
            hdus_.next()
            return None
        except Exception as e:
            return e.message

    def __str__(self):
        return 'CheckFitsReducer'


class ReducerAdapterWithFitsFix(ReducerAdapter):
    def __init__(self, reducers):
        ReducerAdapter.__init__(self, reducers)

    def reduce_product_gen(self, archive, lid, fits_files_):
        try:
            return ReducerAdapter.reduce_product_gen(self, archive,
                                                     lid, fits_files_)
        except Exception:
            return [1, 1, 1, 1, 1]

    def __str__(self):
        return 'ReducerAdapterWithFitsFix(%s)' % str(self.reducers)


class NullReducer2(NullReducer):
    def reduce_archive(self, root, bundles):
        print 'archive %s' % (root,)

    def reduce_bundle(self, archive, lid, collections):
        print 'bundle %s' % (str(lid),)

    def reduce_collection(self, archive, lid, products):
        print 'collection %s' % (str(lid),)

    def __str__(self):
        return 'NullReducer2'

    def __repr__(self):
        return 'NullReducer2()'

if __name__ == '__main__':
    archive = FileArchives.get_any_archive()
    if True:
        # check a pass with limited recursion
        reducer = CheckFitsReducer()
        Runner().run_archive(reducer, archive)
    else:
        # check composite reduction
        rs = 4 * [NullReducer()]
        rs.append(NullReducer2())
        c = CompositeReducer(rs)
        # r = ReducerAdapterWithFitsFix(c)
        r = ReducerAdapter(c)

        # This raises an exception on a malformed FITS file.  How do I
        # catch that and still test the composition?
        Runner().run_archive(r, archive)
