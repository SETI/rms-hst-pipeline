import abc
import itertools
import pyfits
import traceback
import xml.dom

import pdart.rules.Combinators
import pdart.rules.ExceptionInfo
import pdart.rules.Result


class Runner(object):
    def run_archive(self, reducer, archive):
        def get_reduced_bundles():
            bundles = list(archive.bundles())
            return parallel_list('run_archive',
                                 [lambda: self.run_bundle(reducer, bundle)
                                  for bundle in bundles])

        return reducer.reduce_archive(archive.root, get_reduced_bundles)

    def run_bundle(self, reducer, bundle):
        def get_reduced_collections():
            collections = list(bundle.collections())
            return parallel_list('run_bundle',
                                 [lambda: self.run_collection(reducer,
                                                              collection)
                                  for collection in collections])

        return reducer.reduce_bundle(bundle.archive, bundle.lid,
                                     get_reduced_collections)

    def run_collection(self, reducer, collection):
        def get_reduced_products():
            products = list(collection.products())
            return parallel_list('run_collection',
                                 [lambda: self.run_product(reducer, product)
                                  for product in products])

        return reducer.reduce_collection(collection.archive,
                                         collection.lid,
                                         get_reduced_products)

    def run_product(self, reducer, product):
        def get_reduced_files():
            files = list(product.files())
            return parallel_list('run_product',
                                 [lambda: self.run_file(reducer, file)
                                  for file in files])

        return reducer.reduce_product(product.archive, product.lid,
                                      get_reduced_files)

    def run_fits(self, reducer, file):
        def get_reduced_hdus():
            fits = pyfits.open(file.full_filepath())
            try:
                return parallel_list('run_fits',
                                     [lambda: self.run_hdu(self, reducer,
                                                           (n, hdu))
                                      for n, hdu in enumerate(fits)])
            finally:
                fits.close()

        return reducer.reduce_fits(file, get_reduced_hdus)

    def run_hdu(self, reducer, (n, hdu)):
        def get_reduced_header_unit():
            return reducer.reduce_header_unit(n, lambda: hdu.header)

        def get_reduced_data_unit():
            return reducer.reduce_data_unit(n, lambda: hdu.data)

        return reducer.reduce_hdu(n, hdu,
                                  get_reduced_header_unit,
                                  get_reduced_data_unit)

    def run_header_unit(self, reducer, n, hu):
        return reducer.reduce_header_unit(n, get_header_unit)

    def run_data_unit(self, reducer, n, du):
        return reducer.reduce_data_unit(n, get_data_unit)


class Reducer(object):
    def reduce_archive(self, archive_root, get_reduced_bundles):
        pass

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        pass

    def reduce_collection(self, archive, lid, get_reduced_products):
        pass

    def reduce_product(self, archive, lid, get_reduced_files):
        pass

    def reduce_fits(self, file, get_reduced_hdus):
        pass

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        pass

    def reduce_header_unit(self, n, get_header_unit):
        pass

    def reduce_data_unit(self, n, get_data_unit):
        pass


class TestReducer(object):
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()
        print 'In archive at %s' % archive_root

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        print 'bundle %s' % lid

    def reduce_collection(self, archive, lid, get_reduced_products):
        pass

    def reduce_product(self, archive, lid, get_reduced_files):
        pass

    def reduce_fits(self, file, get_reduced_hdus):
        pass

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        pass

    def reduce_header_unit(self, n, get_header_unit):
        pass

    def reduce_data_unit(self, n, get_data_unit):
        pass

if __name__ == '__main__':
    def foo():
        raise Exception("You're killing me!")
    try:
        pdart.rules.Combinators.normalized_exceptions(foo)()
    except pdart.rules.ExceptionInfo.CalculationException as e:
        print e.exception_info.to_pretty_xml()

    print 60 * '-'

    try:
        pdart.rules.Combinators.parallel_list('Labels are for jars!',
                                                   3 * [foo])
    except pdart.rules.ExceptionInfo.CalculationException as e:
        print e.exception_info.to_pretty_xml()
