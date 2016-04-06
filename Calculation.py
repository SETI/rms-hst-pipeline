import abc
import itertools
import pyfits
import traceback

import FileArchives

##############################
# structured exception info
##############################


class ExceptionInfo(object):
    # def to_xml(self):  pass # to be implemented
    pass


class SingleExceptionInfo(ExceptionInfo):
    def __init__(self, exception, stack_trace):
        self.exception = exception
        self.stack_trace = stack_trace

    def __str__(self):
        return 'SimpleExceptionInfo(%s)' % str(self.exception)


class GroupedExceptionInfo(ExceptionInfo):
    def __init__(self, label, exception_infos):
        self.label = label
        self.exception_infos = exception_infos

    def __str__(self):
        return 'GroupedExceptionInfo(%r, %s)' % \
            (self.label, self.exception_infos)


##############################
# Results: values or exceptions
##############################


class _Result(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def is_success(self):
        pass

    def is_failure(self):
        return not self.is_success()


class _Failure(_Result):
    def __init__(self, exception_info):
        _Result.__init__(self)
        self.exception_info = exception_info

    def is_success(self):
        return False

    def __str__(self):
        return '_Failure(%s)' % (self.exception_info, )


class _Success(_Result):
    def __init__(self, value):
        _Result.__init__(self)
        self.value = value

    def is_success(self):
        return True

    def __str__(self):
        return '_Success(%s)' % (self.value, )


##############################
# exception to carry structured info
##############################


class CalculationException(Exception):
    def __init__(self, msg, exception_info):
        Exception.__init__(self, msg)
        self.exception_info = exception_info

##############################
# conversion of code
##############################


def _code_to_rcode(func):
    def rfunc(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
        except Exception as e:
            exception_info = SingleExceptionInfo(e, traceback.format_exc())
            return _Failure(exception_info)
        return _Success(res)
    return rfunc


def _rcode_to_code(rfunc):
    def func(*args, **kwargs):
        res = rfunc(*args, **kwargs)
        if res.is_success():
            return res.value
        else:
            raise CalculationException('', res.exception_info)
    return func


def normalized_exceptions(func):
    return _rcode_to_code(_code_to_rcode(func))


##############################
# combinators
##############################


def multiple_implementations(label, *funcs):
    def afunc(*args, **kwargs):
        exception_infos = []
        for func in funcs:
            res = _code_to_rcode(func)(*args, **kwargs)
            if res.is_success():
                return res.value
            else:
                exception_infos.append(res.exception_info)
        # if we got here, there were no successes
        exception_info = GroupedExceptionInfo(label, exception_infos)
        raise CalculationException(exception_info)
    return afunc


def parallel_arguments(label, func, *arg_funcs):
    def pfunc():
        exception_infos = []
        results = []
        for arg_func in arg_funcs:
            arg_res = _code_to_rcode(arg_func)()
            if arg_res.is_success():
                results.append(arg_res.value)
            else:
                exception_infos.append(arg_res.exception_info)
        if exception_infos:
            # We failed if any arg_func failed
            exception_info = GroupedExceptionInfo(label, exception_infos)
            raise CalculationException(exception_info)
        else:
            return f(results)
    return pfunc


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
        return reducer.reduce_archive(archive.root, bundles_)

    def run_bundle(self, reducer, bundle):
        colls = bundle.collections()
        colls_ = itertools.imap(lambda(x): self.run_collection(reducer, x),
                                colls)
        return reducer.reduce_bundle(bundle.archive, bundle.lid, colls_)

    def run_collection(self, reducer, collection):
        products = collection.products()
        products_ = itertools.imap(lambda(x): self.run_product(reducer, x),
                                   products)
        return reducer.reduce_collection(collection.archive, collection.lid,
                                         products_)

    def run_product(self, reducer, product):
        fits_files = product.files()
        fits_files_ = itertools.imap(lambda(x): self.run_fits(reducer, x),
                                     fits_files)
        return reducer.reduce_product(product.archive, product.lid,
                                      fits_files_)

    def run_fits(self, reducer, file):
        def generate_hdus():
            fits = pyfits.open(file.full_filepath())
            try:
                for n, hdu in enumerate(fits):
                    yield (n, hdu)
            finally:
                fits.close()

        hdus = generate_hdus()
        hdus_ = itertools.imap(lambda(x): self.run_hdu(reducer, x), hdus)
        return reducer.reduce_fits(file, hdus_)

    def run_hdu(self, reducer, (n, hdu)):
        def hu(): yield hdu.header

        def du(): yield hdu.data

        hu_ = itertools.imap(lambda(x): self.run_header_unit(reducer, n, x),
                             hu())
        du_ = itertools.imap(lambda(x): self.run_data_unit(reducer, n, x),
                             du())

        return reducer.reduce_hdu(n, hdu, hu_, du_)

    def run_header_unit(self, reducer, n, hu):
        # No decomposition or recursion needed since there's nothing below
        return reducer.reduce_header_unit(n, hu)

    def run_data_unit(self, reducer, n, du):
        # No decomposition or recursion needed since there's nothing below
        return reducer.reduce_data_unit(n, du)


class Reducer(object):
    # Note that you can ignore the generators completely to avoid
    # recursing deeper.

    def reduce_archive(self, root, bundles_): pass

    def reduce_bundle(self, archive, lid, collections_): pass

    def reduce_collection(self, archive, lid, products_): pass

    def reduce_product(self, archive, lid, fits_files_): pass

    def reduce_fits(self, fits, hdus_): pass

    def reduce_hdu(self, n, hdu, hu_, du_): pass

    def reduce_header_unit(self, n, hu): pass

    def reduce_data_unit(self, n, du): pass


############################################################
# testing
############################################################


class CheckFitsReducer(Reducer):
    # Files map to code
    def reduce_archive(self, root, bundles_):
        list(bundles_)

    def reduce_bundle(self, archive, lid, collections_):
        list(collections_)

    def reduce_collection(self, archive, lid, products_):
        list(products_)

    def reduce_product(self, archive, lid, files_):
        msgs = [msg for msg in list(files_) if msg is not None]
        msg_count = len(msgs)
        if msg_count is 0:
            return

        print 'Product \'%s\' has FITS error(s): %s' % (lid, ' '.join(msgs))

    def reduce_fits(self, fits, hdus_):
        try:
            hdus_.next()
            return None
        except Exception as e:
            return e.message

if __name__ == '__main__':
    archive = FileArchives.get_any_archive()
    reducer = CheckFitsReducer()
    Runner().run_archive(reducer, archive)
