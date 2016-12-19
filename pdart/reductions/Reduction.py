"""
The central data structure of the PDART project is the archive, which
is a hierarchical structure.  The archive consists of bundles, which
consist of collections, which in turn consist of products.  You may
think of the product consisting of various types of files.  FITS files
may be further subdivided: they consist of HDUs, each of which
contains a header section and possibly a data section.

Sometimes when working on hierarchical structures, it's most useful to
think in terms of converting between similar types of hierarchies.
For instance, if we need to write a function that creates PDS4 labels
for an archive, it will consist of bits of code that each create a
PDS4 label for a bundle, and those bits of code will each contain code
to create a PDS4 label for a collection, and so on.

Note that the conversion may collapse some parts of the hierarchy.  In
the extreme case, a conversion may reduce an entire hierarchy to a
single value.

In mathematics, conversions between similar types of hierarchies are
called *catamorphisms*.  In some programming languages, they are
called *folds*.  Here, I call them *reductions* after the Python
function :func:`reduce` which is a built-in catamorphism: it reduces a
list to a single (possibly compound) value.

Frankly, :func:`reduce` isn't used much in Python: for a list it's
usually just as easy to write a ``for``-loop or a list comprehension,
which acts as the boilerplate for the reduction.  But when hierarchies
become large like ours, the boilerplate to do the reduction (multiple
nested ``for``-loops for each level) gets large and bug-prone so it's
easier in the long run to set up some structure to handle it once and
for all.

This module defines the :class:`~pdart.reductions.Reduction.Reduction`
object which contains methods corresponding to each level of the PDART
hierarchy.  Each method tells how to combine (or reduce) the
reductions from the next lower part of the hierarchy.  You then feed a
:class:`~pdart.reductions.Reduction.Reduction` and the archive to
:func:`~.reductions.Reduction.run_reduction` and it will return
the combined result.

For instance, if you wanted to count the number of products in an
archive, rather than writing boilerplate with nested ``for``-loop, you
could think of it in terms of a
:class:`~pdart.reductions.Reduction.Reduction`.

    * A product is reduced to the number 1.

    * A collection is reduced to the sum of its products' reductions.

    * A bundle is reduced to the sum of its collections' reductions.

    * An archive is reduced to the sum of its bundles' reductions.

When you run that all, the hierarchy that makes up an archive is
reduced to a single number: the number of products in the file.

You could implement this as follows::

    class ProductCountReduction(Reduction):
        def reduce_archive(self, archive_root, get_reduced_bundles):
            return sum(get_reduced_bundles())

        def reduce_bundle(self, archive, lid, get_reduced_collections):
            return sum(get_reduced_collections())

        def reduce_collection(self, archive, lid, get_reduced_products):
            return sum(get_reduced_products())

        def reduce_product(self, archive, lid, get_reduced_fits_files):
            return 1

    run_reduction(ProductCountReduction(), archive)

(Note the implementation quirk that instead of returning the reduction
of the next lower level, we return a *function* that returns the
reduction of the next lower level.  This allows us to avoid recursing
all the way to the bottom of the structure when we don't need to.  For
instance, in this case since we do not need to open and parse any FITS
files--an expensive action--we won't, even though the reduction
hierarchy is capable of going down to individual header and data
units.  *If you don't call it, it won't be used.*)

**For the most part, you won't need to write reductions.** We
generally start out with a reduction that flattens the archive
hierarchy into SQLite tables.  This simplifies the logic later on as
flat data is easier to work with and we can also include persistent
data in the database.

Reductions are a bit easier to work with in a language with
typechecking by a compiler.  In Python, it's up to the programmer to
keep things straight.  :func:`reduction_type_documentation` can be
used to produce a documentation string to remind the user what types
are supposed to be used where.
"""
import os.path

import abc
import numpy
import pyfits

from pdart.rules.Combinators import parallel_list

from typing import Any, Callable, TYPE_CHECKING, Union
if TYPE_CHECKING:
    from pdart.pds4.Archive import Archive
    from pdart.pds4.Bundle import Bundle
    from pdart.pds4.Collection import Collection
    from pdart.pds4.Product import Product
    from pdart.pds4.File import File
    from pdart.pds4.LID import LID

    # These are Unions because, for some reason (bugginess?) mypy doesn't
    # recognize the bare type names.
    _HDU = Union[pyfits.FitsHDU]
    _HeaderUnit = Union[pyfits.Header]
    _DataUnit = Union[numpy.ndarray]

# I wish I could provide precise types for Reductions.  But mypy's
# generics seem to be buggy enough that they don't work.  I tried
# making Reduction generic, then introduced a type error in a
# descendant of it in test_Reduction.  Mypy didn't catch the type
# error.  So I'm leaving these all dynamically typed as Any.  :-(


class Reduction(object):
    """
    A collection of methods to reduce PDS4 and FITS structure into a
    new form.
    """
    def reduce_archive(self, archive_root, get_reduced_bundles):
        # type: (unicode, Callable[[], List[Any]]) -> Any
        pass

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        # type: (Archive, LID, Callable[[], List[Any]]) -> Any
        pass

    def reduce_collection(self, archive, lid, get_reduced_products):
        # type: (Archive, LID, Callable[[], List[Any]]) -> Any
        pass

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        # type: (Archive, LID, Callable[[], List[Any]]) -> Any
        pass

    def reduce_fits_file(self, file, get_reduced_hdus):
        # type: (File, Callable[[], List[Any]]) -> Any
        pass

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        # type: (int, _HDU, Callable[[], Any], Callable[[], Any]) -> Any
        pass

    def reduce_header_unit(self, n, header_unit):
        # type: (int, _HeaderUnit) -> Any
        pass

    def reduce_data_unit(self, n, data_unit):
        # type: (int, _DataUnit) -> Any
        pass


def reduction_type_documentation(dict):
    # type: (Dict[str, Any]) -> str
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
    """
    An abstract class to run a
    :class:`~pdart.reductions.Reduction.Reduction` on an
    :class:`~pdart.pds4.Archive.Archive` or one of its substructures
    (:class:`~pdart.pds4.Bundle.Bundle`,
    :class:`~pdart.pds4.Collection.Collection`, etc.).
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def run_archive(self, reduction, archive):
        # type: (Reduction, Archive) -> Any
        """
        Run the :class:`~pdart.reductions.Reduction.Reduction` on an
        :class:`~pdart.pds4.Archive.Archive`.
        """
        pass

    @abc.abstractmethod
    def run_bundle(self, reduction, bundle):
        # type: (Reduction, Bundle) -> Any
        """
        Run the :class:`~pdart.reductions.Reduction.Reduction` on an
        :class:`~pdart.pds4.Bundle.Bundle`.
        """
        pass

    @abc.abstractmethod
    def run_collection(self, reduction, collection):
        # type: (Reduction, Collection) -> Any
        """
        Run the :class:`~pdart.reductions.Reduction.Reduction` on an
        :class:`~pdart.pds4.Collection.Collection`.
        """
        pass

    @abc.abstractmethod
    def run_product(self, reduction, product):
        # type: (Reduction, Product) -> Any
        """
        Run the :class:`~pdart.reductions.Reduction.Reduction` on an
        :class:`~pdart.pds4.Product.Product`.
        """
        pass

    @abc.abstractmethod
    def run_fits_file(self, reduction, file):
        # type: (Reduction, File) -> Any
        """
        Run the :class:`~pdart.reductions.Reduction.Reduction` on an
        FITS file.
        """
        pass

    @abc.abstractmethod
    def run_hdu(self, reduction, n, hdu):
        # type: (Reduction, int, _HDU) -> Any
        """
        Run the :class:`~pdart.reductions.Reduction.Reduction` on an
        HDU within a FITS file.
        """
        pass

    @abc.abstractmethod
    def run_header_unit(self, reduction, n, header_unit):
        # type: (Reduction, int, _HeaderUnit) -> Any
        """
        Run the :class:`~pdart.reductions.Reduction.Reduction` on an
        header unit within a FITS file.
        """
        pass

    @abc.abstractmethod
    def run_data_unit(self, reduction, n, data_unit):
        # type: (Reduction, int, _DataUnit) -> Any
        """
        Run the :class:`~pdart.reductions.Reduction.Reduction` on an
        data unit within a FITS file.
        """
        pass


class DefaultReductionRunner(ReductionRunner):
    """
    An algorithm to recursively reduce PDS4 and FITS structures
    according to a :class:`~pdart.reductions.Reduction.Reduction`
    instance.

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
            filepath = file.full_filepath()
            if os.path.splitext(filepath)[1] == '.fits':
                fits = pyfits.open(filepath)

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
            else:
                # Non-FITS files have no hdus, reduced or otherwise
                return []

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
    # (Reduction, Archive) -> Any
    """
    Run a :class:`~pdart.reductions.Reduction.Reduction` on an
    :class:`~pdart.pds4.Archive.Archive` using the default recursion.
    """
    return DefaultReductionRunner().run_archive(reduction, archive)
