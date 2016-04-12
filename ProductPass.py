import abc
import pprint
import traceback
import pprint

import pyfits

import Debug
import FileArchives
import Heuristic


class ProductPass(object):
    """
    Handles a (conceptual) pass over a product to extract, or
    summarize, the information in the product.

    Since I/O takes so much time (particularly opening and parsing
    FITS files), we want to perform all our passes simultaneously and
    avoid multiple runs over the set of files.  But to make the code
    easy to adapt and maintain, we'd like not to have to cram all the
    code for various passes into the same Python function.

    This class allows you to specify the extracting process in pieces,
    each piece in a method, and ProductPassRunner calls the pieces in
    order.  CompositeProductPass lets you combine separate
    ProductPasses so that the extractions are interleaved, saving on
    time spent in I/O.

    Extraction, or summarization, works bottom-up.  At each level
    (product, file, hdu, header and data) you get the object for that
    level and the summaries you've already made from its lower
    component levels.  This process is sometimes called a "fold" or a
    "catamorphism".
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def process_product(self, product, files):
        """
        Extract the needed information from a product and the
        summaries you already made from its files.
        """
        pass

    @abc.abstractmethod
    def process_file(self, file, hdus):
        """
        Extract the needed information from a product's FITS file
        (ArchiveFile) and the summaries you already made from its
        HDUs.
        """
        pass

    @abc.abstractmethod
    def process_hdu(self, n, hdu, h, d):
        """
        Extract the needed information from the FITS HDU, its index,
        and the header and data summaries you already made from this
        HDU.
        """
        pass

    @abc.abstractmethod
    def process_hdu_header(self, n, header):
        """
        Extract the needed information from the FITS HDU header and
        its index.
        """
        pass

    @abc.abstractmethod
    def process_hdu_data(self, n, data):
        """
        Extract the needed information from the FITS HDU data and
        its index.
        """
        pass

    @abc.abstractmethod
    def __repr__(self): pass

    @abc.abstractmethod
    def __str__(self): pass


class ProductPassRunner(object):
    """
    Functionality to run a ProductPass over a Product.  Uses the
    Heuristic framework to propagate errors.
    """

    def run_product(self, product_pass, product):
        """
        Run the ProductPass over the Product and return a
        Heuristic.Result.  If a Success, the value is the intended
        result; if a Failure, it contains the exceptions thrown during
        the calculation.
        """
        results = Heuristic.sequence([self.run_file(product_pass, file)
                                      for file in product.files()])
        if results.is_failure():
            return results

        return Heuristic.HFunction(
            lambda (files):
            product_pass.process_product(product,
                                         files))(results.value)

    def run_file(self, product_pass, file):
        """
        Run the ProductPass over the File and return a Heuristic.Result.
        """
        filepath = file.full_filepath()
        res = Heuristic.HFunction(lambda fp: pyfits.open(fp))(filepath)
        if res.is_failure():
            return res
        else:
            fits = res.value

        hdus = [self.run_hdu(product_pass, n, hdu)
                for n, hdu in enumerate(fits)]
        fits.close()
        res = Heuristic.sequence(hdus)
        if res.is_failure():
            return res
        else:
            hdus = zip(*res.value)

        return Heuristic.HFunction(
            lambda(_): product_pass.process_file(file, hdus))(None)

    def run_hdu(self, product_pass, n, hdu):
        """
        Run the ProductPass over the HDU and return a Heuristic.Result.
        """
        hdr = product_pass.process_hdu_header(n, hdu.header)
        dat = product_pass.process_hdu_data(n, hdu.data)

        return Heuristic.HFunction(
            lambda(_): product_pass.process_hdu(n, hdu, hdr, dat))(None)

    def __str__(self):
        return 'ProductPassRunner'

    def __repr__(self):
        return 'ProductPassRunner()'


# If you have process_product() for each element pass return a
# key-value pair, then dict(runner.run_product(product,
# compositePass)) returns a useful dictionary.  I like that.

# Now I don't have any pre- or post-processing (i.e., a begin
# parameter).  I'm gambling I don't need it.  We're just extracting
# information functionally.  No state involved, no state needed.

# Try it out.


class CompositeProductPass(ProductPass):
    """
    Combines a list of ProductPasses into a single ProductPass.
    Contains the plumbing to run the various calculations to the right
    places.
    """

    def __init__(self, passes):
        assert passes, 'CompositeProductPass: passes should not be empty'
        self.passes = passes

    def process_hdu_header(self, n, header):
        return [product_pass.process_hdu_header(n, header)
                for product_pass in self.passes]

    def process_hdu_data(self, n, data):
        return [product_pass.process_hdu_data(n, data)
                for product_pass in self.passes]

    def process_hdu(self, n, hdu, hs, ds):
        return [product_pass.process_hdu(n, hdu, h, d)
                for (product_pass, h, d)
                in zip(self.passes, hs, ds)]

    def process_file(self, file, hdus):
        return [product_pass.process_file(file, hdu)
                for (product_pass, hdu)
                in zip(self.passes, hdus)]

    def process_product(self, product, files):
        # since zip() is effectively a transposition, zip() can play
        # its own inverse
        return [product_pass.process_product(product, file)
                for (product_pass, file) in zip(self.passes, zip(*files))]

    def __str__(self):
        return 'CompositeProductPass(%s)' % self.passes

    def __repr__(self):
        return 'CompositeProductPass(%r)' % self.passes
