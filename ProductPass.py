import abc
import pprint
import traceback

import pyfits

import FileArchives
import Heuristic
import LID
import Product


class ProductPassRunner(object):

    def __str__(self):
        return 'ProductPassRunner'

    def __repr__(self):
        return 'ProductPassRunner()'

    def run_product(self, product_pass, product):
        results = Heuristic.sequence([self.run_file(product_pass, file)
                                      for file in product.files()])
        if results.is_failure():
            return results

        return Heuristic.HFunction(
            lambda (files):
            product_pass.process_product(product,
                                         files)).run(results.value)

    def run_file(self, product_pass, file):
        filepath = file.full_filepath()
        res = Heuristic.HFunction(lambda fp: pyfits.open(fp)).run(filepath)
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
            hdus = res.value

        return Heuristic.Success(product_pass.process_file(file, hdus))

    def run_hdu(self, product_pass, n, hdu):
        hdr = product_pass.process_hdu_header(n, hdu.header)
        dat = product_pass.process_hdu_data(n, hdu.data)
        return Heuristic.Success(product_pass.process_hdu(n, hdu, hdr, dat))


class ProductPass(object):

    # Has no internal state.  Names can be changed later.  Runs
    # itself.  Ahh, the rub comes when you try to compose.  So we
    # return a vector of results and pass them up.

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def process_product(self, product, files):
        pass

    @abc.abstractmethod
    def process_file(self, file, hdus):
        pass

    @abc.abstractmethod
    def process_hdu(self, n, hdu, h, d):
        pass

    @abc.abstractmethod
    def process_hdu_header(self, n, header):
        pass

    @abc.abstractmethod
    def process_hdu_data(self, n, data):
        pass


# If you have process_product() for each element pass return a
# key-value pair, then dict(runner.run_product(product,
# compositePass)) returns a useful dictionary.  I like that.

# Now I don't have any pre- or post-processing (i.e., a begin
# parameter).  I'm gambling I don't need it.  We're just extracting
# information functionally.  No state involved, no state needed.

# Try it out.


class TargetProductPass(ProductPass):
    """
    When run, return the pair ('target_set', ts) where ts is the set
    of the values of the key 'TARGNAME' in the primary HDUs of all
    files in the product.  If it does not contain exactly one element,
    we have missing data (0) or ambiguity (>1).
    """

    def __str__(self):
        return 'TargetProductPass'

    def __repr__(self):
        return 'TargetProductPass()'

    def process_hdu_header(self, n, header):
        if n == 0:
            try:
                res = header['TARGNAME']
            except KeyError:
                res = None
        else:
            res = None
        return res

    def process_hdu_data(self, n, data):
        return None

    def process_hdu(self, n, hdu, h, d):
        return h

    def process_file(self, file, hdus):
        # It's found, if at all, in the first HDU
        return hdus[0]

    def process_product(self, product, targs):
        res = ('target_set',
               set([targ for targ in targs if targ is not None]))
        return res


BITPIX_TABLE = {
    # TODO Verify these
    8: 'UnsignedByte',
    16: 'SignedMSB2',
    32: 'SignedMSB4',
    64: 'SignedMSB8',
    -32: 'IEEE754MSBSingle',
    -62: 'IEEE754MSBDouble'
    }


AXIS_NAME_TABLE = {
    1: 'Line',
    2: 'Sample'
    # 3: 'Color'?
    }


class FileAreaProductPass(ProductPass):
    """
    When run, return the pair ('File_Area_Observational', dict) where
    dict is a dictionary with file basenames as keys and lists of HDU
    info as values.
    """

    def __str__(self):
        return 'FileAreaProductPass'

    def __repr__(self):
        return 'FileAreaProductPass()'

    def process_hdu_data(self, n, data):
        pass

    def process_hdu_header(self, n, header):
        res = {}
        res['axes'] = naxis = header['NAXIS']

        res['Axis_Array'] = \
            [{'axis_name': AXIS_NAME_TABLE[i],
              'elements': header['NAXIS%d' % i],
              'sequence_number': i} for i in range(1, naxis + 1)]

        res['data_type'] = BITPIX_TABLE[header['BITPIX']]

        try:
            res['scaling_factor'] = header['BSCALE']
        except KeyError:
            pass

        try:
            res['value_offset'] = header['BZERO']
        except KeyError:
            pass
        return res

    def process_hdu(self, n, hdu, h, d):
        # Grab the result from the hdu_header and augment with info
        # from the hdu's fileinfo()
        res = h
        info = hdu.fileinfo()
        res['header_offset'] = h_off = info['hdrLoc']
        res['data_offset'] = d_off = info['datLoc']
        res['header_size'] = d_off - h_off
        res['local_identifier'] = 'hdu_%d' % n
        return res

    def process_file(self, file, hdus):
        return (file.basename, hdus)

    def process_product(self, product, files):
        # A dict of lists of HDUs indexed by the file's basename
        return ('File_Area_Observational', dict(files))


class CompositeProductPass(ProductPass):
    def __init__(self, passes):
        assert passes, 'x'
        self.passes = passes

    def __str__(self):
        return 'CompositeProductPass(%s)' % self.passes

    def __repr__(self):
        return 'CompositeProductPass(%r)' % self.passes

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


class ProductLabelProductPass(CompositeProductPass):
    """
    When run, produce a dictionary such that dict['target_set'] is the
    set of targets named in primary HDUs within this product and
    dict['File_Area_Observational'] is a dict of lists of HDU
    information for each file, indexed by the files' basenames.
    """
    def __str__(self):
        return 'ProductLabelProductPass'

    def __repr__(self):
        return 'ProductLabelProductPass()'

    def __init__(self):
        passes = [TargetProductPass(), FileAreaProductPass()]
        super(ProductLabelProductPass, self).__init__(passes)

    def process_product(self, product, files):
        res0 = super(ProductLabelProductPass,
                     self).process_product(product, files)
        try:
            res = dict(res0)
            return res
        except:
            return None


if __name__ == '__main__':
    lid = LID.LID('urn:nasa:pds:hst_09746:data_acs_raw:visit_25')
    product = Product.Product(FileArchives.get_any_archive(), lid)
    # pp = FileAreaProductPass()
    # pp = TargetProductPass()
    pp = ProductLabelProductPass()
    ppr = ProductPassRunner()
    print 60 * '-'
    print 8 * '-', product
    try:
        res = ppr.run_product(pp, product)
        print "SUCCESSFUL CALCULATION"
        if res.is_success():
            print pprint.PrettyPrinter(indent=2, width=78).pprint(res.value)
        else:
            print pprint.PrettyPrinter(indent=2,
                                       width=78).pprint(res.exceptions)

    except:
        print "FAILED CALCULATION"
        print traceback.format_exc()
