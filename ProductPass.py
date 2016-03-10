import abc

import pyfits

import Product


class ProductPassRunner(object):

    # Has no internal state.  Turn into a function?  Why did I reify
    # the runner in the other case?

    def run_product(self, productPass, product):
        assert isinstance(productPass, ProductPass), 'e'
        assert isinstance(product, Product.Product), 'f'
        return productPass.process_product(
            product,
            [self.run_file(productPass, f) for f in product.files()])

    def run_file(self, productPass, file):
        assert isinstance(productPass, ProductPass), 'g'
        assert file, 'd'
        try:
            fits = pyfits.open(file.full_filepath())
            try:
                return productPass.process_file(
                    file,
                    [self.run_hdu(productPass, n, hdu)
                     for n, hdu in enumerate(fits)])
            finally:
                fits.close()
        except Exception as e:
            return productPass.process_file(file, e)

    def run_hdu(self, productPass, n, hdu):
        assert isinstance(productPass, ProductPass), 'a'
        assert isinstance(n, int), 'b'
        assert hdu is not None, 'c'
        h = productPass.process_hdu_header(n, hdu.header)
        if hdu.fileinfo()['datSpan']:
            d = productPass.process_hdu_data(n, hdu.data)
        else:
            d = None
        return productPass.process_hdu(hdu, h, d)


class ProductPass(object):

    # Has no internal state.  Names can be changed later.  Runs
    # itself.  Ahh, the rub comes when you try to compose.  So we
    # return a vector of results and pass them up.

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def process_product(self, product, files):
        pass

    @abc.abstractmethod
    def process_file(self, hdu_list_or_exception):
        pass

    @abc.abstractmethod
    def process_hdu(self, hdu, h, d):
        pass

    @abc.abstractmethod
    def process_hdu_header(self, n, header):
        pass

    @abc.abstractmethod
    def process_hdu_data(self, n, data):
        pass


class CompositeProductPass(ProductPass):
    def __init__(self, passes):
        assert passes, 'x'
        self.passes = passes

    def process_hdu_header(self, n, header):
        return [productPass.process_hdu_header(n, header)
                for productPass in self.passes]

    def process_hdu_data(self, n, data):
        return [productPass.process_hdu_data(n, data)
                for productPass in self.passes]

    def process_hdu(self, hdus, hs, ds):
        return [productPass.process_hdu(hdu, h, d)
                for (productPass, hdu, h, d) in zip(self.passes, hdus, hs, ds)]

    def process_file(self, file, hdu_lists_or_exception):
        # TODO This is not a solution.
        if isinstance(hdu_lists_or_exception, Exception):
            return [hdu_lists_or_exception for _ in self.passes]
        else:
            return [productPass.process_file(file, he)
                    for (productPass, he)
                    in zip(self.passes, hdu_lists_or_exception)]

    def process_product(self, product, files):
        return [productPass.process_product(product, file)
                for (productPass, file) in zip(self.passes, files)]

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

    def process_hdu(self, hdu, h, d):
        return h

    def process_file(self, file, hs):
        if isinstance(hs, Exception):
            return 'UNKNOWN'
        else:
            return hs[0]

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

    def process_hdu(self, hdu, h, d):
        # Grab the result from the hdu_header and augment with info
        # from the hdu's fileinfo()
        res = h
        info = hdu.fileinfo()
        res['header_offset'] = h_off = info['hdrLoc']
        res['data_offset'] = d_off = info['datLoc']
        res['header_size'] = d_off - h_off
        res['local_identifier'] = 'hdu_%d' % 0  # TODO Wrong!
        return res

    def process_file(self, file, hdu_list):
        return (file.basename, hdu_list)

    def process_product(self, product, files):
        # A dict of lists of HDUs indexed by the file's basename
        return ('File_Area_Observational', dict(files))


class ProductLabelProductPass(CompositeProductPass):
    """
    When run, produce a dictionary such that dict['target_set'] is the
    set of targets named in primary HDUs within this product and
    dict['File_Area_Observational'] is a dict of lists of HDU
    information for each file, indexed by the files' basenames.
    """
    def __init__(self):
        passes = [TargetProductPass(), FileAreaProductPass()]
        super(ProductLabelProductPass, self).__init__(passes)

    def process_product(self, product, targs):
        res0 = super(ProductLabelProductPass,
                             self).process_product(product, targs)
        try:
            res = dict(res0)
            return res
        except:
            return None
