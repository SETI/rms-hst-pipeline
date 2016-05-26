from pdart.reductions.WrappedReduction import *


class BadFitsFileReduction(WrappedReduction):
    """
    Wraps a base reduction but augments it so that if the FITS file is
    bad, it returns a specified value instead of raising an IOError.
    """
    def __init__(self, base_reduction):
        WrappedReduction.__init__(self, base_reduction)

    def bad_fits_file_reduction(self, file):
        """
        Return the value to substitute for a FITS file reduction when
        the FITS file is bad. ("Bad" means "raises an IOError.")  This
        method is intended to be overridden.
        """
        pass

    def reduce_fits_file(self, file, get_reduced_hdus):
        try:
            reduced_hdus = get_reduced_hdus()
        except IOError:
            return self.bad_fits_file_reduction(file)

        def get_reduced_hdus_no_fail():
            return reduced_hdus

        return self.base_reduction.reduce_fits_file(file,
                                                    get_reduced_hdus_no_fail)
