"""
**SCRIPT:** Run through the archive and print the sorted set of LIDs of
the products whose FITS file contains sufficient datetime information
to calculate start and stop times of the observation.
"""

from pdart.pds4.Archives import *
from pdart.exceptions.Combinators import *
from pdart.reductions.Reduction import *

# These are the suffixes in the development archive that have
# sufficient datetime information: c0m c1m crj d0m drz flt q0m raw


def _unions(sets):
    """Union a list of sets."""
    res = set()
    for s in sets:
        res |= s
    return res


class CheckTimesReduction(Reduction):
    """
    When run on an archive, return a set of LIDs of the products whose
    FITS file contains sufficient datetime information to calculate
    start and stop times of the observation.
    """
    def reduce_archive(self, archive_root, get_reduced_bundles):
        return _unions(get_reduced_bundles())

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        return _unions(get_reduced_collections())

    def reduce_collection(self, archive, lid, get_reduced_products):
        reduced_products = [lid
                            for lid in get_reduced_products()
                            if lid is not None]
        return set(reduced_products)

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        file_has_datetime = [f
                             for f in get_reduced_fits_files()
                             if f is not None]
        if file_has_datetime:
            return lid
        else:
            return None

    def reduce_fits_file(self, file, get_reduced_hdus):
        try:
            reduced_hdus = get_reduced_hdus()
            return reduced_hdus[0]
        except IOError:
            # couldn't read the file
            return None

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        if n == 0:
            return get_reduced_header_unit()
        else:
            pass

    def reduce_header_unit(self, n, header_unit):
        if n == 0:
            try:
                date_obs = header_unit['DATE-OBS']
                time_obs = header_unit['TIME-OBS']
                exptime = header_unit['EXPTIME']
                return True
            except KeyError:
                return None


if __name__ == '__main__':
    reduction = CheckTimesReduction()
    archive = get_any_archive()
    lids = run_reduction(reduction, archive)
    print 60 * '-'
    for lid in sorted(lids):
        print lid
