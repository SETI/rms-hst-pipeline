##########################################################################################
# nan_support.py
#
# has_nans(hdulist)
#   return True if any data array in the FITS file contains NaNs.
#
# rewrite_wo_nans(filepath)
#   if a data array in the file contains NaNs, rewrite the FITS file with all NaNs
#   replaced by a non-NaN value that can be identified as a special constant in the PDS4
#   label.
#
# cmp_ignoring_nans(newpath, oldpath)
#   returns True if the files are identical, with the possible exception that the new
#   file contains NaNs in places where the old file contains a replacement value.
##########################################################################################

import filecmp
import numpy as np
import astropy.io.fits as pyfits

def _get_nan_info(hdulist):
    """Return info about any of the data arrays in this FITS file that contain NaNs.
    If the returned list is empty, this file contains no NaNs.

    The returned object is a list of tuples:
        (HDU index, NaN mask, NaN antimask)
    with one entry for each HDU that contains NaNs.
    """

    # Check HDUs one by one...
    nan_info = []
    for k,hdu in enumerate(hdulist):

        # If the data object isn't an array, continue
        if not isinstance(hdu.data, np.ndarray):
            continue

        # Only floating-point arrays can have NaNs
        if hdu.data.dtype.kind != 'f':
            continue

        # Check for NaNs; save info if found
        mask = np.isnan(hdu.data)
        if np.any(mask):
            nan_info.append((k, mask, np.logical_not(mask)))

    return nan_info


def _select_nan_replacement(hdulist, nan_info):
    """Define a constant to replace every NaN value in a data array. The string
    representation of the constant is returned.
    """

    if not nan_info:
        return None

    minvals = []
    for k, _, antimask in nan_info:
        non_nans = hdulist[k].data[antimask]
        minvals.append(np.min(non_nans))

    minval = min(minvals)

    # If all the array values are positive, use zero as the constant
    if minval > 0.:
        return 0.

    # Otherwise, find a lower number represented by an int times a power of ten.
    # Note that this value must be exactly equal to what you will get by interpreting its
    # ASCII representation as found in the XML label!

    # This shouldn't take more than one iteration, but just in case...
    testval = min(minval, -1.)          # start at -1
    while True:
        formatted = '%.0e' % testval    # e.g., "-2e+00"
        testval = float(formatted)
        if testval < minval:
            invalid_constant = testval
            break

        # If this test value was above the cutoff, decrement the negative int before the
        # exponent in the ASCII representation and try again.
        parts = formatted.partition('e')
        formatted = str(int(parts[0])-1) + 'e' + parts[2]
        testval = float(formatted)

    return formatted


def has_nans(hdulist):
    """Return True if any data array in the FITS file contains NaNs.
    """

    return bool(_get_nan_info(hdulist))


def rewrite_wo_nans(filepath):
    """If any of the data arrays in this FITS file contain NaNs, rewrite the file without
    NaNs. Return
        (replacement value, list of modified HDU indices),
    which will be
        (None, [])
    if the file is unchanged.
    """

    hdulist = pyfits.open(filepath)
    nan_info = _get_nan_info(hdulist)
    hdulist.close()
    if not nan_info:
        return (None, [])

    hdulist = pyfits.open(filepath, mode='update')
    replacement = _select_nan_replacement(hdulist, nan_info)
    for k, mask, _ in nan_info:
        hdulist[k].data[mask] = float(replacement)

    hdulist.close()
    return (replacement, [k for (k, _, _) in nan_info])


def cmp_ignoring_nans(newpath, oldpath):
    """Compare two FITS files and return True if the files are identical except for
    possible NaN values in the new file that are not NaN in the old file.
    """

    # Make sure the byte counts match
    file_size = os.path.getsize(oldpath)
    if os.path.getsize(newpath) != file_size:
        return False

    # Gather info about any data arrays in the new file that contain NaNs
    new_hdulist = pyfits.open(newpath)
    nan_info = _get_nan_info(new_hdulist)

    # If there are no NaNs, then a byte-by-byte comparison of the two files will do
    if not nan_info:
        new_hdulist.close()
        return filecmp.cmp(newpath, oldpath)

    # First pass: compare the NaN-containing data arrays one by one...

    # Also, track the "uncompared" byte ranges between these data arrays; they will be
    # tested once we are finished with the data arrays. This is a list of tuples
    #   (start byte, stop byte)
    # that remain to be compared once we are finished with the arrays.
    uncompared = []

    old_hdulist = pyfits.open(oldpath)
    replacement_value = None
    try:            # no matter what happens, be sure to close the FITS files

        # Files must have the same number of HDU lists
        if len(old_hdulist) != len(new_hdulist):
            return False

        # All byte offsets must match
        data_locs = []
        data_spans = []
        for old_hdu, new_hdu in zip(old_hdulist, new_hdulist):

            # The header and data objects must have the same byte counts
            fileinfo = old_hdu.fileinfo()
            header_loc = fileinfo['hdrLoc']
            data_loc   = fileinfo['datLoc']
            data_span  = fileinfo['datSpan']

            fileinfo = new_hdu.fileinfo()
            if fileinfo['hdrLoc'] != header_loc:
                return False
            if fileinfo['datLoc'] != data_loc:
                return False
            if fileinfo['datSpan'] != data_span:
                return False

            data_locs.append(data_loc)
            data_spans.append(data_span)

        # Compare data arrays containing NaNs...
        offset = 0      # byte offset of the first region not yet compared
        for k, mask, antimask in nan_info:

            # The arrays must have the same shape and dtype
            new_data = new_hdulist[k].data
            old_data = old_hdulist[k].data

            if not isinstance(old_data, np.ndarray):
                return False

            if old_data.shape != new_data.shape:
                return False

            if old_data.dtype != new_data.dtype:
                return False

            # The non-NaN array elements must match
            if not np.array_equal(new_data[antimask], old_data[antimask]):
                return False

            # Old values must be a single constant
            old_masked_values = old_data[mask]
            if replacement_value is None:
                replacement_value = old_masked_values[0]

            if np.isnan(replacement_value):
                if not np.all(np.isnan(old_masked_values)):
                    return False
            elif np.any(old_masked_values[:] != replacement_value):
                return False

            # Save the byte range in the file before this array for the second pass
            uncompared.append((offset, data_locs[k]))
            offset = data_locs[k] + data_spans[k]   # byte offset for the next comparison

        # Save the final byte range for the second pass
        if offset != file_size:
            uncompared.append((offset, file_size))

    # No matter what, close the FITS files
    finally:
        old_hdulist.close()
        new_hdulist.close()

    # Second pass: compare the byte ranges between the NaN-containing arrays

    old_file = open(oldpath, encoding='latin-1')
    new_file = open(newpath, encoding='latin-1')
    try:            # no matter what happens, be sure to close the FITS files

        # Test each un-compared byte range
        for (start, stop) in uncompared:
            old_file.seek(start)
            new_file.seek(start)

            size = stop - start
            if old_file.read(size) != new_file.read(size):
                return False

    # No matter what, close the FITS files
    finally:
        old_file.close()
        new_file.close()

    # If every test has passed, the files match
    return True

##########################################################################################
