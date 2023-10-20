#! /usr/local/bin/python3

import sys
import os
import shutil
import numpy as np
from astropy.io import fits

# Turn off annoying VerifyWarnings
import warnings
from astropy.io.fits.verify import VerifyWarning

warnings.simplefilter("ignore", category=VerifyWarning)

################################################################################

NAN_REPLACEMENT: float = 0.0


def replace_NaNs(
    filename: str, test_mode: bool = False, safe_mode: bool = False
) -> int:
    """Replace any NaNs in a FITS file. Return the number of replaced values.

    If test_mode is True, it returns the number of NaNs but does not modify the
    file.

    If safe_mode is True, it saves a copy of the original file before making any
    changes. The copied file has the same name but "-original" appeneded before
    the file extension. The file is only copied if the original file contains
    NaNs.
    """

    # Open fits file...
    if test_mode:
        hdulist = fits.open(filename, mode="readonly")
    else:
        hdulist = fits.open(filename, mode="update")

    replacement_count = 0
    try:
        # For each data array...
        for hdu in hdulist:

            if not isinstance(hdu.data, np.ndarray):
                continue  # not an array
            if hdu.data.dtype.kind != "f":
                continue  # not floats

            # Test or replace NaNs, if any...
            mask = np.isnan(hdu.data)
            nan_count = np.sum(mask)
            if nan_count and not test_mode:
                hdu.data[mask] = NAN_REPLACEMENT

            replacement_count += nan_count

    finally:

        # Copy the file if necessary
        if replacement_count and safe_mode and not test_mode:
            parts = os.path.splitext(filename)
            shutil.copyfile(filename, parts[0] + "-original" + parts[1])

        hdulist.close()

    return replacement_count


################################################################################

if __name__ == "__main__":

    args = sys.argv[1:]

    # An arg "--verbose" or "-v" will cause the program to print replacement
    # counts and filenames. Otherwise, the program acts silently
    verbose_mode = False
    for flag in ("--verbose", "-v"):
        if flag in args:
            verbose_mode = True
            args.remove(flag)

    # An arg "--test" or "-t" will cause the program to print NaN counts and
    # filenames but will not modify any files. This turns on verbose mode too.
    test_mode = False
    for flag in ("--test", "-t"):
        if flag in args:
            test_mode = True
            verbose_mode = True
            args.remove(flag)

    # An arg "--safe" or "-s" will cause the program to save a copy of the
    # original file before modifying it. The original file will have a suffix
    # "-original" appended before the extension.
    safe_mode = False
    for flag in ("--safe", "-s"):
        if flag in args:
            safe_mode = True
            args.remove(flag)

    # Process files...
    exit_status = 0  # this is set to one if any error occurs
    for filename in args:
        try:
            replacement_count = replace_NaNs(filename, test_mode, safe_mode)
            if verbose_mode:
                print("%7d  %s" % (replacement_count, filename))

        except Exception as e:
            print("Error in %s:" % filename, e)
            exit_status = 1

    sys.exit(exit_status)

################################################################################
