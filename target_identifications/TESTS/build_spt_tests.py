# type: ignore
################################################################################
# build_spt_tests.py
#
# This stand-alone application creates the file SPT_TESTS.py found in this
# directory.
#
# Usage:
#   python3 build_spt_tests.py path/to/spt/directory >SPT_TESTS.py
#
# The first argument is a path to the root of any directory tree containing fits
# files, some of which end in _spt.fits, _shm.fits, or _shf.fits.
################################################################################

import os
import sys
import re
from astropy.io import fits as pyfits

# Get the root of the FITS file source directory tree
source = sys.argv[1]

REGEX = re.compile("(.*)(hst_\d{4,5}/)(.*\.fits)", re.I)

# Create a list of all the file paths
paths = []
for root, _, basenames in os.walk(source):

    for basename in basenames:
        test = basename[-9:].lower()
        if test not in ("_spt.fits", "_shm.fits", "_shf.fits"):
            continue

        paths.append(os.path.join(root, basename))

# Sort the file paths, because the ordering can be somewhat random
paths.sort()

# This is a mechanism to ensure that the file only contains unique target
# definitions
all_tuples = set()

# Begin the list of dictionaries
print("SPT_TESTS = [")

for path in paths:

    # Attempt to shorten the path in the output
    match = REGEX.fullmatch(path)
    if match:
        short_path = match.group(2) + match.group(3)
    else:
        short_path = path

    # Get the FITS header
    hdulist = pyfits.open(path)
    header = hdulist[0].header

    # Gather the dictionary info and also create a unique tuple
    new_dict = {}
    new_tuple = ()
    for key in header:

        # This is the dictionary of target info
        if (
            key.startswith("MT_LV")
            or key.startswith("TARKEY")
            or key == "TARGNAME"
            or key == "TARG_ID"
            or key == "PROPOSID"
        ):
            new_dict[key] = header[key]

        # This is a unique tuple containing target info, independent of
        # proposal. A tuple is a static structure that can be added to a set.
        if key.startswith("MT_LV") or key.startswith("TARKEY") or key == "TARGNAME":
            new_tuple += (key, header[key])

    hdulist.close()

    # Omit target definitions already written
    if new_tuple in all_tuples:
        continue

    # SPT_TESTS is a list of tuples (spt_file_path, spt_dictionary)
    print(f'  ("{short_path}",', "{")

    for (key, value) in new_dict.items():
        if isinstance(value, str):
            value = value.strip().replace('"', "")
            print(f'    "{key}": "{value}",')
        else:
            print(f'    "{key}": {value},')

    print("  }),")

    # Make sure this dictionary is not repated
    all_tuples.add(new_tuple)

# End the list of dictionaries
print("]")

################################################################################
