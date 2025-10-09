##########################################################################################
# product_labels/__init__.py
##########################################################################################

import datetime
import fnmatch
import os
import pathlib
import shutil
import sys
from collections import defaultdict

import astropy.io.fits as pyfits
import pdslogger

from . import suffix_info
from .date_support            import (get_header_date,
                                      get_trl_timetags,
                                      merge_trl_timetags,
                                      get_label_retrieval_date,
                                      get_file_creation_date,
                                      set_file_timestamp)
from .hdu_data_descriptions   import fill_hdu_data_descriptions
from .hdu_dictionary_support  import fill_hdu_dictionary, repair_hdu_dictionaries
from .hst_dictionary_support  import fill_hst_dictionary
from .get_time_coordinates    import get_time_coordinates, get_bintable_time_coordinates
from .nan_support             import get_nan_hdus, cmp_ignoring_nans
from .wavelength_ranges       import wavelength_ranges
from .xml_support             import get_modification_history, get_target_identifications

from target_identifications import hst_target_identifications
from pdstemplate import PdsTemplate

LABEL_SUFFIX = '.xml'
DEBUG_DESCRIPTIONS = False

this_dir = os.path.split(suffix_info.__file__)[0]
TEMPLATE_PATH = this_dir + '/../templates/PRODUCT_LABEL.xml'

# From https://archive.stsci.edu/hlsp/ipppssoot.html, valid last chars of the IPPPSSOOT
STANDARD_TRANSMISSION_TAILS = {'b', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'}

# This needs to be updated the program is revised
LABEL_VERSION = '1.0'
timetag = max(os.path.getmtime(__file__), os.path.getmtime(TEMPLATE_PATH))
LABEL_DATE = datetime.datetime.fromtimestamp(timetag).strftime("%Y-%m-%d")

def label_hst_fits_directories(directories, root='', *,
                               match_pattern = '',
                               old_directories = [],
                               old_root = '',
                               retrieval_date = '',
                               logger = None,
                               reset_dates = True):
    """Process one or more directories of HST FITS files, returning the information needed
    for all of their PDS4 labels as a dictionary keyed by the basenames.

    Input:
        directories         directory path or a list of directory paths.
        root                optional path to prepend to each directory path.
        match_pattern       optional fnmatch pattern to use to filter the files processed.
        old_directories     directory path containing the previous versions of the same
                            FITS files.
        old_root            optional path to prepend to each old directory path.
        retrieval_date      date the file was retrieved from MAST, in yyyy-mm-dd format.
                            If blank, the date will be retrieved from a pre-existing
                            label, or else it will be set to the file's creation date.
        logger              pdslogger to use; None for default EasyLogger.
        reset_dates         True to reset the modification date of each file to the date
                            found in the FITS header.
    """

    filepaths = get_filepaths(directories, root, match_pattern)

    if old_directories:
        old_filepaths = get_filepaths(old_directories, old_root, match_pattern)
    else:
        old_filepaths = []

    label_hst_fits_filepaths(filepaths, root,
                             old_filepaths = old_filepaths,
                             old_root = old_root,
                             retrieval_date = '',
                             logger = logger,
                             reset_dates = reset_dates)

############################################

def label_hst_fits_filepaths(filepaths, root='', *,
                             old_filepaths = [],
                             old_root = '',
                             retrieval_date = '',
                             logger = None,
                             reset_dates = True):
    """Process a list of filepaths, returning the information needed for all of their
    PDS4 labels as a dictionary keyed by the basenames.

    Input:
        filepaths           a list of file paths.
        root                an optional directory path to prepend to each file path.
        old_filepaths       a list of file paths to old versions of these files.
        old_root            optional path to prepend to each old file path.
        retrieval_date      date the file was retrieved from MAST, in yyyy-mm-dd format.
                            If blank, the date will be inferred from the file itself.
        logger              pdslogger to use; None for default EasyLogger.
        reset_dates         True to reset the modification date of each file to the date
                            found in the FITS header.
    """

    if logger is None:
        logger = pdslogger.EasyLogger()
        logger.set_format(blanklines=False)

    TEMPLATE = PdsTemplate(TEMPLATE_PATH)
    PdsTemplate.set_logger(logger)

    # Find a common root if one was not provided
    # This is not strictly necessary but creates cleaner logs by suppressing the overall
    # common directory path
    if not root:
        try:
            min_filepath = min(filepaths)
            max_filepath = max(filepaths)
        except ValueError as e:
            logger.error(str(e) + ','.join(filepaths))
            raise e

        for k, chars in enumerate(zip(min_filepath, max_filepath)):
            if chars[0] != chars[1]:
                break

        for j in range(k-1, -1, -1):
            if min_filepath[j] == '/':
                break

        root = min_filepath[:j+1]

    if root:
        logger.info('Root of file paths: ' + root)
        logger.replace_root(root)

    PdsTemplate.set_logger(logger)

    # Make sure the retrieval date, if any, is valid
    if retrieval_date:
        try:
            _ = datetime.date.fromisoformat(retrieval_date)
        except ValueError:
            logger.exception(ValueError)
            sys.exit(1)

    # Create a mapping from basename to old filepath
    old_fullpath_vs_basename = {os.path.basename(f):os.path.join(old_root, f)
                                for f in old_filepaths}

    # Save basic info about each FITS file
    info_by_basename = {}           # info vs. basename
    associations_by_ipppssoot = {}  # association list keyed by IPPPSSOOT
    # Example: associations_by_ipppssoot['odtf010f0'] = [('odtf01e4q', 'SCIENCE'),
    #                                                    ('odtf01e5q', 'GO-WAVECAL')]
    trl_timetags_by_ipppssoot = {}  # time tag dictionary, which maps date to date-time

    prev_instrument_id = ''
    accepted_suffixes = set()
    for filepath in filepaths:

        # Ignore files that are not FITS
        if not filepath.endswith('.fits'):
            continue

        logger.info('Reading', filepath)

        fullpath = os.path.join(root, filepath)
        basename = os.path.basename(fullpath)
        ipppssoot_plus_suffix = basename.partition('.')[0]
        (ipppssoot, _, suffix) = ipppssoot_plus_suffix.partition('_')
        suffix = suffix.lower()

        # Determine the instrument
        instrument_id = suffix_info.INSTRUMENT_FROM_LETTER_CODE[basename[0]]

        if instrument_id != prev_instrument_id:
            logger.info('Instrument identified', instrument_id)
            accepted_suffixes = suffix_info.ACCEPTED_SUFFIXES[instrument_id]
            prev_instrument_id = instrument_id

        # If this suffix is not accepted, skip it
        if suffix not in accepted_suffixes:
            logger.warn(f'Suffix {suffix} rejected', filepath)
            continue

        ##################################################################################
        # Initialize the dictionary of file info keyed by basename, info_by_basename.
        #
        # "basename"       : basename of FITS file.
        # "filepath"       : path to FITS file as given in input.
        # "fullpath"       : full path to FITS file, including root path.
        # "ipppssoot"      : first nine letters of basename, before first underscore.
        # "ref_ipppssoot"  : IPPPSSOOT key after possible update.
        # "suffix"         : suffix following first underscore, excluding ".fits".
        # "short_suffix"   : suffix with out its "tail", e.g., "flt_a" -> "flt".
        # "suffix_tail"    : tail of the suffix, e.g., "flt_a", "_a".
        # "retrieval_date" : the creation date on the file.
        ##################################################################################

        # Identify the retrieval date
        if retrieval_date:
            file_retrieval_date = retrieval_date
        else:
            # Take it from a pre-existing label, if any
            file_retrieval_date = get_label_retrieval_date(fullpath, LABEL_SUFFIX)

        if not file_retrieval_date:
            # Otherwise, use the file creation date. This is a bit dangerous because the
            # pipeline can modify the creation dates of files, meaning this date might be
            # wrong if you run the pipeline on the same directory a second time.
            file_retrieval_date = get_file_creation_date(fullpath)[:10]

        basename_dict = {
            'basename' : basename,
            'filepath' : filepath,
            'fullpath' : fullpath,
            'ipppssoot': ipppssoot,
            'ref_ipppssoot'  : ipppssoot,
            'suffix'         : suffix,
            'short_suffix'   : suffix_info.short_suffix(suffix),
            'suffix_tail'    : suffix_info.suffix_tail(suffix),
            'collection_name': suffix_info.collection_name(suffix, instrument_id),
            'retrieval_date' : file_retrieval_date,
            'label_version'  : LABEL_VERSION,
            'label_date'     : LABEL_DATE,
        }

        if basename in info_by_basename:
            content1 = pathlib.Path(info_by_basename[basename]['fullpath']).read_bytes()
            content2 = pathlib.Path(fullpath).read_bytes()
            new_path = fullpath + '-duplicate'
            os.rename(fullpath, new_path)
            if content1 == content2:
                logger.warn('Duplicated basename, renamed', new_path)
            else:
                logger.error('Same basename, different content', new_path)
            continue

        info_by_basename[basename] = basename_dict

        ##################################################################################
        # Gather info about the prior version, if any, of this FITS file
        #
        # "previous_fullpath"   : path to previous version of FITS file, or "".
        # "version_id"          : version_id for this product as a two-integer tuple.
        # "previous_xml"        : content of old XML label as a single string.
        # "modification_history": list of modification history attributes from old label.
        # "fits_is_identical"   : True if this FITS file is identical to the old version.
        ##################################################################################

        previous_fullpath = old_fullpath_vs_basename.get(basename, '')
        if previous_fullpath:
            with open(previous_fullpath[:-5] + LABEL_SUFFIX) as f:
                xml_content = f.read()

            modification_history = get_modification_history(xml_content)
            old_version = modification_history[-1]['version_id']

            fits_is_identical = cmp_ignoring_nans(fullpath, previous_fullpath)
            if fits_is_identical:
                logger.info('Previous data is identical', filepath)
                version_id = (old_version[0], old_version[1]+1)
            else:
                logger.info('Previous data found', filepath)
                version_id = (old_version[0]+1, 0)
        else:
            version_id = (1, 0)
            xml_content = ''
            modification_history = []
            fits_is_identical = False

        # Update the dictionary
        basename_dict['previous_fullpath'   ] = previous_fullpath
        basename_dict['version_id'          ] = version_id
        basename_dict['previous_xml'        ] = xml_content
        basename_dict['modification_history'] = modification_history
        basename_dict['fits_is_identical'   ] = fits_is_identical

        ##################################################################################
        # Read fundamental info from a few specific files
        #
        # associations_by_ipppssoot[ipppssoot] = dictionary of associated IPPPSSOOTs.
        # trl_timetags_by_ipppssoot[ipppssoot] = dictionary mapping dates to date-times.
        ##################################################################################

        # Open the FITS file
        try:
            hdulist = pyfits.open(fullpath)
        except OSError as e:
            logger.error(str(e) + fullpath)
            continue

        # If this is an association file, read its contents for the IPPPSSOOT
        if suffix == 'asn':
            logger.info('Reading associations', filepath)
            associations = read_associations(hdulist, basename)
            associations_by_ipppssoot.update(associations)

        # If this is a TRL or PDQ file, get the dictionary of date-times vs. date
        if suffix in {'trl', 'pdq'}:
            logger.info('Reading dates', filepath)
            timetag_dict = get_trl_timetags(hdulist[1], filepath, logger)
            if ipppssoot in trl_timetags_by_ipppssoot:
                merge_trl_timetags(trl_timetags_by_ipppssoot[ipppssoot], timetag_dict)
            else:
                trl_timetags_by_ipppssoot[ipppssoot] = timetag_dict

        ##################################################################################
        # Read structure and content info from this file
        #
        # "hdu_dictionaries": a list of dictionaries describing the content of each HDU.
        # "internal_date"   : the internal date, if any, from the first FITS header.
        # "has_nans"        : True if any data array in the file contains NaN.
        # "hdus_with_nans"  : List of HDU indices in which the data array contains NaN.
        ##################################################################################

        # Gather the HDU structure info
        hdu_dictionaries = []
        for k, hdu in enumerate(hdulist):
            try:
                hdu_dictionaries.append(fill_hdu_dictionary(hdu, k, instrument_id,
                                                            filepath, logger))
            except (ValueError, TypeError) as e:
                logger.error('Irrecoverable error reading FITS file', filepath)
                logger.exception(e)
                break

        # Fix known errors
        repair_hdu_dictionaries(hdu_dictionaries, filepath, logger)

        # Update the dictionary
        basename_dict['hdu_dictionaries'] = hdu_dictionaries
        basename_dict['internal_date'   ] = get_header_date(hdulist)
        basename_dict['hdus_with_nans'  ] = get_nan_hdus(hdulist)
        basename_dict['has_nans'        ] = bool(basename_dict['hdus_with_nans'])

        hdulist.close()

    ######################################################################################
    # Define an alternative way to access the basename dictionaries:
    #   info_by_ipppssoot[ipppssoot][suffix] = basename_dict[basename]
    ######################################################################################

    info_by_ipppssoot = defaultdict(dict)
    for basename, basename_dict in info_by_basename.items():
        ipppssoot = basename_dict['ipppssoot']
        suffix = basename_dict['suffix']
        info_by_ipppssoot[ipppssoot][suffix] = basename_dict

    ######################################################################################
    # Report any old file path that isn't in the new retrieval, but should be
    ######################################################################################

    for basename in old_fullpath_vs_basename:
        ipppssoot_plus_suffix = basename.rpartition('.')[0]
        (ipppssoot, _, suffix) = ipppssoot_plus_suffix.partition('_')
        if ipppssoot in info_by_ipppssoot:
            if suffix not in info_by_ipppssoot[ipppssoot]:
                logger.error(f'Old filename {basename} is missing from new retrieval',
                             old_fullpath_vs_basename[basename])

    ######################################################################################
    # Annoyingly, some files match only by IPPPSSOO, not by IPPPSSOOT. Specifically, for
    # files such as jif, jit, cmh, cmi, and cmj, the character before the underscore is
    # "j". Some details are here:
    #   https://archive.stsci.edu/hlsp/ipppssoot.html
    # where the options for the final character are "b", "m", "n", "o", "p", "q", "r",
    # "s", are "t". (No mention of "j".)
    ######################################################################################

    # Organize IPPPSSOOTS by IPPPSSOO
    ipppssoots_from_ipppssoo = defaultdict(list)
    for ipppssoot, ipppssoot_dict in info_by_ipppssoot.items():
        if ipppssoot[-1].isnumeric():       # ignore associated files ending in a digit
            continue

        ipppssoots_from_ipppssoo[ipppssoot[:-1]].append(ipppssoot)

    # Also include the IPPPSSOOTs from the associations
    for associations in associations_by_ipppssoot.values():
        for association in associations:
            ipppssoot = association[0]
            ipppssoots_from_ipppssoo[ipppssoot[:-1]].append(ipppssoot)

    # For each IPPPSSOO...
    for ipppssoo, ipppssoots in ipppssoots_from_ipppssoo.items():

        # If there's only one transmission character, there's nothing to do
        if len(ipppssoots) == 1:
            continue

        # Identify standard and non-standard transmission characters
        tails = {ipppssoot[-1] for ipppssoot in ipppssoots}
        standard_tails = STANDARD_TRANSMISSION_TAILS & tails
        nonstandard_tails = tails - standard_tails

        standard_tails = list(standard_tails)
        standard_tails.sort()

        nonstandard_tails = list(nonstandard_tails)
        nonstandard_tails.sort()

        # Select the reference transmission character, favoring standard values
        tails = standard_tails + nonstandard_tails
        reference_tail = tails[0]

        # Log an error if something happened that we don't understand
        if len(standard_tails) == 0:
            logger.warn('No standard transmission characters found for ' +
                        f'"{ipppssoo}": {nonstandard_tails}')
        elif len(standard_tails) > 1:
            logger.warn('Multiple standard transmission characters found for ' +
                        f'"{ipppssoo}": {standard_tails}')

            # Better to select the most common IPPPSSOOT
            tuples = []
            for tail in standard_tails:
                files_with_ipppssoot = len(info_by_ipppssoot[ipppssoo + tail])
                tuples.append((-files_with_ipppssoot, tail))

            tuples.sort()
            reference_tail = tuples[0][1]

        # Move each file under the reference IPPPSSOOT
        reference_ipppssoot = ipppssoo + reference_tail
        reference_dict = info_by_ipppssoot[reference_ipppssoot]
        for tail in tails:
            if tail == reference_tail:
                continue

            ipppssoot = ipppssoo + tail
            for suffix, basename_dict in info_by_ipppssoot[ipppssoot].items():
                if suffix in reference_dict:
                    logger.error('Duplicated files with the same IPPPSSOO and suffix [1]',
                                 reference_dict[suffix]['filepath'])
                    logger.error('Duplicated files with the same IPPPSSOO and suffix [2]',
                                 basename_dict['filepath'])
                    continue

                basename_dict['ref_ipppssoot'] = reference_ipppssoot
                reference_dict[suffix] = basename_dict
                logger.info('Moved to standardized IPPPSSOOT ' + reference_ipppssoot,
                            basename_dict['filepath'])

            del info_by_ipppssoot[ipppssoot]

            # Move association info if necessary
            if ipppssoot in associations_by_ipppssoot:
                associates = associations_by_ipppssoot[ipppssoot]
                if reference_ipppssoot in associations_by_ipppssoot:
                    associations_by_ipppssoot[reference_ipppssoot] += associates
                else:
                    associations_by_ipppssoot[reference_ipppssoot] = associates
                del associations_by_ipppssoot[ipppssoot]

            # Move time tag info if necessary
            if ipppssoot in trl_timetags_by_ipppssoot:
                trl_dict = trl_timetags_by_ipppssoot[ipppssoot]
                if reference_ipppssoot in trl_timetags_by_ipppssoot:
                    trl_dict.update(trl_timetags_by_ipppssoot[reference_ipppssoot])
                trl_timetags_by_ipppssoot[reference_ipppssoot] = trl_dict
                del trl_timetags_by_ipppssoot[ipppssoot]

    ######################################################################################
    # On rare occasions involving programs with multiple instruments, the IPPPSSOOT of an
    # "_asn" file can end in "0" but all the other files end in another digit.
    ######################################################################################

    solo_ipppssoots = []
    for ipppssoot, ipppssoot_dict in info_by_ipppssoot.items():

        suffixes = set(ipppssoot_dict.keys())
        if suffixes != {'asn'}:
            continue

        # This IPPPSSOOT only has an ASN file
        matches_found = 0
        for k in range(0, 10):
            alt_ipppssoot = ipppssoot[:-1] + str(k)
            if alt_ipppssoot == ipppssoot:
                continue
            if alt_ipppssoot not in info_by_ipppssoot:
                continue
            if 'asn' in info_by_ipppssoot[alt_ipppssoot]:
                continue

            # Move the info to the correctly-numbered IPPPSSOOT
            info_by_ipppssoot[alt_ipppssoot]['asn'] = ipppssoot_dict['asn']
            ipppssoot_dict['asn']['ref_ipppssoot'] = alt_ipppssoot

            logger.info('Moved to standardized IPPPSSOOT ' + alt_ipppssoot,
                        ipppssoot + '_asn.fits')
            matches_found += 1

        if matches_found > 0:
            solo_ipppssoots.append(ipppssoot)
        if matches_found > 1:
            logger.warn('File used by multiple IPPPSSOOTs', ipppssoot + '_asn.fits')

    for ipppssoot in solo_ipppssoots:
        del info_by_ipppssoot[ipppssoot]

    ######################################################################################
    # Insert IPPPSSOOT-wide information into `info_by_ipppssoot`:
    # "ipppssoot"         : IPPPSSOOT.
    # "parent"            : IPPPSSOOT ending in digit if this is CR-SPLIT; otherwise, an
    #                       empty string.
    # "parents"           : list of parent IPPPSSOOTs, for cases in which a file (such as
    #                       a calibration exposure) is associated with multiple other
    #                       files.
    # "associates"        : list of tuples (associated ipppssoot, memtype) for this
    #                       IPPPSSOOT if the associated IPPPSSOOT was found.
    # "missing_associates": list of tuples (associated ipppssoot, memtype) for this
    #                       IPPPSSOOT if the associated IPPPSSOOT was _not_ found.
    # "timetags"          : dictionary mapping date to date-time, from TRL or PDQ file.
    # "all_suffixes"      : all the suffix keys as a set.
    # "merged_suffixes"   : all the suffix keys, including those of associates.
    ######################################################################################

    # Make sure we will have these items defined for every IPPPSSOOT
    for ipppssoot, ipppssoot_dict in info_by_ipppssoot.items():
        all_suffixes = set(ipppssoot_dict.keys())
        ipppssoot_dict['ipppssoot'] = ipppssoot
        ipppssoot_dict['parent'] = ''
        ipppssoot_dict['parents'] = []
        ipppssoot_dict['associates'] = []
        ipppssoot_dict['missing_associates'] = []
        ipppssoot_dict['timetags'] = trl_timetags_by_ipppssoot.get(ipppssoot, {})
        ipppssoot_dict['all_suffixes'] = all_suffixes

    # Insert the associations list from the ASN file
    for ipppssoot, associations in associations_by_ipppssoot.items():
        ipppssoot_dict = info_by_ipppssoot[ipppssoot]

        validated_associates = []
        missing_associates = []
        for (associate, memtype) in associations:
            if associate in info_by_ipppssoot:
                associate_dict = info_by_ipppssoot[associate]
                validated_associates.append((associate, memtype))
                logger.info(f'Associate "{memtype}" found for ' + ipppssoot, associate)
                associate_dict['parent'] = ipppssoot
                associate_dict['parents'].append(ipppssoot)

                # Merge the time tags if necessary
                if associate_dict['timetags']:
                    trl_dict = associate_dict['timetags'].copy()
                    trl_dict.update(ipppssoot_dict['timetags'])
                    ipppssoot_dict['timetags'] = trl_dict

            else:
                # Warn if an associated file is missing
                missing_associates.append((associate, memtype))
                logger.warn('Missing associate for ' + ipppssoot, associate)

        ipppssoot_dict['associates'] = validated_associates
        ipppssoot_dict['missing_associates'] = missing_associates

    # Fill in the merged suffixes
    for ipppssoot, ipppssoot_dict in info_by_ipppssoot.items():
        suffixes = ipppssoot_dict['all_suffixes'].copy()
        for (associate, memtype) in ipppssoot_dict['associates']:
            if associate in info_by_ipppssoot:
                suffixes |= info_by_ipppssoot[associate]['all_suffixes']
        ipppssoot_dict['merged_suffixes'] = suffixes

    # Warn about missing time tags
    for ipppssoot, ipppssoot_dict in info_by_ipppssoot.items():
        if ipppssoot_dict['parent']:
            continue
        if not ipppssoot_dict['timetags']:
            logger.warn(f'Missing trl timetags for {ipppssoot}')

    ######################################################################################
    # Identify the SPT/SHM/SHF file for each IPPPSSOOT.
    #
    # "spt_suffix"  : SPT/SHM/SHF suffix.
    # "spt_fullpath": Full path to the SPT file.
    ######################################################################################

    # Identify all the SPT files
    for ipppssoot, ipppssoot_dict in info_by_ipppssoot.items():
        spt_suffixes = suffix_info.SPT_SUFFIXES & ipppssoot_dict['all_suffixes']
        if spt_suffixes:
            spt_suffix = spt_suffixes.pop()
            ipppssoot_dict['spt_suffix'] = spt_suffix
            basename_dict = ipppssoot_dict[spt_suffix]
            ipppssoot_dict['spt_fullpath'] = basename_dict['fullpath']

    # If an IPPPSSOOT has no SPT, use that of one of its children
    # This is OK because associates all use the same target and optical elements
    for ipppssoot, ipppssoot_dict in info_by_ipppssoot.items():
        if 'spt_suffix' in ipppssoot_dict:
            continue
        for (associate, _) in ipppssoot_dict['associates']:
            if 'spt_suffix' in info_by_ipppssoot[associate]:
                ipppssoot_dict['spt_suffix'] = info_by_ipppssoot[associate]['spt_suffix']
                ipppssoot_dict['spt_fullpath'] = \
                                            info_by_ipppssoot[associate]['spt_fullpath']
                break

    # If there's still no SPT file, use the SPT file of the parent
    for ipppssoot, ipppssoot_dict in info_by_ipppssoot.items():
        if 'spt_suffix' in ipppssoot_dict:
            continue
        parents = ipppssoot_dict['parents']
        if parents and 'spt_suffix' in info_by_ipppssoot[parents[0]]:
            ipppssoot_dict['spt_suffix'] = info_by_ipppssoot[parents[0]]['spt_suffix']
            ipppssoot_dict['spt_fullpath'] = info_by_ipppssoot[parents[0]]['spt_fullpath']
        else:
            # Otherwise, there's no target
            ipppssoot_dict['spt_suffix'] = ''
            ipppssoot_dict['spt_fullpath'] = ''

    ######################################################################################
    # Identify the reference file for each IPPPSSOOT. This is the file that serves as the
    # primary source of HST dictionary values.
    #
    # "shared"             : True if this file is shared among IPPPSSOOTs.
    # "reference_suffix"   : first reference suffix.
    # "reference_suffixes" : set of all available reference suffixes.
    # "reference_dict"     : first or only basename dict of reference file.
    # "reference_dicts"    : list of all reference file dicts
    ######################################################################################

    def find_suffix_dicts(ipppssoot_dict, suffixes):
        suffix_dicts = []
        for suffix in suffixes:
            if suffix in ipppssoot_dict['all_suffixes']:
                suffix_dicts.append(ipppssoot_dict[suffix])
            for associate, _ in ipppssoot_dict['associates']:
                if suffix in info_by_ipppssoot[associate]['all_suffixes']:
                    suffix_dicts.append(info_by_ipppssoot[associate][suffix])
        return suffix_dicts

    for ipppssoot, ipppssoot_dict in info_by_ipppssoot.items():

        ipppssoot_dict['shared'] = len(ipppssoot_dict['parents']) > 1
        if ipppssoot_dict['parent']:
            parent = ipppssoot_dict['parent']
            ipppssoot_dicts = [ipppssoot_dict, info_by_ipppssoot[parent]]
            suffix_options = [ipppssoot_dict['merged_suffixes'],
                              info_by_ipppssoot[parent]['all_suffixes']]
        else:
            ipppssoot_dicts = [ipppssoot_dict]
            suffix_options = [ipppssoot_dict['merged_suffixes']]

        # Check the REF_SUFFIXES list
        tag = 'Reference'
        for k, suffix_option in enumerate(suffix_options):
            reference_suffixes = list(suffix_info.REF_SUFFIXES & suffix_option)
            reference_dicts = find_suffix_dicts(ipppssoot_dicts[k], reference_suffixes)
            if reference_suffixes:
                break

        # If that fails, check the ALT_REF_SUFFIXES list
        if not reference_suffixes:
            tag = 'Alternative reference'
            for k, suffix_option in enumerate(suffix_options):
                reference_suffixes = list(suffix_info.ALT_REF_SUFFIXES & suffix_option)
                reference_dicts = find_suffix_dicts(ipppssoot_dicts[k],
                                                    reference_suffixes)
                if reference_suffixes:
                    break

        ipppssoot_dict['reference_suffixes'] = set(reference_suffixes)
        ipppssoot_dict['reference_dicts'] = reference_dicts

        count = len(reference_dicts)
        if count > 1:
            for k, reference_dict in enumerate(reference_dicts):
                logger.info(f'Multiple {tag.lower()} files found for {ipppssoot} ' +
                             f'({k+1}/{count})', reference_dict['fullpath'])
            reference_suffix = reference_suffixes[0]
            reference_dict = reference_dicts[0]
        elif count == 1:
            logger.info(f'{tag} file found for ' + ipppssoot,
                        reference_dicts[0]['fullpath'])
            reference_suffix = reference_suffixes[0]
            reference_dict = reference_dicts[0]
        else:
            logger.critical('No reference file for ' + ipppssoot)
            raise ValueError('No reference file for ' + ipppssoot)

        ipppssoot_dict['reference_suffix'] = reference_suffix
        ipppssoot_dict['reference_dict'] = reference_dict

    ######################################################################################
    # Gather the metadata for each IPPPSSOOT:
    #
    # "hst_dictionary"        : HST dictionary.
    # "hst_proposal_id"       : HST proposal ID.
    # "instrument_id"         : instrument ID.
    # "instrument_name"       : full name of instrument.
    # "channel_id"            : channel ID.
    # "time_coordinates"      : tuple (start_time, stop_time)
    # "wavelength_ranges"     : list of wavelength_range values.
    # "target_identifications": list of Target_Identification tuples.
    ######################################################################################

    # Identify all reference files and generate the needed info
    for ipppssoot, ipppssoot_dict in info_by_ipppssoot.items():
        reference_dict = ipppssoot_dict['reference_dict']

        # Fill the HST dictionary
        fullpath = reference_dict['fullpath']
        ref_hdulist = pyfits.open(fullpath)
        spt_fullpath = ipppssoot_dict['spt_fullpath']
        if spt_fullpath:
            spt_hdulist = pyfits.open(spt_fullpath)
        else:
            spt_hdulist = ref_hdulist
        hst_dictionary = fill_hst_dictionary(ref_hdulist, spt_hdulist, fullpath, logger)
        instrument_id = hst_dictionary['instrument_id']
        channel_id = hst_dictionary['channel_id']

        ipppssoot_dict['hst_dictionary'] = hst_dictionary
        ipppssoot_dict['hst_proposal_id'] = hst_dictionary['hst_proposal_id']
        ipppssoot_dict['instrument_id'] = instrument_id
        ipppssoot_dict['channel_id'] = channel_id
        ipppssoot_dict['instrument_name'] = suffix_info.INSTRUMENT_NAMES[instrument_id]

        # Time coordinates
        time_coordinates = get_time_coordinates(ref_hdulist, spt_hdulist, fullpath,
                                                logger)
        ipppssoot_dict['time_coordinates'] = time_coordinates[:2]
        ipppssoot_dict['time_is_actual'] = time_coordinates[2]

        if len(ipppssoot_dict['reference_dicts']) > 1:
            min_time = time_coordinates[0]
            max_time = time_coordinates[1]
            for alt_reference_dict in ipppssoot_dict['reference_dicts']:
                if alt_reference_dict is reference_dict:
                    continue
                alt_ref_fullpath = alt_reference_dict['fullpath']
                alt_ref_hdulist = pyfits.open(alt_ref_fullpath)
                alt_time_coordinates = get_time_coordinates(alt_ref_hdulist, spt_hdulist,
                                                            alt_ref_fullpath, logger)
                alt_ref_hdulist.close()
                min_time = min(min_time, alt_time_coordinates[0])
                max_time = max(max_time, alt_time_coordinates[1])
                logger.debug(f'Time range extended for {ipppssoot}', alt_ref_fullpath)
            ipppssoot_dict['time_coordinates'] = (min_time, max_time)

        # Wavelength ranges
        try:
            ranges = wavelength_ranges(hst_dictionary['instrument_id'],
                                       hst_dictionary['detector_ids'],
                                       hst_dictionary['filter_name'])
        except (ValueError, KeyError):
            ranges = ['UNK']
            logger.error('Undetermined wavelength range for ' +
                         hst_dictionary['filter_name'], fullpath)

        ipppssoot_dict['wavelength_ranges'] = ranges

        # Target identifications
        spt_fullpath = ipppssoot_dict['spt_fullpath']
        xml_content = reference_dict['previous_xml']
        if xml_content:                 # use the old identification if available
            target_ids = get_target_identifications(xml_content)
        else:
            try:
                target_ids = hst_target_identifications(spt_hdulist[0].header,
                                                        spt_fullpath, logger)
                logger.info('Target ' + str([rec[0] for rec in target_ids]),
                            ipppssoot_dict['spt_fullpath'])
            except (ValueError, KeyError) as e:
                target_ids = [('UNK', [], 'UNK', [], 'UNK')]
                logger.error(str(e).strip('"'), ipppssoot_dict['spt_fullpath'])
            except Exception as e:
                target_ids = [('UNK', [], 'UNK', [], 'UNK')]
                logger.exception(e)

        ipppssoot_dict['target_identifications'] = target_ids

        if ref_hdulist:
            ref_hdulist.close()
        spt_hdulist.close()

    ######################################################################################
    # For each basename, fill in some basics from the IPPPSSOOT dictionary
    #
    # "hst_dictionary"        : HST dictionary.
    # "instrument_id"         : instrument ID.
    # "instrument_name"       : full name of instrument.
    # "channel_id"            : channel ID.
    # "time_coordinates"      : tuple (start_time, stop_time)
    # "wavelength_ranges"     : list of wavelength_range values.
    # "target_identifications": list of Target_Identification tuples.
    #
    # Also, fill in some suffix-based info:
    #
    # "processing_level": processing_level ("Raw", "Calibrated", etc.)
    # "collection_title": collection_title
    # "browse_info"     : suffix-based browse_info.
    # "collection_lid"  : LID for this collection.
    # "product_lid"     : LID for this product.
    # "product_lidvid"  : LIDVID for this product.
    ######################################################################################

    for basename, basename_dict in info_by_basename.items():
        ipppssoot = basename_dict['ref_ipppssoot']
        ipppssoot_dict = info_by_ipppssoot[ipppssoot]
        for key in ('hst_dictionary', 'instrument_id', 'instrument_name', 'channel_id',
                    'time_coordinates', 'wavelength_ranges', 'target_identifications'):
            basename_dict[key] = ipppssoot_dict[key]

        # Override the time coordinates for EPC files
        if basename.endswith('_epc.fits'):
            basename_dict['time_coordinates'] = get_bintable_time_coordinates(
                                                            basename_dict['fullpath'],
                                                            logger)
            basename_dict['time_is_actual'] = True

        suffix = basename_dict['suffix']
        instrument_id = basename_dict['instrument_id']
        channel_id = basename_dict['channel_id']
        processing_level = suffix_info.get_processing_level(suffix, instrument_id,
                                                            channel_id)
        collection_title_fmt = suffix_info.get_collection_title_fmt(suffix, instrument_id,
                                                                    channel_id)

        hst_dictionary = basename_dict['hst_dictionary']

        ic = instrument_id + ('/' + channel_id if channel_id != instrument_id else '')
        icp_dict = {'I': instrument_id, 'IC': ic, 'P': hst_dictionary['hst_proposal_id']}
        collection_title = collection_title_fmt.format(**icp_dict)

        basename_dict['processing_level'] = processing_level
        basename_dict['collection_title'] = collection_title

        collection_lid = ('urn:nasa:pds:hst_' + str(hst_dictionary['hst_proposal_id']) +
                          ':' + basename_dict['collection_name'])
        product_lid = (collection_lid + ':' + basename_dict['ipppssoot'] + '_' +
                       basename_dict['suffix'])
        basename_dict['collection_lid'] = collection_lid
        basename_dict['product_lid'] = product_lid

        version_id = basename_dict['version_id']
        basename_dict['product_lidvid'] = (product_lid + '::' +
                                           str(version_id[0]) + ':' + str(version_id[1]))

        browse_info = suffix_info.BROWSE_SUFFIX_INFO[instrument_id, suffix]
        basename_dict['browse_info'] = browse_info

    ######################################################################################
    # Identify the associated Internal_Reference files for each basename.
    #
    # "reference_list": set of keys (ipppssoot, suffix) that will appear in this product's
    #                   Reference_List.
    # "prior_pairs"   : set of keys (ipppssoot, suffix) whose time tags must be same as or
    #                   earlier than this file.
    ######################################################################################

    # Make sure every dictionary will have these items
    for basename, basename_dict in info_by_basename.items():
        basename_dict['reference_list'] = set()
        basename_dict['prior_pairs'] = set()

    for ipppssoot, ipppssoot_dict in info_by_ipppssoot.items():
        instrument_id = ipppssoot_dict['instrument_id']

        # Create a set of tuples (ipppssoot, suffix)
        ipppssoot_suffixes = ipppssoot_dict['all_suffixes']
        pairs = {(ipppssoot, suffix) for suffix in ipppssoot_suffixes}

        # Add to the set the (ipppssoot, suffix) pairs for every associate
        for (associate, memtype) in ipppssoot_dict['associates']:
            suffixes = info_by_ipppssoot[associate]['all_suffixes']
            pairs |= {(associate, suffix) for suffix in suffixes}

        # Organize these pairs by suffix
        pairs_by_suffix = defaultdict(set)
        for pair in pairs:
            pairs_by_suffix[pair[1]].add(pair)

        all_suffixes = set(pairs_by_suffix.keys())
        reference_suffixes = ipppssoot_dict['reference_suffixes']

        # The reference files for this IPPPSSOOT must be older than any other
        # observational file with the same IPPPSSOOT.
        for suffix in reference_suffixes:
            for pair in pairs:
                if (pair[0] == ipppssoot and pair[1] not in reference_suffixes
                        and suffix_info.is_observational(pair[1], instrument_id)):
                    ipppssoot_dict[pair[1]]['prior_pairs'] |= pairs_by_suffix[suffix]

        # The TRL and PDQ files are newer than all of the observational files except raw
        for suffix in {'trl', 'pdq'} & ipppssoot_suffixes:
            derived = {p for p in pairs
                       if suffix_info.get_processing_level(p[1], instrument_id)
                          not in ('Raw', 'Ancillary')}
            ipppssoot_dict[suffix]['prior_pairs'] |= derived

        # Every ASN file must be newer than the ipppssoot and all associates.
        if 'asn' in ipppssoot_dict:
            reference_pairs = {p for p in pairs if p[1] in reference_suffixes}
            ipppssoot_dict['asn']['prior_pairs'] |= reference_pairs

        # Insert the prior suffixes from SUFFIX_INFO
        for suffix in ipppssoot_suffixes:
            prior_suffixes = suffix_info.get_prior_suffixes(suffix, instrument_id)
            prior_suffixes &= all_suffixes      # limit to suffixes that actually exist
            prior_pairs = {p for p in pairs if p[1] in prior_suffixes}
            ipppssoot_dict[suffix]['reference_list'] |= prior_pairs
            ipppssoot_dict[suffix]['prior_pairs'] |= prior_pairs

        # The Reference_List for this IPPPSSOOT's reference files include every other file
        for suffix in reference_suffixes:
            if suffix in ipppssoot_dict['all_suffixes']:
                ipppssoot_dict[suffix]['reference_list'] |= pairs
            else:
                parent = ipppssoot_dict['parent']
                if parent and suffix in info_by_ipppssoot[parent]:
                    info_by_ipppssoot[parent][suffix]['reference_list'] |= pairs

    # Finalize the reference list
    for basename, basename_dict in info_by_basename.items():

        # Remove self
        me = (basename_dict['ref_ipppssoot'], basename_dict['suffix'])
        basename_dict['reference_list'] -= {me}

        # Remove files with the same short_suffix; they don't need to reference each other
        short_suffix = basename_dict['short_suffix']
        excluded = {pair for pair in basename_dict['reference_list']
                    if pair[1].startswith(short_suffix)}
        basename_dict['reference_list'] -= excluded

    # Make each Reference_List bi-directional
    for basename, basename_dict in info_by_basename.items():
        me = (basename_dict['ref_ipppssoot'], basename_dict['suffix'])
        for (target_ipppssoot, suffix) in basename_dict['reference_list']:
            target_dict = info_by_ipppssoot[target_ipppssoot][suffix]
            target_dict['reference_list'].add(me)

    ######################################################################################
    # "reference_basenames": sorted list of reference basenames
    # "prior_basenames"    : set of prior basenames
    ######################################################################################

    # Convert each reference_list to a sorted list of basenames
    for basename, basename_dict in info_by_basename.items():
        reference_basenames = []
        for (ipppssoot, suffix) in basename_dict['reference_list']:
            basename = info_by_ipppssoot[ipppssoot][suffix]['basename']
            reference_basenames.append(basename)
        reference_basenames.sort()
        basename_dict['reference_basenames'] = reference_basenames

        prior_basenames = set()
        for (ipppssoot, suffix) in basename_dict['prior_pairs']:
            basename = info_by_ipppssoot[ipppssoot][suffix]['basename']
            prior_basenames.add(basename)
        basename_dict['prior_basenames'] = prior_basenames

    ######################################################################################
    # Fill in each modification date for each basename; make sure it is later than its
    # internal date and also later than the modification date of any of its priors.
    #
    # "modification_date": best guess at a modification date in "yyyy-mm-ddThh:mm:ss"
    #                      format.
    ######################################################################################

    # Fill in each modification date; make sure it is no earlier than its internal date
    # and also later than the modification date of any of its priors.
    for basename, basename_dict in info_by_basename.items():
        fill_modification_date(basename_dict, info_by_ipppssoot, logger)

    # Update the modification dates on the files if necessary
    if reset_dates:
        for basename, basename_dict in info_by_basename.items():
            modification_date = basename_dict['modification_date']
            if modification_date != '0000':     # year zero means date is unknown
                fullpath = basename_dict['fullpath']
                set_file_timestamp(fullpath, modification_date)
                logger.info('Modification date set to ' + modification_date, fullpath)

    ######################################################################################
    # Fill in the description fields for all data objects, i.e.,
    #   info_by_basename[basename]["hdu_dictionaries"][k]["data"]["description"]
    #
    # Also, fill in the product tile for each basename,
    #   info_by_basename[basename]["product_title"] = product_title
    #
    # If DEBUG_DESCRIPTIONS is True, the text of descriptions will be written into the
    # log, file by file, as a series of DEBUG messages. Otherwise, a sorted list of
    # unique descriptions is written to the log as DEBUG messages.
    ######################################################################################

    descriptions = set()
    for ipppssoot, ipppssoot_dict in info_by_ipppssoot.items():
        for suffix in ipppssoot_dict['all_suffixes']:
            descriptions |= fill_hdu_data_descriptions(ipppssoot, suffix,
                                                       info_by_ipppssoot,
                                                       DEBUG_DESCRIPTIONS, logger)

    if not DEBUG_DESCRIPTIONS:
        shortened = set()
        for desc in descriptions:
            if '(' in desc:
                before = desc.partition('(')[0]
                after = desc.partition(')')[2]
                desc = before + '(...)' + after
            shortened.add(desc)
        shortened = list(shortened)
        shortened.sort()
        logger.debug('Descriptions:\n    ' + '\n    '.join(shortened))

    ######################################################################################
    # Write the new labels
    ######################################################################################

    for basename, basename_dict in info_by_basename.items():
        ref_ipppssoot = basename_dict['ref_ipppssoot']
        basename_dict['ipppssoot_dict'] = info_by_ipppssoot[ref_ipppssoot]
        basename_dict['by_basename'] = info_by_basename

        label_path = basename_dict['fullpath'].replace('.fits', LABEL_SUFFIX)
        TEMPLATE.write(basename_dict, label_path)

##########################################################################################

def get_filepaths(directories, root='', match_pattern='', extension='.fits'):
    """Generate a list of file paths for processing.

    Input:
        directories     directory path of a list of directory paths.
        root            optional path to prepend to each directory path.
        match_pattern   optional fnmatch pattern to use to filter the returned list.
    """

    # Allow directories to be a single string
    if isinstance(directories, str):
        directories = [directories]

    # Prepare to strip off the root
    if root:
        root = root.rstrip('/') + '/'
        lroot = len(root)
    else:
        lroot = 0

    # Find file paths
    filepaths = []
    for directory in directories:
        fulldir = os.path.join(root, directory)
        for local_root, dirs, files in os.walk(fulldir):

            # Apply the match pattern if provided
            if match_pattern:
                files = [f for f in files if fnmatch.fnmatch(f, match_pattern)]

            # FITS files only
            files = [file for file in files if file.endswith(extension)]

            # Apply the necessary portion of the local root
            filepaths += [os.path.join(local_root[lroot:], file) for file in files]

    filepaths.sort()
    return filepaths

##########################################################################################

# Order of columns in an association table
MEMNAME = 0
MEMTYPE = 1

def read_associations(hdulist, basename):
    """Read the specified ASN file and return the list of associated IPPPSSOOT values in
    a dictionary keyed by each combined IPPPSSnnn. Each item in the list is a tuple
    (IPPPSSOOT value, memtype), where memtype has been truncated if it contains a suffix
    (following the dash).
    """

    table = hdulist[1].data

    # Identify all the products and their keys
    products = [rec for rec in table if rec[MEMTYPE].startswith('PROD')]
    if not products:
        raise IOError(f'No product MEMTYPES found in {basename}')

    if len(products) == 1:
        product = products[0][MEMNAME].lower()
        exposures = [(rec[MEMNAME].lower(), rec[MEMTYPE]) for rec in table
                     if not rec[MEMTYPE].startswith('PROD')]
        associations = {product: exposures}

    else:
        # products should be "PROD-<key>"
        product_by_key = {rec[MEMTYPE].partition('-')[2]:rec[MEMNAME].lower()
                          for rec in products}
        if '' in product_by_key:
            raise IOError(f'Inconsistent product MEMTYPES in {basename}; ' +
                          'should all be "PROD-<key>"')

        associations_by_key = {key:list() for key in product_by_key}
        for rec in table:
            (memtype, _, key) = rec[MEMTYPE].partition('-')
            if memtype == 'PROD':
                continue

            if key not in associations_by_key:
                raise IOError(f'MEMTYPE does not match product keys in {basename}: ' +
                              rec[MEMTYPE])

            associations_by_key[key].append((rec[MEMNAME].lower(), memtype))

        associations = {}
        for key, product in product_by_key.items():
            associations[product] = associations_by_key[key]

    return associations

##########################################################################################

def fill_modification_date(basename_dict, info_by_ipppssoot, logger):
    """Fill in the modification date associated with this basename."""

    # Return if this modification date is already filled in
    if basename_dict.get('modification_date', ''):
        return

    # Determine the earliest date for this IPPPSSOOT from the TRL if any
    ipppssoot_dict = info_by_ipppssoot[basename_dict['ref_ipppssoot']]
    trl_dates = list(ipppssoot_dict['timetags'].values())
    earliest_trl_date = min(trl_dates) if trl_dates else '0000'     # year zero

    # If this file has no internal date, make sure its date is no earlier than the
    # earliest TRL date
    internal_date = basename_dict['internal_date']
    if not internal_date:
        internal_date = earliest_trl_date

    # Derived products should not be older than the earliest TRL date
    if basename_dict['processing_level'] not in ('Ancillary', 'Raw'):
        internal_date = max(internal_date, earliest_trl_date)

    # This avoids a possible infinite loop, even if blank
    basename_dict['modification_date'] = internal_date

    # Check for the modification dates of priors, recursively
    dates_found = [internal_date]
    for (prior_ipppssoot, prior_suffix) in basename_dict['prior_pairs']:
        prior_basename_dict = info_by_ipppssoot[prior_ipppssoot][prior_suffix]
        fill_modification_date(prior_basename_dict, info_by_ipppssoot, logger)
        dates_found.append(prior_basename_dict['modification_date'])

    # Select the latest date
    latest_date = max(dates_found)

    # Fill in hh:mm:ss from TRL timetag dictionary if necessary and possible
    if latest_date in ipppssoot_dict['timetags']:
        latest_date = ipppssoot_dict['timetags'][latest_date]
    elif 'T' not in latest_date:
        logger.debug(f'No hh:mm:ss for {latest_date}', basename_dict['fullpath'])

    basename_dict['modification_date'] = latest_date
    if latest_date == '0000':
        logger.warn('Modification date unknown', basename_dict['fullpath'])
    else:
        logger.info('Modification date = ' + latest_date, basename_dict['fullpath'])

##########################################################################################
