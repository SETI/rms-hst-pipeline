##########################################################################################
# This file contains all the key instrument-specific and suffix-specific information
# used by the HST pipeline to create PDS4 collections.
#
# Lookup procedure is as follows:
#   1. If (suffix, instrument_id, channel_id) is a key in the dictionary, use this value.
#   2. Otherwise, if (suffix, instrument_id) is a key in the dictionary, use this value.
#   3. Otherwise, use the value keyed by the suffix alone.
#
# The dictionary returns a namedtuple of eight values:
# [0] is_accepted: boolean True if this is a suffix that we want to download.
# [1] processing_level: the processing_level to be used in Primary_Result_Summary, one of:
#     "Raw", "Partially Processed", "Calibrated", "Derived", or "Ancillary". Here,
#     "Ancillary" indicates that this is to be described as Product_Ancillary rather than
#     as Product_Observational.
# [2] hdu_description_fmt: description format string for the first data object in the
#     product file.
# [3] associated_suffix: suffix to which the hdu_description_fmt refers if not this
#     suffix; otherwise, blank. This enables the defining of relationships between files
#     with different suffixes. It there are multiple possible associated suffixes, this is
#     a tuple of values.
# [4] product_title_fmt: title format string for the product.
# [5] collection_title_fmt: title format string for the collection.
# [6] prior_suffixes: the set of suffixes that contribute to this file and must therefore
#     appear in the Reference_List of this file. Only the suffixes that most directly
#     contribute to this file should be included.
# [7] instrument_ids: if the instrument ID is not part of the key, this is the set of
#     instrument IDs to which this information refers. Otherwise, None.
#
# The format string can contain the following keys for replacement:
#   {I}      instrument ID.
#   {IC}     instrument ID/channel ID if this instrument has multiple channels, otherwise
#            just the instrument ID.
#   {P}      HST program ID.
#   {F}      File basename.
#   {lidvid} LIDVID.
#
# The hdu_description_fmt string can also contain any of these:
#   {noun}  the word for the data object, one of "image", "data array", or "table".
#   {nouns} plural of the above.
#   {name}  name of the HDU, FITS EXTNAME + "_" + EXTVER.
#   {hdu}   index number of the HDU, starting at 0.
#   {D}     detector name, if this instrument has more than one detector.
#
# Note that the reference_suffixes sets only need to "point downward" to files of the same
# or lower level of processing. For example, a calibrated file should point to the raw,
# and a mosaic should point to the source files from which it was created. Raw files can
# have an empty list. Any bi-directional reference lists found in some XML labels will be
# derived from the uni-directional info provided here.
##########################################################################################

import re
from collections import namedtuple, defaultdict

WAIVERED_INSTRUMENTS = {'FOC', 'FOS', 'GHRS', 'HSP', 'WFPC'}
UNWAIVERED_INSTRUMENTS = {'ACS', 'COS', 'NICMOS', 'STIS', 'WFC3'}

ALL_INSTRUMENTS = WAIVERED_INSTRUMENTS | UNWAIVERED_INSTRUMENTS | {'FGS', 'WFPC2'}
# For our purposes, FGS and WFPC2 are always special cases.

# First letter of filenames and the corresponding instrument IDs
# From https://archive.stsci.edu/hlsp/ipppssoot.html
INSTRUMENT_FROM_LETTER_CODE = {
    'f': 'FGS',
    'i': 'WFC3',
    'j': 'ACS',
    'l': 'COS',
    'n': 'NICMOS',
    'o': 'STIS',
    'u': 'WFPC2',
    'v': 'HSP',
    'w': 'WFPC',
    'x': 'FOC',
    'y': 'FOS',
    'z': 'GHRS',
}

# Inverse mapping...
LETTER_CODE_FROM_INSTRUMENT = {v:k for k,v in INSTRUMENT_FROM_LETTER_CODE.items()}

INSTRUMENT_NAMES = {
    'ACS'   : 'Advanced Camera for Surveys',
    'COS'   : 'Cosmic Origins Spectrograph',
    'FGS'   : 'Fine Guidance Sensors',
    'FOC'   : 'Faint Object Camera',
    'FOS'   : 'Faint Object Spectrograph',
    'GHRS'  : 'Goddard High Resolution Spectrograph',
    'HSP'   : 'High Speed Photometer',
    'NICMOS': 'Near-Infrared Camera and Multi-Object Spectrometer',
    'STIS'  : 'Space Telescope Imaging Spectrograph',
    'WFC3'  : 'Wide Field Camera 3',
    'WFPC'  : 'Wide Field/Planetary Camera',
    'WFPC2' : 'Wide Field and Planetary Camera 2',
}

##########################################################################################
# For reference, here is the complete list of suffixes associated with the first 5000
# MAST data products associated with each instrument. Possibly useful for some unit tests.

# I added these entries because they are clearly documented and might show up in other
# query results, just not the first 5000:
#   COS: "pha", "pha_b", "rawaccum_b"
#   WFPC: "c3f"
#   WFPC2: "c2m", "x0m"

ALL_SUFFIXES = {

    'ACS'   : {'asn', 'crc', 'crj', 'crj_thumb', 'drc', 'drc_large', 'drc_medium',
               'drc_small', 'drc_thumb', 'drz', 'drz_large', 'drz_medium', 'drz_small',
               'drz_thumb', 'flc', 'flc_409e04_hlet', 'flc_63b893_hlet',
               'flc_7b5ae2_hlet', 'flc_9ec245_hlet', 'flc_dce75a_hlet', 'flc_df78d0_hlet',
               'flc_large', 'flc_medium', 'flc_small', 'flc_thumb', 'flt',
               'flt_409e04_hlet', 'flt_63b893_hlet', 'flt_7b5ae2_hlet', 'flt_9ec245_hlet',
               'flt_dce75a_hlet', 'flt_df78d0_hlet', 'flt_hlet', 'flt_large',
               'flt_medium', 'flt_small', 'flt_thumb', 'jif', 'jit', 'log', 'lrc', 'lsp',
               'raw', 'raw_large', 'raw_medium', 'raw_small', 'raw_thumb', 'sfl', 'spt',
               'trl'},

    'COS'   : {'asn', 'corrtag', 'corrtag_a', 'corrtag_b', 'counts', 'counts_a',
               'counts_b', 'flt', 'flt_a', 'flt_b', 'fltsum', 'jif', 'jit', 'jwf', 'jwt',
               'lampflash', 'log', 'pha', 'pha_a', 'pha_b', 'rawaccum', 'rawaccum_a',
               'rawaccum_b', 'rawacq', 'rawtag', 'rawtag_a', 'rawtag_b', 'spt', 'trl',
               'x1d', 'x1d_prev', 'x1d_thumb', 'x1dsum', 'x1dsum1', 'x1dsum1_prev',
               'x1dsum1_thumb', 'x1dsum2', 'x1dsum2_prev', 'x1dsum2_thumb', 'x1dsum3',
               'x1dsum3_prev', 'x1dsum3_thumb', 'x1dsum4', 'x1dsum4_prev',
               'x1dsum4_thumb', 'x1dsum_prev', 'x1dsum_thumb'},

    'FGS'   : {'a1f', 'a2f', 'a3f', 'cmh', 'cmi', 'cmj', 'dmf', 'jif', 'jit', 'pdq',
               'trl'},

    'FOC'   : {'atx', 'bdx', 'c0d', 'c0f', 'c0f_thumb', 'c0h', 'c1d', 'c1f', 'c1h', 'cbf',
               'cgf', 'cgr', 'cmh', 'cmi', 'cmj', 'ctx', 'cuf', 'd0d', 'd0f', 'd0f_thumb',
               'd0h', 'dgd', 'dgh', 'dgr', 'dqx', 'eix', 'jif', 'jit', 'ocx', 'pdq',
               'pka', 'pkx', 'pod', 'q0d', 'q0f', 'q0h', 'shd', 'shf', 'shh', 'shx',
               'trl', 'trx', 'uld', 'ulf', 'ulh', 'ulx'},

    'FOS'   : {'atx', 'bdx', 'c0d', 'c0f', 'c0f_thumb', 'c0h', 'c1d', 'c1f', 'c1h', 'c2d',
               'c2f', 'c3d', 'c3f', 'c4f', 'c5f', 'c5h', 'c6f', 'c7d', 'c7f', 'c7h',
               'c8f', 'cmh', 'cmi', 'cmj', 'cqd', 'cqf', 'cqh', 'ctx', 'd0d', 'd0f',
               'd0h', 'd1f', 'dgd', 'dgh', 'dqx', 'eix', 'jif', 'jit', 'ocx', 'pdq',
               'pkx', 'poa', 'pod', 'q0d', 'q0f', 'q0h', 'q1f', 'shd', 'shf', 'shh',
               'shx', 'trl', 'trx', 'uld', 'ulf', 'ulh', 'ulx', 'x0d', 'x0f', 'x0h',
               'xqd', 'xqf', 'xqh'},

    'GHRS'  : {'atx', 'c0f', 'c0f_thumb', 'c1f', 'c2f', 'c3f', 'c4f', 'c5f', 'cmh', 'cmi',
               'cmj', 'cqf', 'd0f', 'd1f', 'jif', 'jit', 'ocx', 'pdq', 'q0f', 'q1f',
               'shf', 'trl', 'ulf', 'x0f', 'xqf'},

    'HSP'   : {'atx', 'bdx', 'c0f', 'c1f', 'c2f', 'c3f', 'ctx', 'd0d', 'd0f', 'd1f',
               'd2f', 'd3f', 'dgd', 'dgh', 'dqx', 'eix', 'ocx', 'pdq', 'pkx', 'q0f',
               'q1f', 'q2f', 'q3f', 'shf', 'shx', 'trl', 'trx', 'ulf', 'ulx'},

    'NICMOS': {'asc', 'asn', 'cal', 'cal_thumb', 'clb', 'clf', 'epc', 'ima', 'jif', 'jit',
               'mos', 'mos_thumb', 'pdq', 'ped', 'raw', 'raw_thumb', 'rwb', 'rwf',
               'scn_applied', 'spb', 'spf', 'spr', 'spt', 'trl'},

    'STIS'  : {'asn', 'crj', 'epc', 'flt', 'jif', 'jit', 'jwf', 'jwt', 'log', 'lrc',
               'lsp', 'pdq', 'raw', 'sfl', 'spt', 'sx1', 'sx1_thumb', 'sx2', 'tag', 'trl',
               'wav', 'wsp', 'x1d', 'x1d_thumb', 'x2d'},

    'WFC3'  : {'asn', 'crc', 'crj', 'drc', 'drc_thumb', 'drz', 'drz_thumb', 'flc',
               'flc_d4477c_hlet', 'flc_e4c98e_hlet', 'flc_thumb', 'flt',
               'flt_219774_hlet', 'flt_668208_hlet', 'flt_d4477c_hlet', 'flt_e4c98e_hlet',
               'flt_hlet', 'flt_thumb', 'ima', 'ima_thumb', 'jif', 'jit', 'log', 'raw',
               'raw_thumb', 'spt', 'trl'},

    'WFPC'  : {'atx', 'bdx', 'c0f', 'c0f_thumb', 'c1f', 'c2f', 'c3f', 'cgr', 'ctx', 'd0d',
               'd0f', 'd0f_thumb', 'd0h', 'dgd', 'dgh', 'dgr', 'dqx', 'eix', 'ocx', 'pdq',
               'pkx', 'q0d', 'q0f', 'q0h', 'q1d', 'q1f', 'q1h', 'shd', 'shf', 'shh',
               'shx', 'trl', 'trx', 'ulx', 'x0d', 'x0f', 'x0h'},

    'WFPC2' : {'c0f', 'c0f_thumb', 'c0m', 'c1f', 'c1m', 'c2m', 'c3m', 'c3t', 'cgr', 'cmh',
               'cmi', 'cmj', 'd0f', 'd0f_thumb', 'd0m', 'dgr', 'drz', 'jif', 'jit', 'ocx',
               'pdq', 'q0f', 'q0m', 'q1f', 'q1m', 'shf', 'shm', 'trl', 'x0m'},
}

##########################################################################################
# This might be a useful resource: https://archive.stsci.edu/hst/manifestkeywords.html
##########################################################################################

REF_SUFFIXES = {'raw', 'd0m', 'd0f',
                'a1f', 'a2f', 'a3f',                    # FGS
                'rawtag',   'rawtag_a',   'rawtag_b',   # COS...
                'rawaccum', 'rawaccum_a', 'rawaccum_b',
                'rawacq',   'rawacq_a',   'rawacq_b'}
ALT_REF_SUFFIXES = {'mos', 'drz', 'x1dsum', 'fltsum', 'd1f'}    # d1f needed for GHRS
SPT_SUFFIXES = {'spt', 'shm', 'shf', 'dmf'}

# These are used when we only want to download files for target identification testing
TARGET_IDENTIFICATION_SUFFIXES = {'spt', 'shm', 'shf', 'dmf'}

# Needed to identify some of the known, weird MAST suffixes
SUFFIX_REGEX_TO_IGNORE = re.compile(r'.*_(hlet|thumb|small|medium|large|prev)')

# Certain COS suffixes can have suffixes!
# Table 2.1, https://hst-docs.stsci.edu/cosdhb/chapter-2-cos-data-files/2-2-cos-file-names

# A dictionary that maps any of these extended suffixes to a tuple of two values. The
# first value is the suffix that will appear in the collection name; the second is a
# string that will be appended to the IPPPSSOOT in the product's LID. Example:
#       product = "lb6a03s1q_rawtag_a.fits"
#       LID = "...:data_cos_rawtag:lb6a03s1q_a"

EXTENDED_SUFFIXES = {
    'corrtag_a' : ('corrtag', '_a'),
    'corrtag_b' : ('corrtag', '_b'),
    'counts_a'  : ('counts', '_a'),
    'counts_b'  : ('counts', '_b'),
    'flt_a'     : ('flt', '_a'),
    'flt_b'     : ('flt', '_b'),
    'pha_a'     : ('pha', '_a'),
    'pha_b'     : ('pha', '_b'),
    'rawaccum_a': ('rawaccum', '_a'),
    'rawaccum_b': ('rawaccum', '_b'),
    'rawtag_a'  : ('rawtag', '_a'),
    'rawtag_b'  : ('rawtag', '_b'),
    'x1dsum1'   : ('x1dsum', '_1'),
    'x1dsum2'   : ('x1dsum', '_2'),
    'x1dsum3'   : ('x1dsum', '_3'),
    'x1dsum4'   : ('x1dsum', '_4'),
}

# When defining associated products, we exclude files with a conflicting suffix
EXTENDED_SUFFIX_EXCLUSIONS = {
    '_a':   {'_b'},
    '_b':   {'_a'},
    '_1':   {'_2','_3','_4'},
    '_2':   {'_1','_3','_4'},
    '_3':   {'_1','_2','_4'},
    '_4':   {'_1','_2','_3'},
}

SuffixInfo = namedtuple('SuffixInfo', ['is_accepted',
                                       'processing_level',
                                       'hdu_description_fmt',
                                       'associated_suffix',
                                       'product_title_fmt',
                                       'collection_title_fmt',
                                       'prior_suffixes',
                                       'instrument_ids'])

SUFFIX_INFO = {

    # Support file suffixes shared by multiple instruments

    'asn': (
        True, 'Ancillary',
        'Association table identifying the {IC} products that are part of this observation set.', '',
        'Association table identifying the products that are part of this {IC} observation set in HST Program {P}.',
        '{I} "_asn" association tables describing the observation sets of HST Program {P}.',
        set(),
        {'ACS', 'COS', 'NICMOS', 'STIS', 'WFC3'},
                # see https://hst-docs.stsci.edu/hstdhb/2-hst-file-names/2-4-associations
    ),
    'cmh': (
        True, 'Ancillary',
        'Observatory Monitoring System header information for this {IC} observation.', '',
        'Observatory Monitoring System header information file for this {IC} observation in HST Program {P}.',
        '{I} "_cmh" Observatory Monitoring System header information files for HST Program {P}.',
        set(),
        {'FGS', 'FOC', 'FOS', 'GHRS', 'WFPC2'},
    ),
    'cmi': (
        True, 'Ancillary',
        'Three-second average pointing data for this {IC} observation.', '',
        'Three-second average pointing data file for this {IC} observation in HST Program {P}.',
        '{I} "_cmi" three-second average pointing data files for HST Program {P}.',
        set(),
        {'FGS', 'FOC', 'FOS', 'GHRS', 'WFPC2'},
    ),
    'cmj': (
        True, 'Ancillary',
        'Fine time resolution pointing data for this {IC} observation.', '',
        'Fine time resolution pointing data file for this {IC} observation in HST Program {P}.',
        '{I} "_cmj" fine time resolution pointing data files for HST Program {P}.',
        set(),
        {'FGS', 'FOC', 'FOS', 'GHRS', 'WFPC2'},
    ),
    'dmf': (
        True, 'Ancillary',
        'Standard header packet containing observation parameters for this {IC} observation.', '',
        'Standard header packet file, containing observation parameters, for this {IC} observation from HST Program {P}.',
        '{I} "_dmf" standard header packet files, containing observation parameters for HST Program {P}.',
        set(),
        {'FGS', 'FOS'},
    ),
    'epc': (
        True, 'Ancillary',
        'CCD temperature table for this {IC} observation.', '',
        'CCD temperature table for this {IC} observation in HST Program {P}.',
        '{I} "_epc" CCD temperature tables for HST Program {P}.',
        set(),
        {'NICMOS', 'STIS'},
    ),
    'jif': (
        True, 'Ancillary',
        '{I} 2-D jitter histogram image for the {name} {noun}, HDU[{hdu}] of the associated "_jit" file ({F}, {lidvid}).', 'jit',
        '{I} 2-D jitter histogram image for the associated "_jit" file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_jif" 2-D jitter histogram image files for HST Program {P}.',
        {'jit'},
        ALL_INSTRUMENTS - {'HSP', 'WFPC'},
    ),
    'jit': (
        True, 'Ancillary',
        '{I} pointing jitter table for this {IC} observation.', '',
        '{I} pointing jitter table for this {IC} observation from HST Program {P}.',
        '{I} "_jit" pointing jitter tables for HST Program {P}.',
        set(),
        ALL_INSTRUMENTS - {'HSP', 'WFPC'},
    ),
    'log': (
        False, 'Ancillary',     # Removed; this just duplicates the TRL content as text
        'Processing log for this {IC} observation.', '',
        'Processing log file for this {IC} observation from HST Program {P}.',
        '{I} "_log" processing log text files for HST Program {P}.',
        set(),
        {'ACS', 'COS', 'STIS', 'WFC3'},
    ),
    'lrc': (
        True, 'Ancillary',
        'Local rate check {noun} for this {IC} observation.', '',
        'Local rate check file for this {IC} observation from HST Program {P}.',
        '{I} "_lrc" local rate check files for HST Program {P}.',
        set(),
        {'ACS', 'STIS', 'WFPC'},
    ),
    'lsp': (
        True, 'Ancillary',
        'Local rate check support data for this {IC} observation.', '',
        'Local rate check support file for this {IC} observation from HST Program {P}.',
        '{I} "_lsp" local rate check support files for HST Program {P}.',
        set(),
        {'ACS', 'STIS', 'WFPC'},
    ),
    'pdq': (
        True, 'Ancillary',
        'Post-observation summary and data quality information for this {IC} observation.', '',
        'Post-observation summary and data quality for this {IC} observation from HST Program {P}.',
        '{I} "_pdq" post-observation summary and data quality files for HST Program {P}.',
        set(),
        WAIVERED_INSTRUMENTS | {'FGS', 'NICMOS', 'STIS', 'WFPC2'},
    ),
    'ocx': (
        True, 'Ancillary',
        'Observer comments for this {IC} observation.', '',
        'Observer comments for this {IC} observation from HST Program {P}.',
        '{I} "_ocx" observer comment files for HST Program {P}.',
        set(),
        WAIVERED_INSTRUMENTS | {'WFPC2'},
    ),
    'spt': (
        True, 'Ancillary',
        'Support, planning, and telemetry data for this {IC} observation.', '',
        'Support, planning, and telemetry data for this {IC} observation from HST Program {P}.',
        '{I} "_spt" support, planning, and telemetry data for HST Program {P}.',
        set(),
        UNWAIVERED_INSTRUMENTS - {'FGS'},   # FGS files are "dmf" instead
    ),
    'shf': (
        True, 'Ancillary',
        'Standard header packet for this {IC} observation.', '',
        'Standard header packet file, containing observation parameters, for this {IC} observation from HST Program {P}.',
        '{I} "_shf" standard header packet files, containing observation parameters for HST Program {P}.',
        set(),
        WAIVERED_INSTRUMENTS,
    ),
    'trl': (
        True, 'Ancillary',
        'Processing log for this {IC} observation.', '',
        'Trailer file with processing log for this {IC} observation from HST Program {P}.',
        '{I} "_trl" trailer files with processing logs for HST Program {P}.',
        set(),
        ALL_INSTRUMENTS,
    ),

    # General suffixes for science files, unwaivered instruments

    'crc': (
        False, 'Derived',       # Removed; not normally part of a MAST delivery
        'Combined version of the calibrated, CTE-corrected {D} {nouns} {name}, HDU[{hdu}] of the associated "_flc" files in collection {C}.', 'flc',
        'Combined version of the calibrated, CTE-corrected {I} "_flc" files from HST Program {P}, collection {C}.',
        'Combined, calibrated, {I} "_crc" data files from HST Program {P}.',
        {'flc'},
        {'ACS', 'WFC3'},
    ),
    'crj': (
        False, 'Derived',       # Removed; not normally part of a MAST delivery
        'Combined version of the calibrated {D} {nouns} {name}, HDU[{hdu}] of the associated "_flt" files in collection {C}.', 'flt',
        'Combined version of the calibrated {I} "_flt" files from HST Program {P}, collection {C}.',
        'Combined, calibrated, {I} "_crj" data files from HST Program {P}.',
        {'flt'},
        {'ACS', 'STIS', 'WFC3'},
    ),
    'drc': (
        True, 'Derived',
        'Drizzle-combined version of the calibrated, CTE-corrected {D} {nouns} of the associated "_flc" files in collection {C}.', 'flc',
        'Drizzle-combined version of the associated calibrated, CTE-corrected {IC} "_flc" files from HST Program {P}, collection {C}.',
        'Drizzle-combined, calibrated, CTE-corrected {I} "_drc" data files from HST Program {P}.',
        {'flc'},
        {'ACS', 'WFC3'},
    ),
    'drz': (                    # Overridden below for ACS/WFC, WFC3/UVIS, and WFPC2
        True, 'Derived',
        'Drizzle-combined version of the calibrated {D} {nouns} of the associated "_flt" files in collection {C}.', 'flt',
        'Drizzle-combined version of the associated calibrated {IC} "_flt" files from HST Program {P}, collection {C}.',
        'Drizzle-combined, calibrated {I} "_drz" data files from HST Program {P}.',
        {'flt'},
        {'ACS', 'WFC3'},
    ),
    'flc': (
        True, 'Calibrated',
        'Calibrated, CTE-corrected version of the raw {D} {name} {noun}, HDU[{hdu}] of the associated "_raw" file ({F}, {lidvid}).', 'raw',
        'Calibrated, CTE-corrected version of the associated {IC} "_raw" file ({F}, {lidvid}) from HST Program {P}.',
        'Calibrated, CTE-corrected {I} "_flc" data files from HST Program {P}.',
        {'raw'},
        {'ACS', 'WFC3'},
    ),
    'flt': (                    # Overridden below for ACS/WFC and WFC3/UVIS
        True, 'Calibrated',
        'Calibrated version of the raw {D} {name} {noun}, HDU[{hdu}] of the associated "_raw" file ({F}, {lidvid}).', 'raw',
        'Calibrated version of the associated {IC} "_raw" file ({F}, {lidvid}) from HST Program {P}.',
        'Calibrated {I} "_flt" data files from HST Program {P}.',
        {'raw'},
        {'ACS', 'STIS', 'WFC3'},
    ),
    'raw': (
        True, 'Raw',
        'Raw {D} {noun}.', '',
        'Raw {IC} data from HST Program {P}.',
        '{I} "_raw" data files from HST Program {P}.',
        set(),
        {'ACS', 'NICMOS', 'STIS', 'WFC3'},
    ),

    # General suffixes for science files, waivered instruments (excluding WFPC2)

    'd0f': (
        True, 'Raw',
        'Raw {D} {noun}.', '',
        'Raw {IC} data from HST Program {P}.',
        'Raw {I} "_d0f" data files from HST Program {P}.',
        set(),
        WAIVERED_INSTRUMENTS - {'HSP', 'WFPC'},         # Overridden for HSP, WFPC
    ),
    'q0f': (
        True, 'Raw',
        'Data quality mask for the raw {D} {name} {noun}, HDU[{hdu}] of the associated "_d0f" file ({F}, {lidvid}).', 'd0f',
        'Data quality mask for the associated raw {IC} "_d0f" data file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_q0f" data quality masks for the raw "_d0f" data files of HST Program {P}.',
        {'d0f'},
        WAIVERED_INSTRUMENTS - {'WFPC'},                # Overridden for WFPC
    ),
    'q1f': (
        True, 'Raw',
        'Data quality mask for raw {D} engineering {name} {noun}, HDU[{hdu}] of the associated "_x0f" file ({F}, {lidvid}).', 'x0f',
        'Data quality mask for the associated raw {IC} "_x0f" engineering data file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_q1f" data quality masks for the "_x0f" engineering data files of HST Program {P}.',
        {'x0f'},
        WAIVERED_INSTRUMENTS - {'FOC', 'FOS', 'GHRS', 'WFPC'},  # Not for FOC; overridden for FOS, GHRS, WFPC
    ),
    'ulf': (
        True, 'Ancillary',
        'Unique {D} data log', '',
        '{IC} unique data log from HST Program {P}.',
        '{I} "_ulf" unique data log files from HST Program {P}.',
        {'d0f'},
        {'FOC', 'FOS', 'GHRS', 'HSP'},
    ),
    'x0f': (
        True, 'Raw',
        'Extracted engineering data for the raw {D} {name} {noun}, HDU[{hdu}] of the associated "_d0f" file ({F}, {lidvid}).', 'd0f',
        'Extracted engineering data for the associated raw {I} "_d0f" data file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_x0f" extracted engineering data for the raw "_d0f" data files from HST Program {P}.',
        {'d0f'},
        WAIVERED_INSTRUMENTS - {'FOC', 'FOS', 'HSP', 'WFPC'},   # Not for FOC; overridden for FOS, HSP, WFPC
    ),

    # Files with alternative header/data layouts can be rejected
    'c0d': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'c0h': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'c1d': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'c1h': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'c2d': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'c2h': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'c3d': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'c3h': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'c3t': (False, '', '', '', '', '', set(), {'WFPC2'}),
    'c4d': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'c4h': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'c5d': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'c5h': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'c6d': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'c6h': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'c7d': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'c7h': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'cqd': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'cqh': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'd0d': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'd0h': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'dgd': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'dgh': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'poa': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'pod': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'poh': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'q0d': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'q0h': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'q1d': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'q1h': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'shd': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'shh': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'shx': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'trx': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'uld': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'ulh': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'ulx': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'x0d': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'x0h': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'xqd': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'xqh': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),

    # I can't find any documentation about these files
    'atx': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'bdx': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'cbf': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'cgf': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'ctx': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'cuf': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'dgx': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'dqx': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'eix': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'pka': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),
    'pkx': (False, '', '', '', '', '', set(), WAIVERED_INSTRUMENTS),

    # These just contain duplicates of information found elsewhere; removed
    'cgr': (
        False, 'Ancillary',
        'Group header keyword values for the calibrated {D} {name} {noun}, HDU[{hdu}] in the associated "_c0m" file ({F}, {lidvid}).', 'c0m',
        'Group header keyword values for associated {IC} "_c0m" calibrated image file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_cgr" group header keyword values for the calibrated "_c0m" data files from HST Program {P}.',
        {'c0m'},
        {'FOC', 'WFPC', 'WFPC2'},
    ),

    'dgr': (
        False, 'Ancillary',
        'Group header keyword values for the raw {D} {name} {noun}, HDU[{hdu}] in the associated "_d0m" file ({F}, {lidvid}).', 'd0m',
        'Group header keyword values for associated {IC} "_d0m" raw image file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_dgr" group header keyword values for the raw "_d0m" data files from HST Program {P}.',
        {'d0m'},
        {'FOC', 'WFPC', 'WFPC2'},
    ),

    ######################################################################################
    # ACS
    # Table 2.1, https://hst-docs.stsci.edu/acsdhb/chapter-2-acs-data-structure/2-1-types-of-acs-files
    ######################################################################################

    ('crj', 'ACS', 'WFC'): (    # because crc is also available
        False,  'Derived',      # REMOVED! not normally part of a MAST delivery
        'Combined version of the calibrated but not CTE-corrected {D} {nouns} {name}, HDU[{hdu}] of the associated "_flt" files in collection {C}.', 'flt',
        'Combined version of the associated calibrated but not CTE-corrected {IC} "_flt" images from HST Program {P}, collection {C}.',
        'Combined, calibrated but not not CTE-corrected {I} "_crj" data files from HST Program {P}.',
        {'flt'}, None,
    ),
    ('drz', 'ACS', 'WFC'): (    # because drc is also available
        True, 'Derived',
        'Drizzle-combined version of the calibrated but not CTE-corrected {D} {nouns} of the associated "_flt" files in collection {C}.', 'flt',
        'Drizzle-combined version of the calibrated but not CTE-corrected {IC} "_flt" images from HST Program {P}, collection {C}.',
        'Drizzle-combined, calibrated but not CTE-corrected {I} "_drz" data files from HST Program {P}.',
        {'flt'}, None,
    ),
    ('flt', 'ACS', 'WFC'): (    # because flc is also available
        True, 'Derived',
        'Calibrated but not CTE-corrected version of the {D} {name} {noun}, HDU[{hdu}] of the associated "_raw" file ({F}, {lidvid}).', 'raw',
        'Calibrated but not CTE-corrected version of the associated "_raw" {IC} image file ({F}, {lidvid}) from HST Program {P}.',
        'Calibrated but not CTE-corrected {I} "_flt" data files from HST Program {P}.',
        {'raw'}, None,
    ),
    ('sfl', 'ACS'): (
        True, 'Derived',
        'Sum of the calibrated ACS/SBC MAMA {nouns} of the associated "_flt" files in collection {C}.', 'flt',
        'Sum of the associated ACS/SBC MAMA "_flt" images from HST Program {P}, collection {C}.',
        'Summed, calibrated ACS/SBC MAMA "_sfl" data files from HST Program {P}.',
        {'flt'}, None,
    ),

    ######################################################################################
    # COS
    # Table 2.1, https://hst-docs.stsci.edu/cosdhb/chapter-2-cos-data-files/2-2-cos-file-names
    ######################################################################################

    ('corrtag', 'COS'): (
        True, 'Calibrated',
        'Calibrated TIME-TAG events for the raw {D} {name} {noun}, HDU[{hdu}] of the associated file ({F}, {lidvid}).', ('rawtag', 'rawaccum', 'rawacq'),
        'Calibrated {IC} TIME-TAG events for the associated "_rawtag" file ({F}, {lidvid}) from HST Program {P}.',
        'Calibrated {I} "_corrtag" TIME-TAG events list files for HST Program {P}.',
        {'rawaccum', 'rawacq', 'rawtag'}, None,
    ),
    ('corrtag_a', 'COS'): (
        True, 'Calibrated',
        'Calibrated TIME-TAG events for the raw {D} {name} {noun}, HDU[{hdu}] of the associated file ({F}, {lidvid}).', ('rawtag_a', 'rawaccum_a', 'rawacq_a'),
        'Calibrated {IC} TIME-TAG events for the associated "_rawtag_a" file ({F}, {lidvid}) from HST Program {P}.',
        'Calibrated {I} "_corrtag" TIME-TAG events list files for HST Program {P}.',
        {'rawaccum_a', 'rawacq_a', 'rawtag_a'}, None,
    ),
    ('corrtag_b', 'COS'): (
        True, 'Calibrated',
        'Calibrated TIME-TAG events for the raw {D} {name} {noun}, HDU[{hdu}] of the associated file ({F}, {lidvid}).', ('rawtag_b', 'rawaccum_b', 'rawacq_b'),
        'Calibrated {IC} TIME-TAG events for the associated "_rawtag_b" file ({F}, {lidvid}) from HST Program {P}.',
        'Calibrated {I} "_corrtag" TIME-TAG events list files for HST Program {P}.',
        {'rawaccum_b', 'rawacq_b', 'rawtag_b'}, None,
    ),
    ('counts', 'COS'): (
        True, 'Partially Processed',
        'Intermediate {D} {noun} without flat-fielding.', '',
        'Intermediate {IC} data without flat-fielding from HST Program {P}.',
        'Intermediate {I} "_counts" data files, without flat-fielding, from HST Program {P}.',
        {'rawaccum', 'rawtag'}, None,
    ),
    ('counts_a', 'COS'): (
        True, 'Partially Processed',
        'Intermediate {D} {noun} without flat-fielding.', '',
        'Intermediate {IC} data without flat-fielding from HST Program {P}.',
        'Intermediate {I} "_counts" data files, without flat-fielding, from HST Program {P}.',
        {'rawaccum_a', 'rawtag_a'}, None,
    ),
    ('counts_b', 'COS'): (
        True, 'Partially Processed',
        'Intermediate {D} {noun} without flat-fielding.', '',
        'Intermediate {IC} data without flat-fielding from HST Program {P}.',
        'Intermediate {I} "_counts" data files, without flat-fielding, from HST Program {P}.',
        {'rawaccum_b', 'rawtag_b'}, None,
    ),
    ('flt', 'COS'): (
        True, 'Calibrated',
        'Calibrated {D} {noun}.', '',
        'Calibrated {IC} data from HST Program {P}.',
        'Calibrated {I} "_flt" data files from HST Program {P}.',
        {'rawaccum', 'rawacq', 'rawtag'}, None,
    ),
    ('flt_a', 'COS'): (
        True, 'Calibrated',
        'Calibrated {D} {noun}.', '',
        'Calibrated {IC} data from HST Program {P}.',
        'Calibrated {I} "_flt" data files from HST Program {P}.',
        {'rawaccum_a', 'rawtag_a'}, None,
    ),
    ('flt_b', 'COS'): (
        True, 'Calibrated',
        'Calibrated {D} {noun}.', '',
        'Calibrated {IC} data from HST Program {P}.',
        'Calibrated {I} "_flt" data files from HST Program {P}.',
        {'rawaccum_b', 'rawtag_a'}, None,
    ),
    ('fltsum', 'COS'): (
        True, 'Derived',
        'Sum of the flat-fielded {D} {name} {noun}, HDU[{hdu}] from the associated "_flt" files in collection {C}.', 'flt',
        'Sum of the associated "_flt" flat-fielded {IC} images for HST Program {P}, collection {C}.',
        'Summed, flat-fielded {I} "_fltsum" images for HST Program {P}.',
        {'flt'}, None,
    ),
    ('jwf', 'COS'): (
        True, 'Ancillary',
        '{I} 2-D jitter histogram image for the {name} {noun}, HDU[{hdu}] of the associated "_jwt" file ({F}, {lidvid}).', 'jwt',
        '{I} 2-D jitter histogram images for the associated "_jwt" file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_jwf" 2-D jitter image files for the wavecal observations of HST Program {P}.',
        {'jwt'}, None,
    ),
    ('jwt', 'COS'): (
        True, 'Ancillary',
        '{I} Pointing jitter table for this wavecal observation.', '',
        '{I} pointing jitter table for this wavecal observation in HST Program {P}.',
        '{I} "_jwt" pointing jitter tables for the wavecal observations of HST Program {P}.',
        {'rawtag'}, None,
    ),
    ('lampflash', 'COS'): (
        True, 'Calibrated',
        '1-D extracted TAGFLASH (FLASH=yes) {D} {noun}.', '',
        '1-D extracted TAGFLASH (FLASH=yes) {IC} spectrum from HST Program {P}.',
        '1-D extracted {I} "_lampflash" spectrum files from HST Program {P}.',
        {'rawaccum', 'rawacq', 'rawtag'}, None,
    ),
    ('pha', 'COS'): (
        True, 'Raw',
        'Raw {D} pulse height distribution {noun}.', '',
        'Raw {IC} pulse height distribution file for HST Program {P}.',
        'Raw {I} "_pha" pulse height distribution files for HST Program {P}.',
        {'rawaccum', 'rawtag'}, None,
    ),
    ('pha_a', 'COS'): (
        True, 'Raw',
        'Raw {D} pulse height distribution {noun}.', '',
        'Raw {IC} pulse height distribution file for HST Program {P}.',
        'Raw {I} "_pha" pulse height distribution files for HST Program {P}.',
        {'rawaccum_a', 'rawtag_a'}, None,
    ),
    ('pha_b', 'COS'): (
        True, 'Raw',
        'Raw {D} pulse height distribution {noun}.', '',
        'Raw {IC} pulse height distribution file for HST Program {P}.',
        'Raw {I} "_pha" pulse height distribution files for HST Program {P}.',
        {'rawaccum_b', 'rawtag_b'}, None,
    ),
    ('rawaccum', 'COS'): (
        True, 'Raw',
        'Raw {D} ACCUM {noun}.', '',
        'Raw {IC} ACCUM image for HST Program {P}.',
        '{I} "_rawaccum" data files from HST Program {P}.',
        set(), None,
    ),
    ('rawaccum_a', 'COS'): (
        True, 'Raw',
        'Raw {D} ACCUM {noun}.', '',
        'Raw {IC} ACCUM image for HST Program {P}.',
        '{I} "_rawaccum" data files from HST Program {P}.',
        set(), None,
    ),
    ('rawaccum_b', 'COS'): (
        True, 'Raw',
        'Raw {D} ACCUM {noun}.', '',
        'Raw {IC} ACCUM image for HST Program {P}.',
        '{I} "_rawaccum" data files from HST Program {P}.',
        set(), None,
    ),
    ('rawacq', 'COS'): (
        True, 'Raw',
        'Raw {D} acquisition {noun}.', '',
        'Raw {IC} acquisition file for HST Program {P}.',
        '{I} "_rawacq" acquisition files for HST Program {P}.',
        set(), None,
    ),
    ('rawtag', 'COS'): (
        True, 'Raw',
        'Raw {D} TIME-TAG {noun}.', '',
        'Raw {IC} TIME-TAG events list for HST Program {P}.',
        '{I} "_rawtag" TIME-TAG events list files for HST Program {P}.',
        set(), None,
    ),
    ('rawtag_a', 'COS'): (
        True, 'Raw',
        'Raw {D} TIME-TAG {noun}.', '',
        'Raw {IC} TIME-TAG events list for HST Program {P}.',
        '{I} "_rawtag" TIME-TAG events list files for HST Program {P}.',
        set(), None,
    ),
    ('rawtag_b', 'COS'): (
        True, 'Raw',
        'Raw {D} TIME-TAG {noun}.', '',
        'Raw {IC} TIME-TAG events list for HST Program {P}.',
        '{I} "_rawtag" TIME-TAG events list files for HST Program {P}.',
        set(), None,
    ),
    ('x1d', 'COS'): (
        True, 'Calibrated',
        '{D} 1-D extracted spectrum.', '',
        '1-D extracted {IC} spectrum file for HST Program {P}.',
        '1-D extracted {I} "_x1d" spectrum files for HST Program {P}.',
        {'rawaccum', 'rawtag'}, None,
    ),
    ('x1dsum', 'COS'): (
        True, 'Derived',
        'Combined version of the extracted 1-D {D} spectra {name}, HDU[{hdu}] of the associated "_x1d" files in collection {C}.', 'x1d',
        'Combined, extracted 1-D {IC} version of the associated "_x1d" spectrum files for HST Program {P}, collection {C}.',
        'Combined, extracted 1-D {I} "_x1dsum" spectrum files for HST Program {P}.',
        {'x1d'}, None,
    ),
    ('x1dsum1', 'COS'): (
        True, 'Derived',
        'Combined version of the extracted 1-D {D} spectra {name}, HDU[{hdu}] of the associated "_x1d" files in collection {C}.', 'x1d',
        'Combined, extracted 1-D {IC} version of the associated "_x1d" spectrum files for HST Program {P}, collection {C}.',
        'Combined, extracted 1-D {I} "_x1dsum" spectrum files for HST Program {P}.',
        {'x1d'}, None,
    ),
    ('x1dsum2', 'COS'): (
        True, 'Derived',
        'Combined version of the extracted 1-D {D} spectra {name}, HDU[{hdu}] of the associated "_x1d" files in collection {C}.', 'x1d',
        'Combined, extracted 1-D {IC} version of the associated "_x1d" spectrum files for HST Program {P}, collection {C}.',
        'Combined, extracted 1-D {I} "_x1dsum" spectrum files for HST Program {P}.',
        {'x1d'}, None,
    ),
    ('x1dsum3', 'COS'): (
        True, 'Derived',
        'Combined version of the extracted 1-D {D} spectra {name}, HDU[{hdu}] of the associated "_x1d" files in collection {C}.', 'x1d',
        'Combined, extracted 1-D {IC} version of the associated "_x1d" spectrum files for HST Program {P}, collection {C}.',
        'Combined, extracted 1-D {I} "_x1dsum" spectrum files for HST Program {P}.',
        {'x1d'}, None,
    ),
    ('x1dsum4', 'COS'): (
        True, 'Derived',
        'Combined version of the extracted 1-D {D} spectra {name}, HDU[{hdu}] of the associated "_x1d" files in collection {C}.', 'x1d',
        'Combined, extracted 1-D {IC} version of the associated "_x1d" spectrum files for HST Program {P}, collection {C}.',
        'Combined, extracted 1-D {I} "_x1dsum" spectrum files for HST Program {P}.',
        {'x1d'}, None,
    ),

    ######################################################################################
    # FGS
    ######################################################################################

    ('a1f', 'FGS'): (
        True, 'Raw',
        'Raw {D} data array.', '',
        'Raw {IC} data from HST Program {P}.',
        'Raw FGS1 "_a1f" data files from HST Program {P}.',
        set(), None,
    ),
    ('a2f', 'FGS'): (
        True, 'Raw',
        'Raw {D} data array.', '',
        'Raw {IC} data from HST Program {P}.',
        'Raw FGS2 "_a2f" data files from HST Program {P}.',
        set(), None,
    ),
    ('a3f', 'FGS'): (
        True, 'Raw',
        'Raw {D} data array.', '',
        'Raw {IC} data from HST Program {P}.',
        'Raw FGS3 "_a3f" data files from HST Program {P}.',
        set(), None,
    ),

    ######################################################################################
    # FOC
    # From Table 5.1, page 5-2
    # https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/legacy/_documents/handbooks/FOC-data-handbook.pdf
    ######################################################################################

    ('c0f', 'FOC'): (
        True, 'Partially Processed',
        'Dezoomed, geometrically corrected, uncalibrated version of the raw {name} {noun}, HDU[{hdu}] in the associated "_d0f" file ({F}, {lidvid}).', 'd0f',
        'Dezoomed, geometrically corrected, uncalibrated version of the associated raw {IC} "_d0f" file ({F}, {lidvid}) from HST Program {P}.',
        'Dezoomed, geometrically corrected, uncalibrated {I} "_c0f" files from HST Program {P}.',
        {'d0f'}, None,
    ),
    ('c1f', 'FOC'): (
        True, 'Calibrated',
        'Calibrated version of the dezoomed, geometrically corrected {name} {noun}, HDU[{hdu}] in the associated "_c0f" file ({F}, {lidvid}).', 'c0f',
        'Dezoomed, geometrically corrected, calibrated version of the associated {IC} "_c0f" file ({F}, {lidvid}) from HST Program {P}.',
        'Dezoomed, geometrically corrected, calibrated {I} "_c1f" data files from HST Program {P}.',
        {'c0f'}, None,
    ),

    ######################################################################################
    # FOS
    # Table 30.1, p. 30-2,
    # https://www.stsci.edu/files/live/sites/www/files/home/hst/documentation/_documents/fos/FOS-DHB-1998.pdf
    ######################################################################################

    ('c0f', 'FOS'): (
        True, 'Calibrated',
        'Calibrated {D} wavelength {noun}.', '',
        '{IC} calibrated wavelength file from HST Program {P}.',
        '{I} "_c0f" calibrated wavelength files from HST Program {P}.',
        {'d0f'}, None,
    ),
    ('c1f', 'FOS'): (
        True, 'Calibrated',
        'Calibrated {D} flux data.', '',
        '{IC} calibrated flux data file from HST Program {P}.',
        '{I} "_c1f" calibrated flux data files from HST Program {P}.',
        {'d0f'}, None,
    ),
    ('c2f', 'FOS'): (
        True, 'Calibrated',
        '{D} dropagated statistical error {noun}.', '',
        '{IC} propagated statistical error file from HST Program {P}.',
        '{I} "_c2f" propagated statistical error files from HST Program {P}.',
        {'c0f', 'c1f'}, None,
    ),
    ('c3f', 'FOS'): (
        True, 'Calibrated',
        '{D} special statistics {noun}.', '',
        '{IC} special statistics file for HST Program {P}.',
        '{I} "_c3f" special statistics files for HST Program {P}.',
        {'c0f', 'c1f'}, None,
    ),
    ('c4f', 'FOS'): (
        True, 'Partially Processed',
        '{D} count rate {noun}.', '',
        '{IC} count rate file from HST Program {P}.',
        '{I} "_c4f" count rate files from HST Program {P}.',
        {'c0f', 'c1f'}, None,
    ),
    ('c5f', 'FOS'): (
        True, 'Partially Processed',
        'Flat-fielded object spectrum.', '',
        '{I} flat-fielded object spectra from HST Program {P}.',
        '{I} "_c5f" flat-fielded object spectra files from HST Program {P}.',
        {'c0f', 'c1f'}, None,
    ),
    ('c6f', 'FOS'): (
        True, 'Partially Processed',
        '{D} flat-fielded sky spectrum.', '',
        '{IC} flat-fielded sky spectrum from HST Program {P}.',
        '{I} "_c6f" flat-fielded sky spectra files file from HST Program {P}.',
        {'c0f', 'c1f'}, None,
    ),
    ('c7f', 'FOS'): (
        True, 'Partially Processed',
        '{D} background spectrum.', '',
        '{IC} background spectrum from HST Program {P}.',
        '{I} "_c7f" background spectra files file from HST Program {P}.',
        {'c0f', 'c1f'}, None,
    ),
    ('c8f', 'FOS'): (
        True, 'Partially Processed',
        '{D} flat-fielded object minus smoothed sky spectrum.', '',
        '{IC} flat-fielded object minus smoothed sky spectrum from HST Program {P}.',
        '{I} "_c8f" flat-fielded object minus smoothed sky spectra files from HST Program {P}.',
        {'c0f', 'c1f'}, None,
    ),
    ('cqf', 'FOS'): (
        True, 'Calibrated',
        'Data quality mask for the {D} calibrated {name} {noun}, HDU[{hdu}] in the associated "_c1f" file ({F}, {lidvid}).', 'c1f',
        'Data quality mask for the associated {IC} "_c1f" calibrated data file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_cqf" data quality masks for the "_c1f" calibrated data of HST Program {P}.',
        {'c1f'}, None,
    ),
    ('d1f', 'FOS'): (
        True, 'Ancillary',
        '{D} trailer line {noun}.', '',
        '{IC} trailer line file from HST Program {P}.',
        '{I} "_d1f" trailer line files from HST Program {P}.',
        {'d0f'}, None,
    ),
    ('q1f', 'FOS'): (
        True, 'Ancillary',
        'Data quality mask for the {D} trailer line {name} {noun}, HDU[{hdu}] in the associated "_d1f" file ({F}, {lidvid}).', 'd1f',
        'Data quality mask for the associated {IC} "_d1f" trailer line file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_q1f" data quality masks for the "_d1f" trailer line files of HST Program {P}.',
        {'d1f'}, None,
    ),
    ('x0f', 'FOS'): (
        True, 'Ancillary',
        '{D} header line {noun}.', '',
        '{IC} header line file from HST Program {P}.',
        '{I} "_x0f" header line files from HST Program {P}.',
        {'d0f'}, None,
    ),
    ('xqf', 'FOS'): (
        True, 'Ancillary',
        'Data quality mask for the {D} header line {name} {noun}, HDU[{hdu}] in the associated "_x0f" file ({F}, {lidvid}).', 'x0f',
        'Data quality mask for the associated {IC} "_x0f" header line file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_xqf" data quality masks for the "_x0f" header line files of HST Program {P}.',
        {'x0f'}, None,
    ),

    ######################################################################################
    # GHRS
    # Table 35.1, p. 35-2,
    # https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/legacy/_documents/handbooks/GHRS-data-handbook.pdf
    ######################################################################################

    ('c0f', 'GHRS'): (
        True, 'Calibrated',
        'Calibrated {D} wavelength solution {noun}.', '',
        'Calibrated {IC} wavelength solution for HST Program {P}.',
        'Calibrated {I} "_c0f" wavelength solution files for HST Program {P}.',
        {'d0f'}, None,
    ),
    ('c1f', 'GHRS'): (
        True, 'Calibrated',
        'Calibrated {D} flux {noun}.', '',
        'Calibrated {IC} flux data file from HST Program {P}.',
        '{I} "_c1f" calibrated flux data files from HST Program {P}.',
        {'d0f'}, None,
    ),
    ('c2f', 'GHRS'): (
        True, 'Calibrated',
        '{D} propagated statistical error {noun}.', '',
        '{IC} propagated statistical error file from HST Program {P}.',
        '{I} "_c2f" propagated statistical error files from HST Program {P}.',
        {'c0f', 'c1f'}, None,
    ),
    ('c3f', 'GHRS'): (
        True, 'Calibrated',
        'Calibrated {D} special diodes {noun}.', '',
        'Calibrated {I} special diodes file from HST Program {P}.',
        'Calibrated {I} "_c3f" special diodes files from HST Program {P}.',
        {'c0f', 'c1f'}, None,
    ),
    ('c4f', 'GHRS'): (
        True, 'Calibrated',
        'Data quality mask for the calibrated {D} special diodes {name} {noun}, HDU[{hdu}] in the associated "_c3f" file ({F}, {lidvid}).', 'c3f',
        'Data quality mask for the associated {IC} "_c3f" special diodes file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_c4f" data quality masks for the "_c3f" special diodes files of HST Program {P}.',
        {'c3f'}, None,
    ),
    ('c5f', 'GHRS'): (
        True, 'Calibrated',
        '{D} background {noun}.', '',
        '{IC} background data file from HST Program {P}.',
        '{I} "_c5f" background data files from HST Program {P}.',
        {'c0f', 'c1f'}, None,
    ),
    ('cqf', 'GHRS'): (
        True, 'Calibrated',
        'Data quality mask for the calibrated {D} {name} {noun}, HDU[{hdu}] in the associated "_c0f" file ({F}, {lidvid}).', 'c0f',
        'Data quality mask for the associated calibrated {IC} "_c0f" file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_cqf" data quality masks for the calibrated "_c0f" data files of HST Program {P}.',
        {'c0f', 'c1f'}, None,
    ),
    ('d1f', 'GHRS'): (
        True, 'Raw',
        '{D} return-to-brightness and SSA ACQ/PEAKUP {noun}.', '',
        '{IC} return-to-brightness and SSA ACQ/PEAKUP file from HST Program {P}.',
        '{I} "_d1f" return-to-brightness and SSA ACQ/PEAKUP files from HST Program {P}.',
        {'d0f'}, None,
    ),
    ('q1f', 'GHRS'): (
        True, 'Raw',
        'Data quality mask for the {D} {name} {noun}, HDU[{hdu}] in the associated "_d1f" file ({F}, {lidvid}).', 'd1f',
        'Data quality mask for the associated {IC} "_d1f" file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_q1f" data quality masks for the "_d1f" files of HST Program {P}.',
        {'d1f'}, None,
    ),
    ('xqf', 'GHRS'): (
        True, 'Raw',
        'Data quality mask for the {D} engineering {name} {noun}, HDU[{hdu}] in the associated "_x0f" file ({F}, {lidvid}).', 'x0f',
        'Data quality mask for the associated {IC} "_x0f" engineering file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_xqf" data quality masks for the "_x0f" engineering data files of HST Program {P}.',
        {'x0f'}, None,
    ),

    ######################################################################################
    # HSP
    # Table 40.2, p.40-2 of
    # https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/legacy/_documents/handbooks/HSP_data_handbook.pdf
    ######################################################################################

    ('c0f', 'HSP'): (
        True, 'Calibrated',
        'Calibrated version of the raw {D} digital star {noun}, HDU[{hdu}] in the associated "_d0f" file ({F}, {lidvid}).', 'd0f',
        'Calibrated {IC} digital star data for the associated raw "_d0f" file ({F}, {lidvid}) from HST Program {P}.',
        'Calibrated {I} "_c0f" digital star data from HST Program {P}.',
        {'d0f'}, None,
    ),
    ('c1f', 'HSP'): (
        True, 'Calibrated',
        'Calibrated version of the raw {D} digital sky {noun}, HDU[{hdu}] in the associated "_d1f" file ({F}, {lidvid}).', 'd1f',
        'Calibrated {IC} digital sky data for the associated raw "_d1f" file ({F}, {lidvid}) from HST Program {P}.',
        'Calibrated {I} "_c1f" digital sky data from HST Program {P}.',
        {'d1f'}, None,
    ),
    ('c2f', 'HSP'): (
        True, 'Calibrated',
        'Calibrated version of the raw {D} analog star {noun}, HDU[{hdu}] in the associated "_d2f" file ({F}, {lidvid}).', 'd2f',
        'Calibrated {IC} analog star data for the associated raw "_d2f" file ({F}, {lidvid}) from HST Program {P}.',
        'Calibrated {I} "_c2f" analog star data from HST Program {P}.',
        {'d2f'}, None,
    ),
    ('c3f', 'HSP'): (
        True, 'Calibrated',
        'Calibrated version of the raw {D} analog sky {noun}, HDU[{hdu}] in the associated "_d3f" file ({F}, {lidvid}).', 'd3f',
        'Calibrated {IC} analog sky data for the associated raw "_d3f" file ({F}, {lidvid}) from HST Program {P}.',
        'Calibrated {I} "_c3f" analog sky data from HST Program {P}.',
        {'d3f'}, None,
    ),
    ('d0f', 'HSP'): (
        True, 'Raw',
        'Raw digital {D} star {noun}.', '',
        'Raw {IC} digital star data from HST Program {P}.',
        'Raw {I} "_d0f" digital star data files from HST Program {P}.',
        {'d1f', 'd2f', 'd3f'}, None,
    ),
    ('d1f', 'HSP'): (
        True, 'Raw',
        'Raw digital {D} sky {noun}.', '',
        'Raw {IC} digital sky data from HST Program {P}.',
        'Raw {I} "_d1f" digital sky data files from HST Program {P}.',
        set(), None,
    ),
    ('d2f', 'HSP'): (
        True, 'Raw',
        'Raw analog {D} star {noun}.', '',
        'Raw {IC} analog star data from HST Program {P}.',
        'Raw {I} "_d2f" analog star data files from HST Program {P}.',
        set(), None,
    ),
    ('d3f', 'HSP'): (
        True, 'Raw',
        'Raw analog {D} sky {noun}.', '',
        'Raw {IC} analog sky data from HST Program {P}.',
        'Raw {I} "_d3f" analog sky data files from HST Program {P}.',
        set(), None,
    ),
    ('q0f', 'HSP'): (
        True, 'Raw',
        'Data quality mask for the raw {D} digital star {noun}, HDU[{hdu}] in the associated "_d0f" file ({F}, {lidvid}).', 'd0f',
        'Data quality mask for the associated {IC} raw "_d0f" digital star data file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_q0f" daata quality masks for the digital star data files of HST Program {P}.',
        {'d0f'}, None,
    ),
    ('q1f', 'HSP'): (
        True, 'Raw',
        'Data quality mask for the raw {D} digital sky {noun}, HDU[{hdu}] in the associated "_d1f" file ({F}, {lidvid}).', 'd1f',
        'Data quality mask for the associated {IC} raw "_d1f" digital sky data file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_q1f" data quality masks for the digital sky data files of HST Program {P}.',
        {'d1f'}, None,
    ),
    ('q2f', 'HSP'): (
        True, 'Raw',
        'Data quality mask for the raw {D} analog star {noun}, HDU[{hdu}] in the associated "_d2f" file ({F}, {lidvid}).', 'd2f',
        'Data quality mask for the associated {IC} raw "_d2f" analog star data file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_q2f" data quality masks for the analog star data files of HST Program {P}.',
        {'d2f'}, None,
    ),
    ('q3f', 'HSP'): (
        True, 'Raw',
        'Data quality mask for the raw {D} analog sky {noun}, HDU[{hdu}] in the associated "_d3f" file ({F}, {lidvid}).', 'd3f',
        'Data quality mask for the associated {IC} raw "_d3f" analog sky data file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_q3f" data quality masks for the analog sky data files of HST Program {P}.',
        {'d3f'}, None,
    ),

    ######################################################################################
    # NICMOS
    # Table 2.1, p.10 of
    # https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/legacy/nicmos/_documents/nic_dhb.pdf
    # Also, https://www.cosmos.esa.int/web/hst/nicmos-file-names
    ######################################################################################

    ('ped', 'NICMOS'): (False, '', '', '', '', '', set(), None),
    ('scn_applied', 'NICMOS'): (False, '', '', '', '', '', set(), None),

    ('asc', 'NICMOS'): (
        True, 'Ancillary',
        'Post-calibration association table identifying the {D} products that are part of this observation set.', '',
        'Post-calibration {IC} association table for HST Program {P}.',
        'Post-calibration {I} "_asc" association tables for HST Program {P}.',
        {'cal'}, None,
    ),
    ('cal', 'NICMOS'): (
        True, 'Calibrated',
        'Calibrated version of the {D} {name} {noun}, HDU[{hdu}] in the associated "_raw" file ({F}, {lidvid}).', 'raw',
        'Calibrated version of the associated "_raw" {IC} data file ({F}, {lidvid}) from HST Program {P}.',
        'Calibrated {I} "_cal" image files from HST Program {P}.',
        {'raw'}, None,
    ),
    ('clb', 'NICMOS'): (
        True, 'Calibration',
        'Calibrated version of the raw {D} background {name} {noun}, HDU[{hdu}] in the associated "_rwb" file ({F}, {lidvid}).', 'rwb',
        'Calibrated version of the "_rwb" {IC} background image file ({F}, {lidvid}) from HST Program {P}.',
        'Calibrated {I} "_clb" background image files for the ACCUM observations of HST Program {P}.',
        {'rwb'}, None,
    ),
    ('clf', 'NICMOS'): (
        True, 'Calibration',
        'Calibrated version of the raw {D} flat-field {name} {noun}, HDU[{hdu}] in the associated "_rwf" file ({F}, {lidvid}).', 'rwf',
        'Calibrated version of the raw "_rwf" {IC} flat-field image file ({F}, {lidvid}) from HST Program {P}.',
        'Calibrated {I} "_clf" flat-field image files for the ACCUM observations of HST Program {P}.',
        {'rwf'}, None,
    ),
    ('epc', 'NICMOS'): (
        True, 'Ancillary',
        'CCD temperature table for this {D} observation.', '',
        'CCD temperature table for this {IC} observation in HST Program {P}.',
        '{I} "_epc" CCD temperature tables for the HST Program {P}.',
        set(), None,
    ),
    ('ima', 'NICMOS'): (
        True, 'Partially Processed',
        'Intermediate {D} MULTIACCUM {noun}.', '',
        'Intermediate {I} MULTIACCUM data file for HST Program {P}.',
        'Intermediate {I} "_ima" MULTIACCUM data files for HST Program {P}.',
        {'raw'}, None,
    ),
    ('mos', 'NICMOS'): (
        True, 'Derived',
        'Combined, calibrated {D} {noun} for the associated dithered observations.', '',
        'Combined, calibrated {I} image for dithered observations in HST Program {P}.',
        'Combined, calibrated {I} "_mos" image files for the dithered observations of HST Program {P}.',
        {'cal'}, None,
    ),
    ('raw', 'NICMOS'): (
        True, 'Raw',
        'Raw {D} {noun}.', '',
        'Raw {I} data from HST Program {P}.',
        'Raw {I} "_raw" data files from HST Program {P}.',
        set(), None,
    ),
    ('rwb', 'NICMOS'): (
        True, 'Ancillary',
        'Raw {D} background {noun} for ACCUM observations.', '',
        'Raw {I} background dats for ACCUM observations in HST Program {P}.',
        'Raw {I} "_rwb" background data files for the ACCUM observations of HST Program {P}.',
        {'raw'}, None,
    ),
    ('rwf', 'NICMOS'): (
        True, 'Ancillary',
        'Raw flat-field {noun} for ACCUM observations.', '',
        'Raw {I} flat-field data for ACCUM observations in HST Program {P}.',
        'Raw {I} "_rwf" flat-field data files for the ACCUM observations of HST Program {P}.',
        {'raw'}, None,
    ),
    ('saa', 'NICMOS'): (
        False, 'Ancillary',         # Removed, never turns up in MAST queries
        'Average of POST-SAA-DARK exposures.', '',
        'Average of POST-SAA-DARK {I} exposures in HST Program {P}.',
        '{I} "_saa" files of average POST-SAA-DARK exposures from HST Program {P}.',
        set(), None,
    ),
    ('spb', 'NICMOS'): (
        True, 'Ancillary',
        'SHP and UDL information for the {D} background {name} {noun}, HDU[{hdu}] in the associated "_rwb" file ({F}, {lidvid}).', 'rwb',
        '{I} background SHP and UDL information for the associated "_rwb" file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_spb" background SHP and UDL information for HST Program {P}.',
        {'rwb'}, None,
    ),
    ('spf', 'NICMOS'): (
        True, 'Ancillary',
        'SHP and UDL information for the flat-field {name} {noun}, HDU[{hdu}] in the associated "_rwf" file ({F}, {lidvid}).', 'rwf',
        '{I} flat-field SHP and UDL information for the associated "_rwf" file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_spf" flat-field SHP and UDL information for HST Program {P}.',
        {'rwf'}, None,
    ),
    ('spr', 'NICMOS'): (
        True, 'Ancillary',
        'SAA persistence model.', '',
        '{I} SAA persistence model for HST Program {P}.',
        '{I} "_spr" SAA persistence models for HST Program {P}.',
        set(), None,
    ),

    ######################################################################################
    # STIS
    # Info from Tables 2.1 and 2.2 of the STIS Data Handbook:
    # https://hst-docs.stsci.edu/stisdhb/chapter-2-stis-data-structure/2-2-types-of-stis-files
    # Also from Table C.1 of hst-data-handbook-for-cos-space-telescope-science-institute_compress.pdf
    ######################################################################################

    ('jwf', 'STIS'): (
        True, 'Ancillary',
        '{I} 2-D jitter histogram image for the {name} {noun}, HDU[{hdu}] of the associated "_jwt" file ({F}, {lidvid}).', 'jwt',
        '{I} 2-D jitter histogram image for the associated "_jwt" file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_jwf" 2-D jitter image files for the wavecal observations of HST Program {P}.',
        {'jwt', 'wav'}, None,
    ),
    ('jwt', 'STIS'): (
        True, 'Ancillary',
        'Pointing jitter table for this {D} wavecal observation.', '',
        '{I} pointing jitter table for this wavecal observation in HST Program {P}.',
        '{I} "_jwt" pointing jitter tables for the wavecal observations in HST Program {P}.',
        {'wav'}, None,
    ),
    ('lrc', 'STIS'): (
        True, 'Raw',
        'Local rate check for this {D} image.', '',
        '{IC} ;ocal rate check image for this observation in HST Program {P}.',
        '{I} "_lrc" local rate check image files for HST Program {P}.',
        {'raw'}, None,
    ),
    ('lsp', 'STIS'): (
        True, 'Ancillary',
        'Support text header for the {D} local rate check {name} {noun}, HDU[{hdu}] from the associated "_lrc" file ({F}, {lidvid}).', 'lrc',
        '{IC} local rate check support text header for the associated "_lrc" file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_lsp" local rate check support text headers for HST Program {P}.',
        {'lrc'}, None,
    ),
    ('sfl', 'STIS'): (
        True, 'Derived',
        'Sum of the calibrated {D} {nouns} {name}, HDU[{hdu}] of the associated "_flt" files in collection {C}.', 'flt',
        'Sum of the associated {I}/MAMA "_flt" calibrated data files from HST Program {P}, collection {C}.',
        'Summed, calibrated {I}/MAMA "_sfl" data files from HST Program {P}.',
        {'flt'}, None,
    ),
    ('sx1', 'STIS'): (
        True, 'Derived',
        'Sum of the extracted {D} 1-D {name} spectra, HDU[{hdu}] from the associated "_x1d" files in collection {C}.', 'x1d',
        'Sum of the associated {I} "_x1d" extracted spectrum files of HST Program {P}, collection {C}.',
        'Summed, extracted {I} "_sx1" spectrum files from HST Program {P}.',
        {'flt', 'x1d'}, None,
    ),
    ('sx2', 'STIS'): (
        True, 'Derived',
        'Sum of the {D} direct or spectral {nouns} {name}, HDU[{hdu}] in the associated "_x2d" files in collection {C}.', 'x2d',
        'Sum of the associated {IC} "_x2d" direct or spectral image files from HST Program {P}, collection {C}.',
        'Summed {I} "_sx2" direct or spectral image files from HST Program {P}.',
        {'flt', 'x2d'}, None,
    ),
    ('tag', 'STIS'): (
        True, 'Raw',
        'Raw {D} TIME-TAG event list.', '',
        'Raw {IC} TIME-TAG event list for HST Program {P}.',
        'Raw {I} "_tag" TIME-TAG event list files for HST Program {P}.',
        {'raw', 'wav'}, None,
    ),
    ('wav', 'STIS'): (
        True, 'Raw',
        '{D} wavecal exposure data.', '',
        '{IC} wavecal exposure data for HST Program {P}.',
        '{I} "_wav" wavecal exposure files for HST Program {P}.',
        {'raw', 'tag'}, None,
    ),
    ('wsp', 'STIS'): (
        True, 'Ancillary',
        'Support, planning, and telemetry information for the {D} wavecal {name} {noun}, HDU[{hdu}] in the associated "_wav" file ({F}, {lidvid}).', 'wav',
        'Support, planning, and telemetry information for the associated {IC} "_wav" wavecal file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_wsp" support, planning, and telemetry information for the associated "_wav" wavecal files in HST Program {P}.',
        {'wav'}, None,
    ),
    ('x1d', 'STIS'): (
        True, 'Calibrated',
        'Extracted 1-D spectrum {noun}.', '',
        'Extracted 1-D {IC} spectrum for HST Program {P}.',
        'Extracted 1-D {I} "_x1d" spectrum files for HST Program {P}.',
        {'raw', 'tag', 'wav'}, None,
    ),
    ('x2d', 'STIS'): (
        True, 'Calibrated',
        '2-D direct or spectral {I} image.', '',
        '2-D direct or spectral {IC} images for HST Program {P}.',
        '2-D direct or spectral {I} "_x2d" image files for HST Program {P}.',
        {'raw', 'tag', 'wav'}, None,
    ),

    ######################################################################################
    # WFC3
    # From Table 2.1, https://hst-docs.stsci.edu/wfc3dhb/chapter-2-wfc3-data-structure/2-1-types-of-wfc3-files
    # Also, https://www.cosmos.esa.int/web/hst/wfc3-file-names
    ######################################################################################

    ('crj', 'WFC3', 'UVIS'): (  # because crc is also available; removed!
        False, 'Derived',       # REMOVED! not normally part of a MAST delivery
        'Sum of the calibrated but not CTE-corrected {D} {nouns} {name}, HDU[{hdu}] in the associated "_flt" files in collection {C}.', 'flt',
        'Sum of the associated calibrated but not CTE-corrected {IC} "_flt" images from HST Program {P}, collection {C}.',
        'Sums of the calibrated but not CTE-corrected {I} "_flt" image files from HST Program {P}.',
        {'flt'}, None,
    ),
    ('drz', 'WFC3', 'UVIS'): (  # because drc is also available
        True, 'Derived',
        'Drizzle-combined calibrated but not CTE-corrected {D} {nouns} {name}, HDU[{hdu}] in the associated "_flt" files in collection {C}.', 'flt',
        'Drizzle-combined calibrated but not CTE-corrected {IC} "_flt" images from HST Program {P}, collection {C}.',
        'Drizzle-combined, calibrated but not CTE-corrected {I} "_drz" data files from HST Program {P}.',
        {'flt'}, None,
    ),
    ('flt', 'WFC3', 'UVIS'): (  # because flc is also available
        True, 'Calibrated',
        'Calibrated but not CTE-corrected version of the {D} {name} {noun}, HDU[{hdu}] in the associated "_raw" file ({F}, {lidvid}).', 'raw',
        'Calibrated but not CTE-corrected version of the associated {IC} "_raw" image file ({F}, {lidvid}) from HST Program {P}.',
        'Calibrated but not CTE-corrected {I} "_flt" data files from HST Program {P}.',
        {'raw'}, None,
    ),
    ('ima', 'WFC3'): (
        True, 'Partially Processed',
        'Calibrated intermediate version of the {D} {name} {noun}, HDU[{hdu}] in the associated "_raw" file ({F}, {lidvid}).', 'raw',
        'Calibrated intermediate version of the associated {IC} "_raw" image file ({F}, {lidvid}) from HST Program {P}.',
        'Calibrated, intermediate {I} "_ima" image files from HST Program {P}.',
        {'raw'}, None,
    ),

    ######################################################################################
    # WFPC
    # From Table 44.1, p. 44-2
    # https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/legacy/_documents/handbooks/HST_WFPC_Data_Handbook.pdf
    ######################################################################################

    ('c0f', 'WFPC'): (
        True, 'Calibrated',
        'Calibrated images, one per detector, stacked as a 3-D array. ' +
            'These are calibrated versions of the raw WFPC image stack in HDU #0 of the associated "_d0f" file ({F}, {lidvid}).', 'd0f',
        'Calibrated versions of the raw WFPC "_d0f" image file ({F}, {lidvid}) from HST Program {P}.',
        'Calibrated WFPC "_c0f" image files from HST Program {P}.',
        {'d0f'}, None,
    ),
    ('c1f', 'WFPC'): (
        True, 'Calibrated',
        'Image quality masks, one per detector, stacked as a 3-D array. ' +
            'These apply to the calibrated WFPC image stack in HDU #0 of the associated "_c0f" file ({F}, {lidvid}).', 'c0f',
        'Data quality masks for the associated WFPC "_c0f" image file ({F}, {lidvid}) from HST Program {P}.',
        'WFPC "_c1f" data quality masks for the calibrated "_c0f" image files of HST Program {P}.',
        {'c0f'}, None,
    ),
    ('c2f', 'WFPC'): (
        True, 'Calibrated',
        'Data histograms, one per detector, stacked as a 3-D array. ' +
            'These refer to the calibrated WFPC image stack in HDU #0 of the associated "_c0f" file ({F}, {lidvid}).', 'c0f',
        'Data histograms for the associated WFPC "_c0f" image file ({F}, {lidvid}) from HST Program {P}.',
        'WFPC "_c2f" data histograms for the calibrated "_c0f" image files of HST Program {P}.',
        {'c0f'}, None,
    ),
    ('c3f', 'WFPC'): (
        True, 'Calibrated',
        'Saturated pixel maps, one per detector, stacked as a 3-D array. ' +
            'These apply to the calibrated WFPC image stack in HDU #0 of the associated "_c0f" file ({F}, {lidvid}).', 'c0f',
        'Saturated pixel maps for the associated WFPC "_c0f" image file ({F}, {lidvid}) from HST Program {P}.',
        'WFPC "_c3f" saturated pixel maps for the calibrated "_c0f" image files of HST Program {P}.',
        {'c0f'}, None,
    ),
    ('d0f', 'WFPC'): (
        True, 'Raw',
        'Raw WFPC images, one per detector, stacked as a 3-D array.', '',
        'Raw WFPC images from HST Program {P}.',
        'Raw WFPC "_d0f" data files from HST Program {P}.',
        set(), None,
    ),
    ('q0f', 'WFPC'): (
        True, 'Raw',
        'Image quality masks, one per detector, stacked as a 3-D array. ' +
            'These apply to the raw WFPC image stack in HDU #0 of the associated "_d0f" file ({F}, {lidvid}).', 'd0f',
        'Data quality masks for the associated raw WFPC "_d0f" image file ({F}, {lidvid}) from HST Program {P}.',
        'WFPC "_q0f" data quality masks for the raw "_d0f" image files of HST Program {P}.',
        {'d0f'}, None,
    ),
    ('q1f', 'WFPC'): (
        True, 'Raw',
        'Data quality masks, one per detector, stacked as a 3-D array. ' +
            'These apply to the raw WFPC engineering data array in HDU #0 of the associated "_x0f" file ({F}, {lidvid}).', 'x0f',
        'Data quality masks for the associated raw WFPC "_x0f" engineering file ({F}, {lidvid}) from HST Program {P}.',
        'WFPC "_q1f" data quality masks for the "_x0f" engineering data files of HST Program {P}.',
        {'x0f'}, None,
    ),
    ('x0f', 'WFPC'): (
        True, 'Raw',
        'Extracted engineering data arrays, one for each detector, stacked as a 3-D array. ' +
            'These support the raw WFPC image stack in HDU #0 of the associated "_d0f" file ({F}, {lidvid}).', 'd0f',
        'Extracted engineering data for the associated raw WFPC "_d0f" image file ({F}, {lidvid}) from HST Program {P}.',
        'WFPC "_x0f" extracted engineering data for the raw "_d0f" data files from HST Program {P}.',
        {'d0f'}, None,
    ),

    ######################################################################################
    # WFPC2
    # From Table 2.1, https://www.stsci.edu/instruments/wfpc2/Wfpc2_dhb/wfpc2_ch22.html
    ######################################################################################

    # Suppress waivered files that would otherwise be downloaded for WFPC2
    ('c0f', 'WFPC2'): (False, '', '', '', '', '', set(), None),
    ('c1f', 'WFPC2'): (False, '', '', '', '', '', set(), None),
    ('c2f', 'WFPC2'): (False, '', '', '', '', '', set(), None),
    ('c3f', 'WFPC2'): (False, '', '', '', '', '', set(), None),
    ('d0f', 'WFPC2'): (False, '', '', '', '', '', set(), None),
    ('q0f', 'WFPC2'): (False, '', '', '', '', '', set(), None),
    ('q1f', 'WFPC2'): (False, '', '', '', '', '', set(), None),
    ('shf', 'WFPC2'): (False, '', '', '', '', '', set(), None),
    ('x0f', 'WFPC2'): (False, '', '', '', '', '', set(), None),

    ('c0m', 'WFPC2'): (
        True, 'Calibrated',
        'Calibrated version of the raw {D} {name} {noun}, HDU[{hdu}] in the associated "_d0m" file ({F}, {lidvid}).', 'd0m',
        'Calibrated version of the associated {I} "_d0m" image file ({F}, {lidvid}) from HST Program {P}.',
        'Calibrated {I} "_c0m" image files from HST Program {P}.',
        {'d0m'}, None,
    ),
    ('c1m', 'WFPC2'): (
        True, 'Calibrated',
        'Data quality mask for the calibrated {D} {name} {noun}, HDU[{hdu}] in the associated "_c0m" file ({F}, {lidvid}).', 'c0m',
        'Data quality masks for the associated {I} "_c0m" image file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_c1m" data quality masks for the calibrated "_c0m" image files of HST Program {P}.',
        {'c0m'}, None,
    ),
    ('c2m', 'WFPC2'): (
        True, 'Calibrated',
        'Data histogram for the calibrated {D} {name} {noun}, HDU[{hdu}] in the associated "_c0m" file ({F}, {lidvid}).', 'c0m',
        'Data histograms for the associated {I} "_c0m" image file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_c2m" data histograms for the calibrated "_c0m" image files of HST Program {P}.',
        {'c0m'}, None,
    ),
    ('c3m', 'WFPC2'): (
        True, 'Ancillary',
        'Filter/instrument throughput table for the {D} {name} {noun}, HDU[{hdu}] in the associated "_d0m" file ({F}, {lidvid}).', 'd0m',
        '{I} filter/instrument throughput table for the associated image files of HST Program {P}.',
        '{I} "_c3m" filter/instrument throughput tables for the image files of HST Program {P}.',
        {'d0m', 'c0m'}, None,
    ),
    ('drz', 'WFPC2'): (
        True, 'Derived',
        'Drizzle-combined version of the calibrated {I} {nouns} {name}, HDU[{hdu}] from the associated "_c0m" files in collection {C}.', 'c0m',
        'Drizzle-combined version of the associated "_c0m" {I} image files from HST Program {P}, collection {C}.',
        'Drizzle-combined, calibrated (I) "_drz" image files from HST Program {P}.',
        {'c0m'},
        None,
    ),
    ('d0m', 'WFPC2'): (
        True, 'Raw',
        'Raw {D} image.', '',
        'Raw {I} image file from HST Program {P}.',
        'Raw {I} "_d0m" image files from HST Program {P}.',
        set(), None,
    ),
    ('q0m', 'WFPC2'): (
        True, 'Raw',
        'Data quality mask for the raw {D} {name} {noun}, HDU[{hdu}] in the associated "_d0m" file ({F}, {lidvid}).', 'd0m',
        'Data quality masks for the associated raw {I} "_d0m" image file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_q0m" data quality masks for the raw "_d0m" image files of HST Program {P}.',
        {'d0m'}, None,
    ),
    ('q1m', 'WFPC2'): (
        True, 'Raw',
        'Data quality mask for the raw {D} engineering {name} {noun}, HDU[{hdu}] in the associated "_x0m" file ({F}, {lidvid}).', 'x0m',
        'Data quality masks for the associated raw {I} "_x0m" engineering data file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_q1m" data quality masks for the "_x0m" engineering data files of HST Program {P}.',
        {'x0m'}, None,
    ),
    ('shm', 'WFPC2'): (
        True, 'Ancillary',
        'Standard header packet containing observation parameters.', '',
        'Standard header packet file, containing observation parameters, for this {I} observation from HST Program {P}.',
        '{I} "_shm" standard header packet files containing observation parameters for HST Program {P}.',
        {'d0m'}, None,
    ),
    ('x0m', 'WFPC2'): (
        True, 'Raw',
        'Extracted engineering data for the raw {D} {name} {noun}, HDU[{hdu}] in the associated "_d0m" file ({F}, {lidvid}).', 'd0m',
        'Extracted engineering data for the associated raw {I} "_d0m" image file ({F}, {lidvid}) from HST Program {P}.',
        '{I} "_x0m" extracted engineering data for the raw "_d0m" image files of HST Program {P}.',
        {'d0m'}, None,
    ),
}

##########################################################################################
# Derived quantities
##########################################################################################

# Convert each tuple in SUFFIX_INFO into a namedtuple
# Convert keys of type str into single-valued tuples
items = list(SUFFIX_INFO.items())   # save the content so the loop below can replace keys
for key, value in items:
    if isinstance(key, str):
        del SUFFIX_INFO[key]
        key = (key,)
    SUFFIX_INFO[key] = SuffixInfo(*value)

# Get a string containing the accepted letter codes
instrument_ids = set()
for key, info in SUFFIX_INFO.items():
    if info.is_accepted:
        if len(key) > 1:
            instrument_ids.add(key[1])
        else:
            instrument_ids |= info.instrument_ids

letter_codes = [LETTER_CODE_FROM_INSTRUMENT[key] for key in instrument_ids]
letter_codes.sort()

ACCEPTED_LETTER_CODES = ''.join(letter_codes)

# Create dictionaries that return the sets of accepted suffixes and rejected suffixes for
# each instrument
ACCEPTED_SUFFIXES = defaultdict(set)
REJECTED_SUFFIXES = defaultdict(set)
for key, info in SUFFIX_INFO.items():
    if info.is_accepted:
        suffix_dict = ACCEPTED_SUFFIXES
    else:
        suffix_dict = REJECTED_SUFFIXES

    suffix = key[0]
    if len(key) == 1:
        for instrument_id in info.instrument_ids:
            suffix_dict[instrument_id].add(suffix)
    else:
        instrument_id = key[1]
        suffix_dict[instrument_id].add(suffix)

##########################################################################################
# Support for browse products
##########################################################################################

# Derived from numerous MAST queries. Given an instrument, this is the list of browse
# suffixes that should be downloaded from MAST. For the most part, we ignore the thumbnail
# browse products for purposes of PDS4 archiving. However, the thumbnail spectra are
# well done and we will want to use them in OPUS/Viewmaster, so we're including them in
# the bundles.
ACCEPTED_BROWSE_SUFFIXES = {
    'ACS'   : ['drc_large', 'drz_large', 'flc', 'flt', 'raw'],
    'COS'   : ['x1dsum', 'x1dsum1', 'x1dsum2', 'x1dsum3', 'x1dsum4'],
    'FGS'   : [],
    'FOC'   : ['d0f', 'c0f'],
    'FOS'   : ['c0f', 'c0f_thumb'],
    'GHRS'  : ['c0f', 'c0f_thumb'],
    'HSP'   : [],
    'NICMOS': ['cal', 'mos', 'raw'],
    'STIS'  : ['sx1', 'sx1_thumb', 'x1d', 'x1d_thumb'],
    'WFC3'  : ['drc', 'drz', 'flc', 'flt', 'ima', 'raw'],
    'WFPC'  : ['d0f', 'c0f'],
    'WFPC2' : ['d0f', 'c0f'],   # Note: MAST browse products use the old, waivered names!
}

# For each instrument that needs this, this is a dictionary that maps a MAST browse suffix
# to the local browse suffix, i.e. the one that we will use in the bundle. Certain browse
# products from MAST have to be renamed so that their suffixes match those of the
# corresponding FITS file.
MAST_BROWSE_SUFFIX_TRANSLATOR = {
    'ACS'   : {'drc_large': 'drc', 'drz_large': 'drz'},     # strip "_large"
    'WFPC2' : {'c0f': 'c0m', 'd0f': 'd0m'},                 # convert to unwaivered
}

# Our criterion for deciding if the pipeline should generate its own browse products is
# that the instrument ID fall within the first set and the suffix fall within the second.
# We don't have a mechanism in place for creating spectral plots, so we will always use
# the MAST browse products for those.
LOCAL_BROWSE_INSTRUMENTS = {
    'ACS', 'FOC', 'NICMOS', 'WFC3', 'WFPC', 'WFPC2',
}

LOCAL_BROWSE_SUFFIXES = {
    'c0f', 'c0m', 'cal', 'd0f', 'd0m', 'drc', 'drz', 'flt', 'flc', 'ima', 'mos', 'raw',
}

# Class definition...
BrowseProductInfo = namedtuple('BrowseProductInfo', ['suffix',
                                                     'mast_suffix',
                                                     'collection_name',
                                                     'lid_suffix',
                                                     'is_locally_generated'])
# [0] suffix: the suffix on the browse product, following IPPPSSOOT. This is normally the
#     the same as that for the FITS file, but might have "_thumb" appended.
# [1] mast_suffix: the suffix used in MAST.
# [2] collection_name: name of the collection, e.g., "browse_acs_flt".
# [3] lid_suffix: any suffix to append to the IPPPSSOOT in the LID; mainly for the "_a",
#     "_b", "_1", etc. suffixes used by COS.
# [4] is_locally_generated: True if this product is generated locally; False if retrieved
#     from MAST.

# Create a dictionary that maps (instrument_id, FITS suffix) to a list of
# BrowseProductInfo objects for the MAST-generated products.
BROWSE_SUFFIX_INFO = defaultdict(list)
for instrument_id, mast_suffixes in ACCEPTED_BROWSE_SUFFIXES.items():
    translator = MAST_BROWSE_SUFFIX_TRANSLATOR.get(instrument_id, {}) # for "_large" and
                                                                      # WFPC2
    for mast_suffix in mast_suffixes:

        # Translate MAST suffix to local suffix, e.g., "drc_large" -> "drc"
        local_suffix = translator.get(mast_suffix, mast_suffix)

        # Translate local suffix to FITS suffix
        fits_suffix = local_suffix.replace('_thumb', '')

        # Translate fits_suffix to collection name and LID suffix
        (short_suffix, lid_suffix) = EXTENDED_SUFFIXES.get(fits_suffix, (fits_suffix, ''))
        collection_name = 'browse_' + instrument_id.lower() + '_' + short_suffix

        # Create the BrowseProductInfo object and append
        info = BrowseProductInfo(local_suffix, mast_suffix, collection_name, lid_suffix,
                                 False)
        BROWSE_SUFFIX_INFO[instrument_id, fits_suffix].append(info)

# Update for the locally-generated browse products, which supersede MAST products.
# For these products, the FITS suffix and the local browse suffix are always the same
for instrument_id in LOCAL_BROWSE_INSTRUMENTS:
    accepted_suffixes = ACCEPTED_SUFFIXES[instrument_id]
    for local_suffix in LOCAL_BROWSE_SUFFIXES:

        # Skip suffixes that are not accepted for a given instrument, or unrecognized
        try:
            if local_suffix not in accepted_suffixes:
                continue
        except ValueError:
            continue

        # If there's a matching MAST product, replace it with the local version
        updated = False
        browse_info_list = BROWSE_SUFFIX_INFO[instrument_id, local_suffix]
        for k, info in enumerate(browse_info_list):
            if info.suffix == local_suffix:
                new_info = BrowseProductInfo(local_suffix, '', info.collection_name,
                                             info.lid_suffix, True)
                browse_info_list[k] = new_info
                updated = True
                break

        # Otherwise, append this product info
        if not updated:
            collection_name = 'browse_' + instrument_id.lower() + '_' + local_suffix
            new_info = BrowseProductInfo(local_suffix, '', collection_name, '', True)
            browse_info_list.append(new_info)

##########################################################################################
# API supporting suffix identifications
##########################################################################################

def is_accepted(suffix, instrument_id):
    """True if this suffix should be archived; False otherwise. Raise a ValueError if the
    suffix is not recognized.
    """

    if suffix in ACCEPTED_SUFFIXES[instrument_id]:
        return True
    if suffix in REJECTED_SUFFIXES[instrument_id]:
        return False
    if SUFFIX_REGEX_TO_IGNORE.fullmatch(suffix):
        return False
    raise ValueError(f'unrecognized suffix for {instrument_id}: "{suffix}"')

def is_rejected(suffix, instrument_id):
    """True if this suffix should be rejected; False otherwise. Raise a ValueError if the
    suffix is not recognized.
    """

    if suffix in REJECTED_SUFFIXES[instrument_id]:
        return True
    if suffix in ACCEPTED_SUFFIXES[instrument_id]:
        return False
    if SUFFIX_REGEX_TO_IGNORE.fullmatch(suffix):
        return True
    raise ValueError(f'unrecognized suffix for {instrument_id}: "{suffix}"')

def is_recognized(suffix, instrument_id):
    """True if this suffix is recognized, meaning that we know whether or not it needs to
    be archived. False if it is unrecognized.
    """

    if suffix in ACCEPTED_SUFFIXES[instrument_id]:
        return True
    if suffix in REJECTED_SUFFIXES[instrument_id]:
        return True
    if SUFFIX_REGEX_TO_IGNORE.fullmatch(suffix):
        return True

    return False

##########################################################################################
# API functions supporting access to the SUFFIX_INFO
##########################################################################################

def _suffix_info_key(suffix, instrument_id, channel_id=None):
    """The SUFFIX_INFO key based on the instrument, channel, and suffix."""

    # Try two options for the dictionary key
    for key in [(suffix, instrument_id, channel_id),
                (suffix, instrument_id)]:
        if key in SUFFIX_INFO:
            return key

    # For a solo suffix as key, make sure it applies to the this instrument
    key = (suffix,)
    info = SUFFIX_INFO.get(key, None)
    if info:
        if instrument_id in info.instrument_ids:
            return key
        else:
            raise KeyError(f'suffix "{suffix}" is not applicable to instrument ' +
                           f'{instrument_id}')

    # Otherwise, report an error
    if instrument_id is None:
        raise KeyError(f'suffix "{suffix}" not found in SUFFIX_INFO')
    elif channel_id is None:
        raise KeyError(f'Key based on suffix "{suffix}", ' +
                       f'instrument "{instrument_id}" ' +
                       'not found in SUFFIX_INFO')
    else:
        raise KeyError(f'Key based on suffix "{suffix}", ' +
                       f'instrument "{instrument_id}", ' +
                       f'channel "{channel_id}" ' +
                       'not found in SUFFIX_INFO')

def _suffix_info(suffix, instrument_id, channel_id=None):
    """The SUFFIX_INFO based on the instrument, channel, and suffix."""

    return SUFFIX_INFO[_suffix_info_key(suffix, instrument_id, channel_id)]

def get_processing_level(suffix, instrument_id, channel_id=None):
    return _suffix_info(suffix, instrument_id, channel_id).processing_level

def get_hdu_description_fmt(suffix, instrument_id, channel_id=None):
    return _suffix_info(suffix, instrument_id, channel_id).hdu_description_fmt

def get_associated_suffix(suffix, instrument_id, channel_id=None):
    return _suffix_info(suffix, instrument_id, channel_id).associated_suffix

def get_product_title_fmt(suffix, instrument_id, channel_id=None):
    return _suffix_info(suffix, instrument_id, channel_id).product_title_fmt

def get_collection_title_fmt(suffix, instrument_id, channel_id=None):
    return _suffix_info(suffix, instrument_id, channel_id).collection_title_fmt

def get_prior_suffixes(suffix, instrument_id, channel_id=None):
    return _suffix_info(suffix, instrument_id, channel_id).prior_suffixes

def is_ancillary(suffix, instrument_id, channel_id=None):
    return _suffix_info(suffix, instrument_id, channel_id).processing_level == 'Ancillary'

def is_observational(suffix, instrument_id, channel_id=None):
    return _suffix_info(suffix, instrument_id, channel_id).processing_level != 'Ancillary'

def collection_name(suffix, instrument_id):
    """The name of the collection for this instrument and suffix.
    """

    level = _suffix_info(suffix, instrument_id).processing_level
    return (('miscellaneous' if level == 'Ancillary' else 'data')
            + '_'
            + instrument_id.lower()
            + '_'
            + EXTENDED_SUFFIXES.get(suffix, (suffix, ''))[0])

def lid_suffix(suffix):
    """When a suffix has its own suffix, the latter suffix has to be appended to the
    IPPPSSOOT in the LID because it is not part of the collection name.
    """

    return EXTENDED_SUFFIXES.get(suffix, (suffix, ''))[1]

def excluded_lid_suffixes(lid_suffix):
    """When a suffix has its own suffix, the list of associated products should exclude
    those with a conflicting suffix.
    """

    return EXTENDED_SUFFIX_EXCLUSIONS.get(lid_suffix, set())

##########################################################################################

def test_recognized():
    """Every listed suffix needs to be recognized.
    """

    for instrument_id, suffixes in ALL_SUFFIXES.items():
        for suffix in suffixes:
            assert is_recognized(suffix, instrument_id)

def test_associated_suffixes():
    """Every associated suffix must be a prior suffix. It must also be referenced in the
    HDU description and the HDU description must mention it.
    """

    for info in SUFFIX_INFO.values():
        desc = info.hdu_description_fmt
        suffix = info.associated_suffix
        if isinstance(suffix, tuple):
            for suffix_ in suffix:
                assert suffix_ in info.prior_suffixes
        elif suffix:
            assert '_' + suffix in desc
            assert suffix in info.prior_suffixes
        else:
            assert not (' file' in desc and '_' in desc)

def test_collection_titles():
    """Every collection title should mention the suffix.
    """

    for key, info in SUFFIX_INFO.items():
        if info.is_accepted:
            suffix = key[0]
            short_suffix = EXTENDED_SUFFIXES.get(suffix, (suffix, ''))[0]
            assert '_' + short_suffix in info.collection_title_fmt

def test_keys():
    """Every item in SUFFIX_INDEX is used.
    """

    for key, info in SUFFIX_INFO.items():
        if info.is_accepted and len(key) > 1:
            (suffix, instrument_id) = key[:2]
            assert suffix in ALL_SUFFIXES[instrument_id]

##########################################################################################
