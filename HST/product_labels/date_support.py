##########################################################################################
# date_support.py
#
# get_header_date(header0)
#   Return the latest date string ("yyyy-mm-dd" or "yyyy-mm-ddThh:mm:ss" found in the
#   given header (which should be hdulist[0].header); otherwise, an empty string.
#
# get_trl_timetags(hdu1)
#   Return a dictionary that provides the latest date-time string associated with a
#   particular date, based on a scraping of the content of HDU #1 from a TRL file. This
#   can potentially be used to fill in the hours/minutes/seconds of a header date that is
#   lacking any time information.
#
# get_label_retrieval_date(filepath)
#   If the given data file already has a label, get the retrieval date from that label.
#   Otherwise, return an empty string.
#
# get_file_creation_date(filepath)
#   Get the creation date of a file if it is available, or the last access date otherwise.
#   This is OS-dependent.
#
# set_file_timestamp(filepath, date)
#   Set the file's modification date.
##########################################################################################

import datetime
import os
import re
import astropy.io.fits as pyfits

import julian

current_year = datetime.datetime.now().year
yyyy_since_2020 = '|'.join([str(y) for y in range(2020, current_year+1)])
yy_since_2020 = '|'.join([str(y) for y in range(20, current_year+1-2000)])

YYYY = r'(199\d|20[01]\d|' + yyyy_since_2020 + ')'  # matches 1990 to 2049
YY   = r'([901]\d|' + yy_since_2020 + ')'
MON  = r'(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)'
MM   = r'(0\d|1[0-2])'
DOY  = r'([0-2]\d\d|3[0-6]\d)'
DD   = r'([0-2]\d|3[01])'
D    = r'([0-2]\d|3[01]| [1-9]|[1-9])'

HHMMSS   = r'([0-2]\d[0-5]\d[0-6]\d)'
HH_MM    = r'([0-2]\d:[0-5]\d)'
HH_MM_SS = r'([0-2]\d:[0-5]\d:[0-6]\d)'
HH_MM_SS_FFF = r'([0-2]\d:[0-5]\d:[0-6]\d(?:\.\d*|))'

D_MM_YY_PATTERN = re.compile(f'{D}/{MM}/{YY}')
YYYY_MM_DD_PATTERN = re.compile(f'{YYYY}-{MM}-{DD}')
YYYY_MM_DD_HH_MM_SS_PATTERN = re.compile(f'{YYYY}-{MM}-{DD}T{HH_MM_SS}')

DADSDATE_PATTERN = re.compile(f'{DD}[ -]{MON}[ -]{YYYY} {HH_MM_SS}', re.I)
FITSDATE_PATTERN = re.compile(f'{D}-{MON}-{YYYY}', re.I)
IRAF_TLM_PATTERN = re.compile(f'{HH_MM_SS} \({DD}/{MM}/{YYYY}\)')
PROCTIME_PATTERN = re.compile(f'{YYYY}\.{DOY}:{HH_MM_SS}')
COMMENT_PATTERN  = re.compile(r'.*{YYYY}-{MM}-{DD}')
INFLIGHT_YMD_PATTERN = re.compile(f' *INFLIGHT {YYYY}-{MM}-{DD} {YYYY}-{MM}-{DD}')
INFLIGHT_DMY_PATTERN = re.compile(f' *INFLIGHT {DD}/{MM}/{YYYY} {DD}/{MM}/{YYYY}')

MONTHS = {
    'JAN': '01',
    'FEB': '02',
    'MAR': '03',
    'APR': '04',
    'MAY': '05',
    'JUN': '06',
    'JUL': '07',
    'AUG': '08',
    'SEP': '09',
    'OCT': '10',
    'NOV': '11',
    'DEC': '12',
}

def get_header_date(hdulist):
    """Return the most plausible file creation date found in this FITS HDUlist.

    Many files contain mutually contradictory dates; we choose the latest of these.
    """

    dates_found = []
    header0 = hdulist[0].header

    if len(hdulist) > 1:
        header1 = hdulist[1].header
    else:
        header1 = header0

    # Some files have DADSDATE with full date and time "dd-mon-yyyy hh:mm:ss"
    value = header0.get('DADSDATE', '')
    match = DADSDATE_PATTERN.fullmatch(value)
    if match:
        (dd, mon, yyyy, hh_mm_ss) = match.groups()
        mm = MONTHS[mon.upper()]
        dates_found.append(f'{yyyy}-{mm}-{dd}T{hh_mm_ss}')
    elif value:
        raise ValueError(f'unsupported DADSDATE format: "{value}"')

    # Some files have FITSDATE "dd-mon-yyyy" or "d-mon-yyyy" or "dd/mm/yy"
    value = header0.get('FITSDATE', '')
    match = FITSDATE_PATTERN.fullmatch(value)
    if match:
        (d, mon, yyyy) = match.groups()
        dd = ('0' + d if len(d) == 1 else d.replace(' ', '0'))
        mm = MONTHS[mon.upper()]
        dates_found.append(f'{yyyy}-{mm}-{dd}')
    elif match := D_MM_YY_PATTERN.fullmatch(value):
        (d, mm, yy) = match.groups()
        dd = ('0' + d if len(d) == 1 else d.replace(' ', '0'))
        yyyy = ('19' + yy if yy[0] == '9' else '20' + yy)
        dates_found.append(f'{yyyy}-{mm}-{dd}')
    elif match := YYYY_MM_DD_PATTERN.fullmatch(value):
        (yyyy, mm, dd) = match.groups()
        dates_found.append(f'{yyyy}-{mm}-{dd}')
    elif value:
        raise ValueError(f'unsupported FITSDATE format: "{value}"')

    # Some files have IRAF-TLM "hh:mm:ss (dd/mm/yyyy)
    value = header0.get('IRAF-TLM', '')
    match = IRAF_TLM_PATTERN.fullmatch(value)
    if match:
        (hh_mm_ss, dd, mm, yyyy) = match.groups()
        dates_found.append(f'{yyyy}-{mm}-{dd}T{hh_mm_ss}')
    elif value:
        raise ValueError(f'unsupported IRAF-TLM format: "{value}"')

    # Most files have DATE "yyyy-mm-dd" or "dd/mm/yy"
    value = header0.get('DATE', '')
    match = YYYY_MM_DD_PATTERN.fullmatch(value)
    if match:
        dates_found.append(value)
    elif match := D_MM_YY_PATTERN.fullmatch(value):
        (d, mm, yy) = match.groups()
        dd = ('0' + d if len(d) == 1 else d.replace(' ', '0'))
        yyyy = ('19' + yy if yy[0] == '9' else '20' + yy)
        dates_found.append(f'{yyyy}-{mm}-{dd}')
    elif match := YYYY_MM_DD_HH_MM_SS_PATTERN.fullmatch(value):
        dates_found.append(value)
    elif value:
        raise ValueError(f'unsupported DATE format: "{value}"')

    # Some files, including jif and jit, contain PROCTIME "yyyy.doy:hh:mm:ss"
    value = header1.get('PROCTIME', '')
    match = PROCTIME_PATTERN.fullmatch(value)
    if match:
        (yyyy, doy, hh_mm_ss) = match.groups()
        date = datetime.date(int(yyyy),1,1) + datetime.timedelta(int(doy) - 1)
        dates_found.append(f'{yyyy}-{date.month:02d}-{date.day:02d}T{hh_mm_ss}')

    # WFPC2/C3M files have ORIGIN with comment "Tables version 2002-02-22"
    if 'ORIGIN' in header0:
        comment = header0.comments['ORIGIN']
        match = COMMENT_PATTERN.fullmatch(comment)
        if match:
            (yyyy, mm, dd) = match.groups()
            dates_found.append(f'{yyyy}-{mm}-{dd}')

    # Last resort... Look for "INFLIGHT dd/mm/yyyy dd/mm/yyyy" in HISTORY comments
# NEVER MIND: These dates are not reliable!
#     for history in header0.get('HISTORY', []):
#         match = INFLIGHT_YMD_PATTERN.fullmatch(history)
#         if match:
#             (yyyy, mm, dd, yyyy1, mm1, dd1) = match.groups()
#             dates_found += [f'{yyyy}-{mm}-{dd}', f'{yyyy1}-{mm1}-{dd1}']
#         elif match := INFLIGHT_DMY_PATTERN.fullmatch(history):
#             (dd, mm, yyyy, dd1, mm1, yyyy1) = match.groups()
#             dates_found += [f'{yyyy}-{mm}-{dd}', f'{yyyy1}-{mm1}-{dd1}']

    if dates_found:
        return max(dates_found)
    else:
        return ''

##########################################################################################

# Each of the following patterns have been seen somewhere in a TRL file, based on
# extensive trial and error.

YYYYDOYHHMMSS           = re.compile(f'{YYYY}{DOY}{HHMMSS}.*')
D_MON_YY_HH_MM_SS       = re.compile(f'.*[^\d]{D}-{MON}-{YY} {HH_MM_SS}.*', re.I)
MON_D_HH_MM_SS_TZ_YYYY  = re.compile(f'.*{MON} {D} {HH_MM_SS} ([A-Z]+) {YYYY}.*', re.I)
D_MON_YYYY_HH_MM_SS     = re.compile(f'.*{D}-{MON}-{YYYY} {HH_MM_SS}.*', re.I)
D_MON_YY_HH_MM_SS       = re.compile(f'.*{D}-{MON}-{YY},? {HH_MM_SS}.*', re.I)
HH_MM_SS_DD_MON_YYYY    = re.compile(f'.*{HH_MM_SS} {DD}-{MON}-{YYYY}.*', re.I)
HH_MM_SS_DD_MON_YY      = re.compile(f'.*{HH_MM_SS} {DD}-{MON}-{YY}[^\d].*', re.I)
HH_MM_SS_FFF_DD_MM_YYYY = re.compile(f'.*{HH_MM_SS_FFF}.? .?{DD}/{MM}/{YYYY}.*')
YYYY_MM_DD_HH_MM_SS_FFF = re.compile(f'.*{YYYY}-{MM}-{DD}.{HH_MM_SS_FFF}.*')
HH_MM_SS_TZ_DD_MON_YYYY = re.compile(f'.*{HH_MM_SS} ([A-Z]+) {DD}-{MON}-{YYYY}.*', re.I)
MON_D_YYYY_HH_MM_SS     = re.compile(f'.*{MON} {D} {YYYY},? {HH_MM_SS}.*', re.I)
D_MM_YY_HH_MM_SS        = re.compile(f'.*{D}/{MM}/{YY} +{HH_MM_SS}.*')
D_MON_YYYY_HH_MM        = re.compile(f'.*{D}-{MON}-{YYYY} {HH_MM}.*', re.I)

# If some other weird format is encountered, it will _probably_ be interpreted correctly,
# using the patterns below, but a warning message will also be logged.

TIME_TEST   = re.compile(r'(.*[^\d])' + HH_MM_SS_FFF + r'(.*)')
MONTH_TEST  = re.compile(r'(.* )' + MON + r'(.*)', re.I)
YYYY_D_TEST = re.compile(f'.*[^\d]{YYYY}[^\d]+{D}[^\d].*')
YY_D_TEST   = re.compile(f'.*[^\d]{YY}[^\d]+{D}[^\d].*')
D_YYYY_TEST = re.compile(f'.*[^\d]{D}[^\d]+{YYYY}[^\d].*')
D_YY_TEST   = re.compile(f'.*[^\d]{D}[^\d]+{YY}[^\d].*')
D_MM_YYYY_TEST = re.compile(f'.*[^\d]{D}[^d]+{MM}[^\d]+{YYYY}[^\d].*')
D_MM_YY_TEST   = re.compile(f'.*[^\d]{D}[^d]+{MM}[^\d]+{YY}[^\d].*')
MM_D_YYYY_TEST = re.compile(f'.*[^\d]{MM}[^d]+{D}[^\d]+{YYYY}[^\d].*')
MM_D_YY_TEST   = re.compile(f'.*[^\d]{MM}[^d]+{D}[^\d]+{YY}[^\d].*')
YYYY_MM_D_TEST = re.compile(f'.*[^\d]{YYYY}[^d]+{MM}[^\d]+{D}[^\d].*')
YY_MM_D_TEST   = re.compile(f'.*[^\d]{YY}[^d]+{MM}[^\d]+{D}[^\d].*')

# These are the only explicit time zone tags I have encountered. The program raises an
# ValueError exception if a new one is encountered but not handled. The dictionary returns
# the offset in hours from UTC.
TIMEZONES = {
    'MEST': 2,
    'MET' : 1,
    'UTC' : 0,
    'GMT' : 0,
    'EST' : -5,     # I have not encountered EST or EDT in any TRL files, but we might as
    'EDT' : -4,     # well be ready for them in case we do.
}

def get_trl_timetags(hdu_1, filepath, logger=None):
    """Return a dictionary that returns a full date-time string given a date, based on
    the scraping of recognizable date/time strings from the records in a TRL table.

    Input:
        hdu1            hdulist[1] from an opened TRL file.
        filepath        optional path to the TRL file, for error logging.
        logger          optional logger.

    Return:
        date_dict       dictionary keyed by a UTC date string ("yyyy-mm-dd"), which
                        returns the latest date-time string ("yyyy-mm-ddThh:mm:ss[.fff]")
                        associated with that date.
    """

    header1 = hdu_1.header
    column_name = header1['TTYPE1']

    # The file's timestamp is the latest possible value of the earliest logged date-time
    timestamp = os.path.getmtime(filepath)
    dt = datetime.datetime.utcfromtimestamp(timestamp)
    earliest_date = dt.isoformat()

    date_dict = {}
    for rec in hdu_1.data:
        text = rec[column_name]
        tz = 'UTC'  # unless otherwise specified
        match = YYYYDOYHHMMSS.fullmatch(text)
        if match:
            (yyyy, doy, hhmmss) = match.groups()
            hh = hhmmss[:2]
            mm = hhmmss[2:4]
            ss = hhmmss[4:6]
            hh_mm_ss = f'{hh}:{mm}:{ss}'

            date = datetime.date(int(yyyy),1,1) + datetime.timedelta(int(doy) - 1)
            mm = '%02d' % date.month
            dd = '%02d' % date.day

        elif match := D_MON_YY_HH_MM_SS.fullmatch(text):
            (d, mon, yy, hh_mm_ss) = match.groups()
            dd = ('0' + d if len(d) == 1 else d.replace(' ', '0'))
            mm = MONTHS[mon.upper()]
            yyyy = ('19' + yy if yy[0] == '9' else '20' + yy)

        elif match := MON_D_HH_MM_SS_TZ_YYYY.fullmatch(text):
            (mon, d, hh_mm_ss, tz, yyyy) = match.groups()
            dd = ('0' + d if len(d) == 1 else d.replace(' ', '0'))
            mm = MONTHS[mon.upper()]

        elif match := D_MON_YYYY_HH_MM_SS.fullmatch(text):
            (d, mon, yyyy, hh_mm_ss) = match.groups()
            dd = ('0' + d if len(d) == 1 else d.replace(' ', '0'))
            mm = MONTHS[mon.upper()]

        elif match := D_MON_YY_HH_MM_SS.fullmatch(text):
            (d, mon, yy, hh_mm_ss) = match.groups()
            dd = ('0' + d if len(d) == 1 else d.replace(' ', '0'))
            mm = MONTHS[mon.upper()]
            yyyy = ('19' + yy if yy[0] == '9' else '20' + yy)

        elif match := HH_MM_SS_DD_MON_YYYY.fullmatch(text):
            (hh_mm_ss, dd, mon, yyyy) = match.groups()
            mm = MONTHS[mon.upper()]

        elif match := HH_MM_SS_DD_MON_YY.fullmatch(text):
            (hh_mm_ss, dd, mon, yy) = match.groups()
            mm = MONTHS[mon.upper()]
            yyyy = ('19' + yy if yy[0] == '9' else '20' + yy)

        elif match := HH_MM_SS_FFF_DD_MM_YYYY.fullmatch(text):
            (hh_mm_ss, dd, mm, yyyy) = match.groups()

        elif match := YYYY_MM_DD_HH_MM_SS_FFF.fullmatch(text):
            (yyyy, mm, dd, hh_mm_ss) = match.groups()

        elif match := HH_MM_SS_TZ_DD_MON_YYYY.fullmatch(text):
            (hh_mm_ss, tz, dd, mon, yyyy) = match.groups()
            mm = MONTHS[mon.upper()]

        elif match := MON_D_YYYY_HH_MM_SS.fullmatch(text):
            (mon, d, yyyy, hh_mm_ss) = match.groups()
            dd = ('0' + d if len(d) == 1 else d.replace(' ', '0'))
            mm = MONTHS[mon.upper()]

        elif match := D_MM_YY_HH_MM_SS.fullmatch(text):
            (d, mm, yy, hh_mm_ss) = match.groups()
            dd = ('0' + d if len(d) == 1 else d.replace(' ', '0'))
            yyyy = ('19' + yy if yy[0] == '9' else '20' + yy)

        elif match := D_MON_YYYY_HH_MM.fullmatch(text):
            (d, mon, yyyy, hh_mm) = match.groups()
            dd = ('0' + d if len(d) == 1 else d.replace(' ', '0'))
            mm = MONTHS[mon.upper()]
            hh_mm_ss = hh_mm + ':00'

        # last resort, some other random time string
        else:
            match = TIME_TEST.fullmatch(text)
            if not match:
                continue

            (before, hh_mm_ss, after) = match.groups()
            remainder = before[-14:] + ' ' + after[14:]

            match = MONTH_TEST.fullmatch(remainder)
            if match:
                (before, mon, after) = match.groups()
                mm = MONTHS[mon.upper()]
                remainder = before + ' ' + after
                if match := YYYY_D_TEST.fullmatch(remainder):
                    (yyyy, d) = match.groups()
                elif match := YY_D_TEST.fullmatch(remainder):
                    (yy, d) = match.groups()
                    yyyy = ('19' + yy if yy[0] == '9' else '20' + yy)
                elif match := D_YYYY_TEST.fullmatch(remainder):
                    (d, yyyy) = match.groups()
                elif match := D_YY_TEST.fullmatch(remainder):
                    (d, yy) = match.groups()
                    yyyy = ('19' + yy if yy[0] == '9' else '20' + yy)
                else:
                    continue

            elif match := D_MM_YYYY_TEST.fullmatch(remainder):
                (d, mm, yyyy) = match.groups()
            elif match := D_MM_YY_TEST.fullmatch(remainder):
                (d, mm, yy) = match.groups()
                yyyy = ('19' + yy if yy[0] == '9' else '20' + yy)
            elif match := MM_D_YYYY_TEST.fullmatch(remainder):
                (mm, d, yyyy) = match.groups()
            elif match := MM_D_YY_TEST.fullmatch(remainder):
                (mm, d, yy) = match.groups()
                yyyy = ('19' + yy if yy[0] == '9' else '20' + yy)
            elif match := YYYY_MM_D_TEST.fullmatch(remainder):
                (yyyy, mm, d) = match.groups()
            elif match := YY_MM_D_TEST.fullmatch(remainder):
                (yy, mm, d) = match.groups()
                yyyy = ('19' + yy if yy[0] == '9' else '20' + yy)
            else:
                continue

            dd = ('0' + d if len(d) == 1 else d.replace(' ', '0'))
            logger.warn('Unexpected time format interpreted as ' +
                        f'{yyyy}-{mm}-{dd}T{hh_mm_ss}',
                        filepath)
            logger.warn(text.strip())

        yyyy_mm_dd = f'{yyyy}-{mm}-{dd}'
        iso_date = yyyy_mm_dd + 'T' + hh_mm_ss

        try:
            offset = TIMEZONES[tz]
        except KeyError:
            logger.error(f'Unsupported time zone {tz} found; replaced with UTC',
                         filepath)
            offset = 0

        if offset:
            dt = (datetime.datetime.fromisoformat(iso_date)
                  - datetime.timedelta(seconds=offset*3600))
            iso_date = dt.isoformat()
            yyyy_mm_dd = iso_date[:10]

        date_dict[yyyy_mm_dd] = iso_date
        earliest_date = min(earliest_date, iso_date)

    if not date_dict:
        logger.error('No dates found', filepath)

    date_dict['earliest'] = earliest_date

    return date_dict

##########################################################################################

def get_label_retrieval_date(filepath, label_suffix='.xml'):
    """If the given data file already has a label, return the retrieval date from that
    label. Otherwise, return an empty string.

    This looks for specific text inside the label, so it will need to be modified if the
    template is modified.
    """

    label_path = os.path.splitext(filepath)[0] + label_suffix
    if not os.path.exists(label_path):
        return ''

    with open(label_path) as f:
        for rec in f:
            if '<Observation_Area>' in rec or '<Context_Area>' in rec:
                break

        for rec in f:
            if '<comment>' in rec:
                break

        rec = f.readline().strip() + ' ' + f.readline().strip()

    parts = rec.split('as obtained from the Mikulski ' +
                      'Archive for Space Telescopes (MAST) data archive on ')

    if len(parts) == 2:
        return parts[1][:-1]    # strip off the final period

    return ''

##########################################################################################

def get_file_creation_date(filepath):
    """Get the creation date of a file if it is available, or the last access date
    otherwise. Whether the creation date is available is OS-dependent.
    """

    # Works for Mac and some Unix systems
    try:
        # Works for Mac and some Unix systems
        time = os.stat(filepath).st_birthtime
    except AttributeError:
        # This works for Windows but not Mac or Unix
        ctime = os.path.getctime(filepath)

        # For Unix, this may be our best option
        mtime = os.path.getmtime(filepath)

        time = min(ctime, mtime)

    return datetime.datetime.fromtimestamp(time).isoformat()

##########################################################################################

def set_file_timestamp(filepath, date):
    """Set the modification date of the given file to the given date string, where the
    date is of the form "yyyy-mm-dd" or "yyyy-mm-ddThh:mm:ss[.fff]" as a time UTC.
    """

    if not date:
        return

    if 'T' not in date:
        date = date + 'T12:00:00'

    dt = datetime.datetime.fromisoformat(date)
    timestamp = (dt - datetime.datetime(1970, 1, 1)).total_seconds()

    access_time = os.path.getatime(filepath)    # don't change the access time
    os.utime(filepath, (access_time, timestamp))

##########################################################################################
