##########################################################################################
##########################################################################################

import julian
import pdslogger

def get_time_coordinates(ref_hdulist, spt_hdulist, filepath='', logger=None):
    """Return the tuple (start_time, stop_time).

    Input:
        ref_hdulist     HDU list for the reference data file, as returned by
                        astropy.io.fits.open().
        spt_hdulist     HDU list for the SPT data file.
        filepath        name of the reference file, primarily for error logging.
        logger          pdslogger to use.
    """

    logger = logger or pdslogger.NullLogger()

    header0 = ref_hdulist[0].header
    header1 = ref_hdulist[1].header

    # Quick internal function
    def get_from_header(key, alt=None):
        for header in (header0, header1, spt_hdulist[0].header):
            try:
                return header[key]
            except KeyError:
                pass

        return alt

    date_obs = get_from_header('DATE-OBS')
    time_obs = get_from_header('TIME-OBS')
    exptime  = get_from_header('EXPTIME', 0.)

    # Initial guess
    if date_obs and time_obs:
        if '/' in date_obs:     # really??
            old_date_obs = date_obs
            dd = date_obs[:2]
            mm = date_obs[3:5]
            yy = date_obs[6:8]
            date_obs = f'19{yy}-{mm}-{dd}'
            logger.debug(f'DATE-OBS format corrected from {old_date_obs} to {date_obs}',
                         filepath)

        if ' ' in date_obs:
            old_date_obs = date_obs
            date_obs = date_obs.replace(' ', '0')
            logger.debug(f'DATE-OBS format corrected from {old_date_obs} to {date_obs}',
                         filepath)

        start_date_time = date_obs + 'T' + time_obs + 'Z'
        stop_tai = julian.tai_from_iso(start_date_time) + exptime
        stop_date_time = julian.ymdhms_format_from_tai(stop_tai, suffix='Z')
        guess = (start_date_time, stop_date_time)
    else:
        guess = None

    # Either EXPSTART or TEXPSTART should be available, same for EXPEND/TEXPEND
    expstart = get_from_header('EXPSTART') or get_from_header('TEXPSTART')
    expend   = get_from_header('EXPEND')   or get_from_header('TEXPEND')
    if not expstart or not expend:
        if guess:
            logger.warn('Missing FITS keywords EXPSTART/EXPEND; '
                        'using DATE-OBS and TIME-OBS', filepath)
            return guess
        else:
            logger.error('Missing FITS keywords EXPSTART/EXPEND', filepath)
            return ('UNK', 'UNK')

    # HST documents indicate that times are only accurate to a second or so. This is
    # consistent with the fact that start times indicated by DATE-OBS and TIME-OBS often
    # disagree with the times as indicated by EXPSTART at the level of a second or so.
    # For any individual time, this is fine, but we want to be sure that the difference
    # between the start and stop times is compatible with the exposure time, whenever
    # appropriate.
    #
    # I say 'whenever appropriate' because there are times when multiple images have been
    # drizzled or otherwise merged. In this case, the start and stop times refer to the
    # first and last of the set of images, respectively, and their difference can be much
    # greater than the exposure time.
    #
    # It takes some careful handling to get the behavior we want.

    # Decide which delta-time to use...
    # Our start and stop times are only ever good to the nearest second, but we
    # want to ensure that the difference looks right. For this purpose,
    # non-integral exposure times should be rounded up to the next integer.

    delta_from_mjd = (expend - expstart) * 86400.0
    if delta_from_mjd > exptime + 2.0:  # if the delta is too large, we know
                                        # multiple images were combined
        delta = delta_from_mjd
    else:
        delta = -(-exptime // 1.0)      # rounded up to nearest int

    # Fill in the start time; update the expstart in MJD units if necessary. If
    # DATE-OBS and TIME-OBS values are provided, we use this as the start time
    # because it is the value our users would expect. There exist cases when these
    # values are not provided, and in that case we use EXPSTART, converted from
    # MJD. Note that these MJD values are in UTC, not TAI. In other words, we need
    # to ignore leapseconds in these time conversions.

    if date_obs and time_obs:
        if '/' in date_obs:         # dd/mm/yy
            date_obs = date_obs.replace(' ', '0')
            parts = date_obs.split('/')
            date_obs = '-'.join(parts[::-1])
            if date_obs[0] == '9':          # 90 -> 1990; 99 -> 1999
                date_obs = '19' + date_obs
            else:                           # 00 -> 2000
                date_obs = '20' + date_obs
        start_date_time = date_obs + 'T' + time_obs + 'Z'
        day = julian.day_from_iso(date_obs)
        sec = julian.sec_from_iso(time_obs)
        expstart = julian.mjd_from_day_sec(day, sec)

    else:
        (day, sec) = julian.day_sec_from_mjd(expstart)
        start_date_time = julian.ymdhms_format_from_day_sec(day, sec, suffix='Z')

    # Fill in the stop time. We ensure that this differs from the start time by
    # the expected amount.

    expend = expstart + delta / 86400.0
    (day, sec) = julian.day_sec_from_mjd(expend)
    stop_date_time = julian.ymdhms_format_from_day_sec(day, sec, suffix='Z')

    return (start_date_time, stop_date_time)

##########################################################################################
