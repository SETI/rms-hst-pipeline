##########################################################################################
# tests/test_date_support.py
#
# Tests for the date-times that get_trl_timetags reads from the text of TRL files.
##########################################################################################

import pdslogger
import pytest
from astropy.io import fits

from hst_pipeline.product_labels.date_support import get_trl_timetags


def _timetags(tmp_path, lines):
    """Return get_trl_timetags of a TRL table holding the given lines, and the number of
    warnings it logged."""
    hdu = fits.BinTableHDU.from_columns([fits.Column(name='TEXT_FILE', format='132A',
                                                     array=lines)])
    path = tmp_path / 'test_trl.fits'
    path.write_text('')
    logger = pdslogger.PdsLogger(f'test.date-support-{len(list(tmp_path.iterdir()))}'
                                 f'-{abs(hash(tuple(lines)))}')
    logger.open('test')
    timetags = get_trl_timetags(hdu, str(path), logger)
    warnings = logger.close()[2]
    return (timetags, warnings)


@pytest.mark.parametrize('line, expected', [
    # astrodrizzle; the fallback once read this as 2011-06-03 from "Version 3.11.0"
    ('[INFO    ] 21:42:47 (19-May-2026): Processing j9mm06ecq_flt.fits with '
     'astrodrizzle Version 3.11.0', '2026-05-19T21:42:47'),
    ('[INFO    ] 09:18:18 (20-May-2026): Finished processing', '2026-05-20T09:18:18'),
    ('[INFO    ] 09:18:18.123 ( 5-May-2026): Finished', '2026-05-05T09:18:18.123'),
    # two-digit days once lost their first digit (19-May read as 9-May)
    ('Begin    19-May-2026 21:42:43 UTC', '2026-05-19T21:42:43'),
    ('End       5-May-2026 21:42:43 UTC', '2026-05-05T21:42:43'),
    ('CGAT09-CGPHSP END: 21-MAY-92, 00:17:03', '1992-05-21T00:17:03'),
    ('Processing 21/05/92  00:17:03', '1992-05-21T00:17:03'),
    ('Start 21-May-2026 13:45 here', '2026-05-21T13:45:00'),
    ('=== BEP 1.6 (2009 Mar 26)  starting at  Sun Aug 16 09:29:09 2009',
     '2009-08-16T09:29:09'),
])
def test_trl_formats(tmp_path, line, expected):
    (timetags, warnings) = _timetags(tmp_path, [line])
    assert timetags == {expected[:10]: expected}
    assert warnings == 0


def test_python_build_date_is_ignored(tmp_path):
    line = ('[INFO    ] Python Version 3.12.13 | packaged by conda-forge | '
            '(main, Mar  5 2026, 16:50:00) [GCC 14.3.0]')
    (timetags, _) = _timetags(tmp_path, [line, 'Begin    19-May-2026 21:42:43 UTC'])
    assert timetags == {'2026-05-19': '2026-05-19T21:42:43'}


def test_fallback_reads_the_date_beside_the_time(tmp_path):
    # An unknown format: the fallback uses the text next to the time, and warns.
    (timetags, warnings) = _timetags(tmp_path, ['Run at 21:42:47 on 2026/05/19 v3.11.0'])
    assert timetags == {'2026-05-19': '2026-05-19T21:42:47'}
    assert warnings == 2        # the interpretation and the line itself
