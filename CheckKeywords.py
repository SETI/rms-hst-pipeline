"""
**SCRIPT:** Run through the archive and print the sorted set of
keywords found in the first header unit of the FITS files of RAW
products.
"""
from pdart.exceptions.Combinators import *
from pdart.pds4.Archives import *
from pdart.pds4.Collection import *
from pdart.reductions.Reduction import *


def _union_dicts(dicts):
    """
    Summarize a list of dictionaries from keywords to integers by
    summing the integer values.
    """
    res = {}
    for d in dicts:
        for k, v in d.items():
            if k not in res:
                res[k] = 0
            res[k] += v
    return res


class CheckKeywordsReduction(Reduction):
    """
    Summarize the archive into a dictionary of keywords found in the
    first header unit of all RAW FITS files in the archive.
    """
    def reduce_archive(self, archive_root, get_reduced_bundles):
        # dict
        return _union_dicts(get_reduced_bundles())

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        # dict
        return _union_dicts(get_reduced_collections())

    def reduce_collection(self, archive, lid, get_reduced_products):
        # dict
        collection = Collection(archive, lid)
        if collection.suffix() == 'raw':
            return _union_dicts([rp for rp in get_reduced_products()
                                 if rp is not None])
        else:
            return {}

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        # dict or None
        return get_reduced_fits_files()[0]

    def reduce_fits_file(self, file, get_reduced_hdus):
        # dict or None
        return get_reduced_hdus()[0]

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        # dict or None
        if n == 0:
            return {k: 1 for k in hdu.header.iterkeys() if k != ''}

if __name__ == '__main__':
    reduction = CheckKeywordsReduction()
    archive = get_any_archive()

    def thunk():
        d = run_reduction(reduction, archive)
        print sorted(list(d.iterkeys()))
    raise_verbosely(thunk)

# Keywords found in the RAW FITS files of the archive are:
#
# 'ACQNAME', 'APERTURE', 'ASN_ID', 'ASN_MTYP', 'ASN_TAB', 'ATODCORR',
# 'ATODGNA', 'ATODGNB', 'ATODGNC', 'ATODGND', 'ATODTAB', 'BADINPDQ',
# 'BIASCORR', 'BIASFILE', 'BIASLEVA', 'BIASLEVB', 'BIASLEVC',
# 'BIASLEVD', 'BITPIX', 'BLEVCORR', 'BPIXTAB', 'CAL_VER', 'CCDAMP',
# 'CCDGAIN', 'CCDOFSTA', 'CCDOFSTB', 'CCDOFSTC', 'CCDOFSTD', 'CCDTAB',
# 'CFLTFILE', 'COMPTAB', 'CRCORR', 'CRDS_CTX', 'CRDS_VER', 'CRMASK',
# 'CRRADIUS', 'CRREJTAB', 'CRSIGMAS', 'CRSPLIT', 'CRTHRESH', 'CTEDIR',
# 'CTEIMAGE', 'D2IMFILE', 'DARKCORR', 'DARKFILE', 'DARKTIME', 'DATE',
# 'DATE-OBS', 'DEC_TARG', 'DETECTOR', 'DFLTFILE', 'DGEOFILE',
# 'DIRIMAGE', 'DQICORR', 'DRIZCORR', 'DRKCFILE', 'EQUINOX', 'EXPEND',
# 'EXPFLAG', 'EXPSCORR', 'EXPSTART', 'EXPTIME', 'EXTEND', 'FGSLOCK',
# 'FILENAME', 'FILETYPE', 'FILTER1', 'FILTER2', 'FLASHCUR',
# 'FLASHDUR', 'FLASHSTA', 'FLATCORR', 'FLSHCORR', 'FLSHFILE',
# 'FW1ERROR', 'FW1OFFST', 'FW2ERROR', 'FW2OFFST', 'FWSERROR',
# 'FWSOFFST', 'GRAPHTAB', 'GROUPS', 'GYROMODE', 'IDCTAB', 'IMAGETYP',
# 'IMPHTTAB', 'INITGUES', 'INSTRUME', 'JWROTYPE', 'LFLTFILE',
# 'LINENUM', 'LRFWAVE', 'MDRIZTAB', 'MEANEXP', 'MLINTAB', 'MOONANGL',
# 'MTFLAG', 'NAXIS', 'NEXTEND', 'NPOLFILE', 'NRPTEXP', 'OBSMODE',
# 'OBSTYPE', 'OPUS_VER', 'OSCNTAB', 'P1_ANGLE', 'P1_CENTR',
# 'P1_FRAME', 'P1_LSPAC', 'P1_NPTS', 'P1_ORINT', 'P1_PSPAC',
# 'P1_PURPS', 'P1_SHAPE', 'PATTERN1', 'PATTSTEP', 'PA_V3', 'PCTECORR',
# 'PCTETAB', 'PFLTFILE', 'PHOTCORR', 'PHOTTAB', 'POSTARG1',
# 'POSTARG2', 'PRIMESI', 'PROCTIME', 'PROPAPER', 'PROPOSID',
# 'PR_INV_F', 'PR_INV_L', 'PR_INV_M', 'QUALCOM1', 'QUALCOM2',
# 'QUALCOM3', 'QUALITY', 'RA_TARG', 'READNSEA', 'READNSEB',
# 'READNSEC', 'READNSED', 'REFFRAME', 'REJ_RATE', 'ROOTNAME',
# 'RPTCORR', 'SCALENSE', 'SCLAMP', 'SHADCORR', 'SHADFILE', 'SHUTRPOS',
# 'SIMPLE', 'SKYSUB', 'SKYSUM', 'SPOTTAB', 'STATFLAG', 'SUBARRAY',
# 'SUNANGLE', 'SUN_ALT', 'TARGNAME', 'TELESCOP', 'TIME-OBS',
# 'T_SGSTAR', 'UREKA_V', 'WRTERR'
