from typing import Generator, List

from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID


"""The suffixes considered raw data, in order of preference."""
RAW_SUFFIXES: List[str] = ["raw", "flt", "drz", "crj", "d0f", "d0m", "c0f", "c0m"]


def associated_lids(association_lid: LID, memname: str) -> Generator[LID, None, None]:
    """
    Generate LIDs for associated files from the LID of the ASN file
    and the memname in the assocation table inside the ASN file.
    """
    for suffix in RAW_SUFFIXES:
        other_lid = association_lid.to_other_suffixed_lid(suffix)
        parts = other_lid.parts()
        parts[2] = memname.lower()
        yield LID.create_from_parts(parts)


def associated_lidvids(
    association_lidvid: LIDVID, memname: str
) -> Generator[LIDVID, None, None]:
    """
    Generate LIDVIDs for associated files from the LID of the ASN file
    and the memname in the assocation table inside the ASN file.  We
    force the VID to "1.0".  This is a hack and will not work when we
    have multiple values.
    """
    initial_vid = VID("1.0")
    association_lid = association_lidvid.lid()
    for lid in associated_lids(association_lid, memname):
        yield LIDVID.create_from_lid_and_vid(lid, initial_vid)
