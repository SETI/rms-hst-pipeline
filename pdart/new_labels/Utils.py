from pdart.pds4.LIDVID import LIDVID


def lidvid_to_lid(lidvid):
    # type: (str) -> str
    return str(LIDVID(lidvid).lid())


def lidvid_to_vid(lidvid):
    # type: (str) -> str
    return str(LIDVID(lidvid).vid())
