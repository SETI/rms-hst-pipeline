from pdart.pds4.VID import VID


def vid_to_dir_name(vid):
    # type: (VID) -> unicode
    return 'v$%s' % str(vid)


def dir_name_to_vid(dirname):
    # type: (unicode) -> VID
    assert dirname[0] == 'v'
    assert dirname[1] == '$'
    return VID(dirname[2:])
