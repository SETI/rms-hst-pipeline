from pdart.fs.VersionView import VersionView

from fs.memoryfs import MemoryFS


def test_version_view():
    versioned_fs = MemoryFS()
    versioned_fs.makedirs(u'/hst_00000/v1')
    versioned_fs.makedirs(u'/hst_00000/v2')
    version_view = VersionView(u'urn:nasa:pds:hst_00000::2', versioned_fs)
    assert version_view.bundle_id == u'hst_00000'
    assert version_view.version_id == u'2'
    threw = False
    try:
        VersionView(u'urn:nasa:pds:hst_00000::3', versioned_fs)
    except:
        threw = True
    assert threw, "should've thrown"
