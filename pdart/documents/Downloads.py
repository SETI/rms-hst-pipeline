from typing import TYPE_CHECKING
import urllib2

import fs.path

if TYPE_CHECKING:
    from typing import List, Set, Tuple


def _retrieve_doc(url, filepath):
    # type: (unicode, unicode) -> bool
    """Retrieves the text at that URL or returns False."""
    try:
        resp = urllib2.urlopen(url)
        contents = resp.read()
        with open(filepath, 'w') as f:
            f.write(contents)
            return True
    except Exception:
        return False


def download_product_documents(proposal_id, download_dir):
    # type: (int, unicode) -> Set[unicode]
    """
    Using the templates, try to download the documentation files for
    this proposal ID into a directory and return a set of the
    basenames of the files successfully downloaded.
    """
    table = [
        (u'https://www.stsci.edu/hst/phase2-public/%d.apt', u'phase2.apt'),
        (u'https://www.stsci.edu/hst/phase2-public/%d.pdf', u'phase2.pdf'),
        (u'https://www.stsci.edu/hst/phase2-public/%d.pro', u'phase2.pro'),
        (u'https://www.stsci.edu/hst/phase2-public/%d.prop', u'phase2.prop')
        ]  # type: List[Tuple[unicode, unicode]]

    res = set()  # type: Set[unicode]

    for (url_template, basename) in table:
        url = url_template % proposal_id
        filepath = fs.path.join(download_dir, basename)
        if _retrieve_doc(url, filepath):
            res.add(basename)
    return res
