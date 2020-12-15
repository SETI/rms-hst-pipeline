import urllib.error
import urllib.parse
import urllib.request
from typing import List, Set, Tuple

import fs.path

DOCUMENT_SUFFIXES = [".apt", ".pdf", ".pro", ".prop"]


def _retrieve_doc(url: str, filepath: str) -> bool:
    """Retrieves the text at that URL or returns False."""
    try:
        resp = urllib.request.urlopen(url)
        contents: bytes = resp.read()
        with open(filepath, "wb") as f:
            f.write(contents)
            return True
    except urllib.error.URLError as e:
        print(e)
        return False


def download_product_documents(proposal_id: int, download_dir: str) -> Set[str]:
    """
    Using the templates, try to download the documentation files for
    this proposal ID into a directory and return a set of the
    basenames of the files successfully downloaded.
    """
    table: List[Tuple[str, str]] = [
        (f"https://www.stsci.edu/hst/phase2-public/{proposal_id}.apt", "phase2.apt"),
        (f"https://www.stsci.edu/hst/phase2-public/{proposal_id}.pdf", "phase2.pdf"),
        (f"https://www.stsci.edu/hst/phase2-public/{proposal_id}.pro", "phase2.pro"),
        (f"https://www.stsci.edu/hst/phase2-public/{proposal_id}.prop", "phase2.prop"),
    ]

    res: Set[str] = set()

    for (url, basename) in table:
        filepath = fs.path.join(download_dir, basename)
        if _retrieve_doc(url, filepath):
            res.add(basename)

    return res
