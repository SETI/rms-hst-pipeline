"""
**SCRIPT:** Work in progress.  Create the document collections for
each bundle in the archive.  Note that currently this requires calls
over the Internet to retrieve the documents.
"""

from pdart.pds4.Archives import *
import os.path
import urllib2
import xml.etree.ElementTree


def _get_apt_url(proposal_id):
    """Return the URL to get the APT file for the given proposal id."""
    return 'https://www.stsci.edu/hst/phase2-public/%d.apt' % proposal_id


def _get_pro_url(proposal_id):
    """Return the URL to get the PRO file for the given proposal id."""
    return 'https://www.stsci.edu/hst/phase2-public/%d.pro' % proposal_id


def _retrieve_doc(url):
    """Retrives the text at that URL or raises an exception."""
    resp = urllib2.urlopen(url)
    return resp.read()


def _retrieve_apt(proposal_id, docs_dir):
    """
    Retrieve the APT file for the given proposal id, write it into the
    document directory (creating the directory if necessary), then
    extract the abstract from it and write the abstract into the
    document directory
    """
    apt_xml = _retrieve_doc(_get_apt_url(proposal_id))
    if not os.path.isdir(docs_dir):
        os.mkdir(docs_dir)
    apt_fp = os.path.join(docs_dir, 'phase2.apt')
    with open(apt_fp, 'w') as f:
        f.write(apt_xml)
    print '# Wrote', apt_fp

    abstract_fp = os.path.join(docs_dir, 'abstract.txt')
    root = xml.etree.ElementTree.parse(apt_fp).getroot()
    assert root is not None
    abst = root.find('.//Abstract')
    assert abst is not None
    assert abst.text is not None
    with open(abstract_fp, 'w') as f:
        f.write(abst.text)
    print '# Wrote', abstract_fp


def _retrieve_pro(proposal_id, docs_dir):
    """
    Retrieve the PRO file for the given proposal id and write it
    into the document directory.
    """
    pro_xml = _retrieve_doc(_get_pro_url(proposal_id))
    pro_fp = os.path.join(docs_dir, 'phase2.txt')
    with open(pro_fp, 'w') as f:
        f.write(pro_xml)
    print '# Wrote', pro_fp


if __name__ == '__main__':
    archive = get_any_archive()
    for b in archive.bundles():
        print '#', b, 'proposal_id =', b.proposal_id()

        proposal_id = b.proposal_id()
        docs_dir = os.path.join(b.absolute_filepath(), 'document')
        try:
            _retrieve_apt(proposal_id, docs_dir)
            _retrieve_pro(proposal_id, docs_dir)
            # TODO Create the phase2.pdf file
        except urllib2.HTTPError as e:
            print e

        # TODO Create the collection_document.tab.

        # TODO Is this a utility to be run occasionally on its own, or
        # is this to be part of a general process.  If the latter, its
        # error handling needs to be rejiggered so it fits into the
        # task framework.
