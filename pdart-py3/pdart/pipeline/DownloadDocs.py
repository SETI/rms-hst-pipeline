import os
import os.path

from pdart.documents.Downloads import download_product_documents


def do_download_docs(documents_dir: str, proposal_id: int) -> None:
    assert not os.path.isdir(documents_dir)
    os.makedirs(documents_dir)
    download_product_documents(proposal_id, documents_dir)


def download_docs(documents_dir: str, proposal_id: int) -> None:
    if not os.path.isdir(documents_dir):
        do_download_docs(documents_dir, proposal_id)
