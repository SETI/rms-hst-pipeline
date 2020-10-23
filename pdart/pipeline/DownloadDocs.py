import os
import os.path

from pdart.documents.ConvertToUtf8 import convert_documents_to_utf8
from pdart.documents.Downloads import download_product_documents
from pdart.pipeline.Stage import MarkedStage


class DownloadDocs(MarkedStage):
    """
    This stage downloads document files to the documents directory,
    creating the directory if necessary.

    After this stage runs, there should be a documents directory that
    contains at least one document file.
    """

    def _do_download_docs(self, documents_dir: str, proposal_id: int) -> None:
        assert not os.path.isdir(documents_dir)
        os.makedirs(documents_dir)
        docs = download_product_documents(proposal_id, documents_dir)
        convert_documents_to_utf8(documents_dir, docs)

    def _run(self) -> None:
        documents_dir: str = self.documents_dir()
        if not os.path.isdir(documents_dir):
            self._do_download_docs(documents_dir, self._proposal_id)

        assert os.path.isdir(documents_dir)
