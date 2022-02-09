from subprocess import CompletedProcess, run
from typing import Set
from os.path import join, splitext


def _convert_apt_file_to_utf8(document: str) -> None:
    completed_process: CompletedProcess = run(["./convert_apt_to_utf8", document])
    completed_process.check_returncode()


def convert_documents_to_utf8(documents_dir: str, documents: Set[str]) -> None:
    for document in documents:
        _root, ext = splitext(document)
        if ext.lower() == ".apt":
            _convert_apt_file_to_utf8(join(documents_dir, document))
