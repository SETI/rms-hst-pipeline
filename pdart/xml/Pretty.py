"""
Pretty-printing functionality.
"""
from pdart.xml.Schema import run_subprocess


def pretty_print(str):
    # (unicode) -> unicode
    """Reformat XML using xmllint --format."""
    (exit_code, stderr, stdout) = run_subprocess(['xmllint', '--format', '-'],
                                                 str)
    if exit_code == 0:
        return stdout
    else:
        # ignore stdout
        raise Exception('pretty_print failed')
