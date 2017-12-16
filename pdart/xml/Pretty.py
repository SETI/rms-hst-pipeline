"""
Pretty-printing functionality.
"""
from pdart.xml.Schema import run_subprocess, verify_label_or_raise


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


def pretty_and_verify(label, verify):
    # type: (unicode, bool) -> unicode
    label = pretty_print(label)
    if verify:
        verify_label_or_raise(label)
    return label
