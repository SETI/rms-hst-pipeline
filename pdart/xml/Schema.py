u"""
Functionality to validate XML against various schemas using external
programs.

:func:`verify_label_or_raise` is the main function used for validating
PDS4 labels.
"""
from contextlib import closing
import os
import subprocess
import tempfile
import xml.dom.minidom

from pdart.xml.Pds4Version import *
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence, Tuple, Union
    _Cmd = Union[str, Sequence[str]]

PDS_XML_SCHEMA = './xml/PDS4_PDS_%s.xsd.xml' % PDS4_SHORT_VERSION
# type: str

PDS_SCHEMATRON_SCHEMA = './xml/PDS4_PDS_%s.sch.xml' % 1600 \
    # PDS4_SHORT_VERSION -- FIXME
# type: str

HST_XML_SCHEMA = './xml/PDS4_HST_%s_0200.xsd.xml' % 1600  \
    # PDS4_SHORT_VERSION -- FIXME
# type: str

HST_SCHEMATRON_SCHEMA = './xml/PDS4_HST_%s_0200.sch.xml' % PDS4_SHORT_VERSION
# type: str


def run_subprocess(cmd, stdin=None):
    # type: (_Cmd, unicode) -> Tuple[int, unicode, unicode]
    """
    Run a command in a subprocess.  Command can be either a string
    with the command name, or an array of the command name and its
    arguments.  If stdin is given, use it as input.  Return a triple
    of exit_code, stderr, and stdout.  (NOTE: alphabetical order.)
    The last two are strings.
    """
    if stdin is None:
        p = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
    else:
        p = subprocess.Popen(cmd,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        stdout, stderr = p.communicate(stdin)
    exit_code = p.returncode
    # return in alphabetical order
    return (exit_code, stderr, stdout)


def _xsd_validator_schema(filepath,
                          stdin=None,
                          schemas=[PDS_XML_SCHEMA, HST_XML_SCHEMA]):
    # type: (str, unicode, List[str]) -> Tuple[int, unicode, unicode]
    """
    Run XsdValidator.jar on the XML at the filepath (ignored if stdin
    is not None) or on stdin, validating against the schemas.  Returns
    a triple of exit_code, stderr, and stdout.
    """
    args = ['java', '-jar', 'XsdValidator.jar']
    args.extend(schemas)
    if stdin is None:
        args.append(filepath)
        return run_subprocess(args)
    else:
        args.append('-')
        return run_subprocess(args, stdin=stdin)


def _xmllint_schema(filepath, stdin=None, schema=PDS_XML_SCHEMA):
    # type: (str, unicode, str) -> Tuple[int, unicode, unicode]
    """
    Run xmllint on the XML at the filepath (ignored if stdin is not
    None) or on stdin, validating against the schema.  Returns a
    triple of exit_code, stderr and stdout.
    """
    if stdin is None:
        return run_subprocess(['xmllint', '--noout',
                               '--schema', schema, filepath])
    else:
        return run_subprocess(['xmllint', '--noout', '--schema', schema, '-'],
                              stdin=stdin)


def xml_schema_failures(filepath, stdin=None, schema=PDS_XML_SCHEMA):
    # type: (str, unicode, str) -> unicode
    """
    Run an XML Schema validator on the XML at the filepath (ignored if
    stdin is not None) or in stdin, validating against the schema.
    Returns None if there are no failures; returns a string containing
    the failures if they exist.
    """
    if True:
        exit_code, stderr, _ = _xsd_validator_schema(filepath, stdin=stdin)
    else:
        exit_code, stderr, _ = _xmllint_schema(filepath,
                                               stdin=stdin,
                                               schema=schema)
    if exit_code == 0:
        return None
    else:
        # ignore stdout
        assert stderr
        return stderr


def probatron(filepath, schema=PDS_SCHEMATRON_SCHEMA):
    # type: (str, str) -> Tuple[int, unicode, unicode]
    """
    Run probatron on the XML at the filepath validating against the
    schema.  Returns a triple of exit_code, stderr and stdout.
    """
    return run_subprocess(['java', '-jar', 'probatron.jar',
                           '-r0',  # output report as terse SVRL
                           '-n1',  # emit line/col numbers in report
                           filepath, schema])


def probatron_with_stdin(filepath, stdin=None, schema=PDS_SCHEMATRON_SCHEMA):
    # type: (str, unicode, str) -> Tuple[int, unicode, unicode]
    """
    Run probatron on the XML at the filepath (ignored if stdin is not
    None) or in stdin, validating against the schema.  Returns a
    triple of exit_code, stderr and stdout.
    """
    if stdin is None:
        return probatron(filepath, schema)
    else:
        (handle, filepath) = tempfile.mkstemp()
        try:
            f = os.fdopen(handle, 'w')
            try:
                f.write(stdin)
            finally:
                f.close()
            return probatron(filepath, schema)
        finally:
            os.remove(filepath)


def probatron_with_svrl_result(filepath,
                               stdin=None,
                               schema=PDS_SCHEMATRON_SCHEMA):
    # type: (str, unicode, str) -> xml.dom.minidom.Document
    """
    Run probatron on the XML at the filepath (ignored if stdin is not
    None) or in stdin, validating against the schema.  Returns the
    SVRL as XML.
    """
    exit_code, stderr, stdout = probatron_with_stdin(filepath, stdin, schema)
    assert stderr == '', 'stderr = %r' % stderr
    assert exit_code == 0, 'exit_code = %r' % exit_code
    return xml.dom.minidom.parseString(stdout)


def _svrl_failures(svrl):
    # type: (xml.dom.minidom.Document) -> Sequence[xml.dom.minidom.Node]
    return svrl.documentElement.getElementsByTagName('svrl:failed-assert')


def svrl_has_failures(svrl):
    # type: (xml.dom.minidom.Document) -> bool
    """
    Given an SVRL document, return True iff it contains failures.
    """
    return len(_svrl_failures(svrl)) > 0


def schematron_failures(filepath, stdin=None, schema=PDS_SCHEMATRON_SCHEMA):
    # type: (str, unicode, str) -> unicode
    """
    Run probatron on the XML at the filepath (ignored if stdin is not
    None) or in stdin, validating against the schema.  Returns None if
    there are no failures; returns a string containing the failures if
    they exist.
    """
    svrl = probatron_with_svrl_result(filepath, stdin, schema)
    failures = _svrl_failures(svrl)
    if len(failures) > 0:
        # should I have a pretty option here for human-readability?

        # TODO The replace probably ought to be done only in the
        # reporting, only as necessary
        return ('\n'.join([f.toxml() for f in failures])).replace('\n', '\\n')
    else:
        return None


def verify_label_or_raise_fp(filepath):
    # type: (str) -> None
    with closing(open(filepath, 'r')) as f:
        label = f.read()
    verify_label_or_raise(label)


def verify_label_or_raise(label):
    # type: (str) -> None
    """
    Given the text of a PDS4 label, run XML Schema *and* Schematron
    validations on it.  Raise an exception on failures.
    """
    try:
        failures = xml_schema_failures(None, label)
        if failures is not None:
            raise Exception('XML schema validation errors: ' + failures)
        failures = schematron_failures(None, label)
        if failures is not None:
            raise Exception('Schematron validation errors: ' + failures)
    except:
        # Debugging functionality: write the label to disk.
        PRINT_AND_SAVE_LABEL = False
        if PRINT_AND_SAVE_LABEL:
            import time
            print label
            t = int(time.time() * 1000)
            fp = 'tmp%d.xml' % t
            with open(fp, 'w') as f:
                f.write(label)
        raise
