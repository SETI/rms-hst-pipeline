u"""
Functionality to validate XML against various schemas using external
programs.

:func:`verify_label_or_raise` is the main function used for validating
PDS4 labels.
"""
import os
import os.path
import subprocess
import tempfile
import xml.dom.minidom
from contextlib import closing

from typing import TYPE_CHECKING

from pdart.xml.Pds4Version import HST_SHORT_VERSION, PDS4_SHORT_VERSION

if TYPE_CHECKING:
    from typing import List, Optional, Sequence, Tuple, Union

    _Cmd = Union[str, Sequence[str]]

PDS_XML_SCHEMA = ('./xml/PDS4_PDS_%s.xsd.xml' %
                  PDS4_SHORT_VERSION)  # type: str

PDS_SCHEMATRON_SCHEMA = ('./xml/PDS4_PDS_%s.sch.xml' %
                         PDS4_SHORT_VERSION)  # type: str

HST_XML_SCHEMA = ('./xml/PDS4_HST_%s.xsd.xml' %
                  HST_SHORT_VERSION)  # type: str

HST_SCHEMATRON_SCHEMA = ('./xml/PDS4_HST_%s.sch.xml' %
                         HST_SHORT_VERSION)  # type: str


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
    # type: (Optional[unicode], Optional[unicode], List[str]) -> Tuple[int, unicode, unicode]
    """
    Run XsdValidator.jar on the XML at the filepath (ignored if stdin
    is not None) or on stdin, validating against the schemas.  Returns
    a triple of exit_code, stderr, and stdout.
    """
    args = ['java', '-jar', 'XsdValidator.jar']

    for filename in schemas:
        assert os.path.isfile(filename), 'schema %s exists' % filename

    args.extend(schemas)
    if stdin is None:
        args.append(str(filepath))  # illogical cast to shut up mypy
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
    # type: (Optional[unicode], unicode, str) -> Optional[unicode]
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
    # type: (unicode, str) -> Tuple[int, unicode, unicode]
    """
    Run probatron on the XML at the filepath validating against the
    schema.  Returns a triple of exit_code, stderr and stdout.
    """
    return run_subprocess(['java', '-jar', 'probatron.jar',
                           '-r0',  # output report as terse SVRL
                           '-n1',  # emit line/col numbers in report
                           str(filepath),  # illogical cast to shut up mypy
                           schema])


def probatron_with_stdin(filepath, stdin=None, schema=PDS_SCHEMATRON_SCHEMA):
    # type: (Optional[unicode], Optional[unicode], str) -> Tuple[int, unicode, unicode]
    """
    Run probatron on the XML at the filepath (ignored if stdin is not
    None) or in stdin, validating against the schema.  Returns a
    triple of exit_code, stderr and stdout.
    """
    if stdin is None:
        assert filepath
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
    # type: (Optional[unicode], Optional[unicode], str) -> xml.dom.minidom.Document
    """
    Run probatron on the XML at the filepath (ignored if stdin is not
    None) or in stdin, validating against the schema.  Returns the
    SVRL as XML.
    """
    exit_code, stderr, stdout = probatron_with_stdin(filepath, stdin, schema)
    assert stderr == '', 'stderr = %r' % stderr
    assert exit_code == 0, 'exit_code = %r' % exit_code
    return xml.dom.minidom.parseString(stdout)


def svrl_failures(svrl):
    # type: (xml.dom.minidom.Document) -> Sequence[xml.dom.minidom.Node]
    return svrl.documentElement.getElementsByTagName('svrl:failed-assert')


def svrl_has_failures(svrl):
    # type: (xml.dom.minidom.Document) -> bool
    """
    Given an SVRL document, return True iff it contains failures.
    """
    return len(svrl_failures(svrl)) > 0


def schematron_failures(filepath, stdin=None, schema=PDS_SCHEMATRON_SCHEMA):
    # type: (Optional[unicode], Optional[unicode], str) -> Optional[unicode]
    """
    Run probatron on the XML at the filepath (ignored if stdin is not
    None) or in stdin, validating against the schema.  Returns None if
    there are no failures; returns a string containing the failures if
    they exist.
    """
    svrl = probatron_with_svrl_result(filepath, stdin, schema)
    failures = svrl_failures(svrl)
    if len(failures) > 0:
        # should I have a pretty option here for human-readability?

        # TODO The replace probably ought to be done only in the
        # reporting, only as necessary
        return ('\n'.join([f.toxml() for f in failures])).replace('\n', '\\n')
    else:
        return None


def verify_label_or_raise_fp(filepath):
    # type: (unicode) -> None
    with closing(open(filepath, 'r')) as f:
        label = f.read()
    verify_label_or_raise(label)


def verify_label_or_raise(label):
    # type: (unicode) -> None
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
    except Exception:
        # Debugging functionality: write the label to disk.
        PRINT_AND_SAVE_LABEL = True
        if PRINT_AND_SAVE_LABEL:
            import time
            print label
            t = int(time.time() * 1000)
            fp = 'tmp%d.xml' % t
            with open(fp, 'w') as f:
                f.write(label.encode('utf-8'))
        raise
