"""
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
from typing import List, Optional, Sequence, Tuple, Union

from pdart.xml.Pds4Version import (
    DISP_SHORT_VERSION,
    HST_SHORT_VERSION,
    PDS4_SHORT_VERSION,
)

_Cmd = Union[str, Sequence[str]]


PDS_XML_SCHEMA: str = f"./xml/PDS4_PDS_{PDS4_SHORT_VERSION}.xsd.xml"

PDS_SCHEMATRON_SCHEMA: str = f"./xml/PDS4_PDS_{PDS4_SHORT_VERSION}.sch.xml"

_USE_NEW_SCHEMA: bool = True

if _USE_NEW_SCHEMA:

    HST_XML_SCHEMA: str = f"./xml/PDS4_HST_{HST_SHORT_VERSION}-new.xsd.xml"

    HST_SCHEMATRON_SCHEMA: str = f"./xml/PDS4_HST_{HST_SHORT_VERSION}- new.sch.xml"

else:

    HST_XML_SCHEMA = f"./xml/PDS4_HST_{HST_SHORT_VERSION}.xsd.xml"

    HST_SCHEMATRON_SCHEMA = f"./xml/PDS4_HST_{HST_SHORT_VERSION}.sch.xml"

DISP_XML_SCHEMA: str = f"./xml/PDS4_DISP_{DISP_SHORT_VERSION}.xsd.xml"

DISP_SCHEMATRON_SCHEMA: str = f"./xml/PDS4_DISP_{DISP_SHORT_VERSION}.sch.xml"


def run_subprocess(
    cmd: _Cmd, stdin: Optional[bytes] = None
) -> Tuple[int, bytes, bytes]:
    """
    Run a command in a subprocess.  Command can be either a string
    with the command name, or an array of the command name and its
    arguments.  If stdin is given, use it as input.  Return a triple
    of exit_code, stderr, and stdout.  (NOTE: alphabetical order.)
    The last two are strings.
    """
    if stdin is None:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
    else:
        p = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = p.communicate(stdin)
    exit_code = p.returncode
    # return in alphabetical order
    return (exit_code, stderr, stdout)


def _xsd_validator_schema(
    filepath: Optional[str],
    stdin: Optional[bytes] = None,
    schemas: List[str] = [PDS_XML_SCHEMA, DISP_XML_SCHEMA, HST_XML_SCHEMA],
) -> Tuple[int, bytes, bytes]:
    """
    Run XsdValidator.jar on the XML at the filepath (ignored if stdin
    is not None) or on stdin, validating against the schemas.  Returns
    a triple of exit_code, stderr, and stdout.
    """
    args: List[str] = ["java", "-jar", "XsdValidator.jar"]

    for filename in schemas:
        assert os.path.isfile(filename), f"schema {filename} required"

    args.extend(schemas)
    if stdin is None:
        assert filepath is not None
        args.append(filepath)
        return run_subprocess(args)
    else:
        args.append("-")
        return run_subprocess(args, stdin=stdin)


def xml_schema_failures(
    filepath: Optional[str], stdin: Optional[bytes] = None, schema: str = PDS_XML_SCHEMA
) -> Optional[bytes]:
    """
    Run an XML Schema validator on the XML at the filepath (ignored if
    stdin is not None) or in stdin, validating against the schema.
    Returns None if there are no failures; returns a string containing
    the failures if they exist.
    """
    exit_code, stderr, _ = _xsd_validator_schema(filepath, stdin=stdin)
    if exit_code == 0:
        return None
    else:
        # ignore stdout
        assert stderr
        return stderr


def probatron(
    filepath: str, schema: str = PDS_SCHEMATRON_SCHEMA
) -> Tuple[int, bytes, bytes]:
    """
    Run probatron on the XML at the filepath validating against the
    schema.  Returns a triple of exit_code, stderr and stdout.
    """
    return run_subprocess(
        [
            "java",
            "-jar",
            "probatron.jar",
            "-r0",  # output report as terse SVRL
            "-n1",  # emit line/col numbers in report
            filepath,
            schema,
        ]
    )


def probatron_with_stdin(
    filepath: str, stdin: Optional[bytes] = None, schema: str = PDS_SCHEMATRON_SCHEMA
) -> Tuple[int, bytes, bytes]:
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
            f = os.fdopen(handle, "wb")
            try:
                f.write(stdin)
            finally:
                f.close()
            return probatron(filepath, schema)
        finally:
            os.remove(filepath)


def probatron_with_svrl_result(
    filepath: str, stdin: Optional[bytes] = None, schema: str = PDS_SCHEMATRON_SCHEMA
) -> xml.dom.minidom.Document:
    """
    Run probatron on the XML at the filepath (ignored if stdin is not
    None) or in stdin, validating against the schema.  Returns the
    SVRL as XML.
    """
    exit_code, stderr, stdout = probatron_with_stdin(filepath, stdin, schema)
    assert stderr == b"", f"stderr = {stderr!r}"
    assert exit_code == 0, f"exit_code = {exit_code!r}"
    return xml.dom.minidom.parseString(stdout)


def svrl_failures(svrl: xml.dom.minidom.Document) -> Sequence[xml.dom.minidom.Node]:
    return svrl.documentElement.getElementsByTagName("svrl:failed-assert")


def svrl_has_failures(svrl: xml.dom.minidom.Document) -> bool:
    """
    Given an SVRL document, return True iff it contains failures.
    """
    return len(svrl_failures(svrl)) > 0


def schematron_failures(
    filepath: Optional[str],
    stdin: Optional[bytes] = None,
    schema: str = PDS_SCHEMATRON_SCHEMA,
) -> Optional[str]:
    """
    Run probatron on the XML at the filepath (ignored if stdin is not
    None) or in stdin, validating against the schema.  Returns None if
    there are no failures; returns a string containing the failures if
    they exist.
    """
    svrl = probatron_with_svrl_result(filepath or "", stdin, schema)
    failures = svrl_failures(svrl)
    if len(failures) > 0:
        # should I have a pretty option here for human-readability?

        # TODO The replace probably ought to be done only in the
        # reporting, only as necessary
        return ("\n".join([f.toxml() for f in failures])).replace("\n", "\\n")
    else:
        return None


def verify_label_or_raise_fp(filepath: str) -> None:
    with closing(open(filepath, "rb")) as f:
        label: bytes = f.read()
    verify_label_or_raise(label)


def verify_label_or_raise(label: bytes) -> None:
    """
    Given the text of a PDS4 label, run XML Schema *and* Schematron
    validations on it.  Raise an exception on failures.
    """
    try:
        failures_from_xml_schema = xml_schema_failures(None, label)
        if failures_from_xml_schema is not None:
            raise Exception(
                f"XML schema validation errors: {str(failures_from_xml_schema)}"
            )
        failures_from_schematron = schematron_failures(None, label)
        if failures_from_schematron is not None:
            raise Exception(f"Schematron validation errors: {failures_from_schematron}")
    except Exception:
        # Debugging functionality: write the label to disk.
        PRINT_AND_SAVE_LABEL = True
        if PRINT_AND_SAVE_LABEL:
            import time

            print(label)
            t = int(time.time() * 1000)
            fp = f"tmp{t}.xml"
            with open(fp, "wb") as f:
                f.write(label)
        raise
