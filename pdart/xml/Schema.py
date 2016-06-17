import os
import subprocess
import tempfile
import xml.dom.minidom


PDS_XML_SCHEMA = './xml/PDS4_PDS_1600.xsd.xml'
PDS_SCHEMATRON_SCHEMA = './xml/PDS4_PDS_1600.sch.xml'
HST_XML_SCHEMA = './xml/PDS4_HST_1600_0200.xsd.xml'
HST_SCHEMATRON_SCHEMA = './xml/PDS4_HST_1600_0200.sch.xml'


def run_subprocess(cmd, stdin=None):
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
    """
    Run probatron on the XML at the filepath validating against the
    schema.  Returns a triple of exit_code, stderr and stdout.
    """
    return run_subprocess(['java', '-jar', 'probatron.jar',
                           '-r0',  # output report as terse SVRL
                           '-n1',  # emit line/col numbers in report
                           filepath, schema])


def probatron_with_stdin(filepath, stdin=None, schema=PDS_SCHEMATRON_SCHEMA):
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
    return svrl.documentElement.getElementsByTagName('svrl:failed-assert')


def svrl_has_failures(svrl):
    return len(_svrl_failures(svrl)) > 0


def schematron_failures(filepath, stdin=None, schema=PDS_SCHEMATRON_SCHEMA):
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
