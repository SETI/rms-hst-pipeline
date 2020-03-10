from pdart.xml.Schema import PDS_SCHEMATRON_SCHEMA, probatron, \
    probatron_with_stdin, probatron_with_svrl_result, run_subprocess, \
    schematron_failures, svrl_failures, svrl_has_failures, xml_schema_failures
from pdart.xml.Utils import path_to_testfile

import unittest


class TestXmlSchema(unittest.TestCase):
    # Exploratory testing: I want to use external programs that aren't
    # well documented as part of my code, so I'm using unittests to
    # explore the behavior of the system.
    def test_run_subprocess(self):
        # type: () -> None
        exit_code, stderr, stdout = run_subprocess('ls')
        self.assertEquals(0, exit_code)
        self.assertEquals('', stderr)
        self.assertNotEquals('', stdout)

        exit_code, stderr, stdout = run_subprocess(['echo', 'foo', 'bar'])
        self.assertEquals(0, exit_code)
        self.assertEquals('', stderr)
        self.assertEquals('foo bar\n', stdout)

        stdin = 'foo bar\nbaz\n'
        exit_code, stderr, stdout = run_subprocess('cat', stdin)
        self.assertEquals(0, exit_code)
        self.assertEquals('', stderr)
        self.assertEquals(stdin, stdout)

        exit_code, stderr, stdout = \
            run_subprocess(['cat', 'this_file_does_not_exist'])
        self.assertNotEquals(0, exit_code)
        self.assertNotEquals('', stderr)
        self.assertEquals('', stdout)

    def test_run_xml_schema(self):
        # type: () -> None
        # Correct XML, but doesn't match the schema
        failures = \
            xml_schema_failures('-',
                                stdin='<library><book/><book/></library>')
        self.assertIsNotNone(failures)

        # Valid XML according to the schema.
        failures = xml_schema_failures(path_to_testfile('bundle.xml'))
        self.assertIsNone(failures)

    def test_run_schematron(self):
        # type: () -> None
        exit_code, stderr, stdout = probatron(path_to_testfile('bundle.xml'))
        self.assertEquals(0, exit_code)
        self.assertEquals('', stderr)
        self.assertNotEquals('', stdout)

        exit_code, stderr, stdout = \
            probatron_with_stdin('<test_Schema.test_run_schematron>',
                                 stdin='<library><book/><book/></library>',
                                 schema=PDS_SCHEMATRON_SCHEMA)
        self.assertEquals(0, exit_code)
        self.assertEquals('', stderr)
        self.assertNotEquals('', stdout)

        svrl = probatron_with_svrl_result(path_to_testfile('bundle.xml'))
        failures = svrl_failures(svrl)
        self.assertFalse(failures, 
                         ('\n'.join([f.toxml()
                                     for f in failures])).replace('\n',
                                                                  '\\n'))

        svrl = probatron_with_svrl_result(path_to_testfile('bad_bundle.xml'))
        failures = svrl_failures(svrl)
        self.assertTrue(failures)

        self.assertIsNone(schematron_failures(path_to_testfile('bundle.xml')))
        failures = schematron_failures(path_to_testfile('bad_bundle.xml'))
        self.assertIsNotNone(failures)
