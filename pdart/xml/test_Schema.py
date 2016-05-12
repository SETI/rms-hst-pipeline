from pdart.xml.Schema import *

import unittest


class TestXmlSchema(unittest.TestCase):
    # Exploratory testing: I want to use external programs that aren't
    # well documented as part of my code, so I'm using unittests to
    # explore the behavior of the system.
    def test_run_subprocess(self):
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
        # Correct XML, but doesn't match the schema
        failures = \
            xml_schema_failures('-',
                                stdin='<library><book/><book/></library>')
        self.assertIsNotNone(failures)

        # Valid XML according to the schema.
        failures = xml_schema_failures('./testfiles/bundle.xml')
        self.assertIsNone(failures)

    def test_run_schematron(self):
        exit_code, stderr, stdout = probatron('./testfiles/bundle.xml')
        self.assertEquals(0, exit_code)
        self.assertEquals('', stderr)
        self.assertNotEquals('', stdout)

        exit_code, stderr, stdout = \
            probatron_with_stdin(None,
                                 stdin='<library><book/><book/></library>',
                                 schema=SCHEMATRON_SCHEMA)
        self.assertEquals(0, exit_code)
        self.assertEquals('', stderr)
        self.assertNotEquals('', stdout)

        svrl = probatron_with_svrl_result('./testfiles/bundle.xml')
        self.assertFalse(svrl_has_failures(svrl))

        svrl = probatron_with_svrl_result('./testfiles/bad_bundle.xml')
        self.assertTrue(svrl_has_failures(svrl))

        self.assertIsNone(schematron_failures('./testfiles/bundle.xml'))
        failures = schematron_failures('./testfiles/bad_bundle.xml')
        self.assertIsNotNone(failures)
