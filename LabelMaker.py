import abc
import os
import os.path
import xml.dom

import ArchiveComponent
import Info
import XmlSchema
import XmlUtils


class LabelMaker(XmlUtils.XmlUtils):
    """
    An abstract class of objects that can build PDS4 labels for
    ArchiveComponents.
    """

    def __init__(self, component, info):
        """
        Create the label for an ArchiveComponent with the help of a
        matching Info object to provide values for fields.
        """
        assert isinstance(component, ArchiveComponent.ArchiveComponent)
        self.component = component
        assert isinstance(info, Info.Info)
        self.info = info
        document = xml.dom.getDOMImplementation().createDocument(None,
                                                                 None,
                                                                 None)
        super(LabelMaker, self).__init__(document)

        self.create_default_xml()

    @abc.abstractmethod
    def create_default_xml(self):
        """Create the XML label for the component."""
        pass

    @abc.abstractmethod
    def default_xml_name(self):
        """The default name for the XML label for this type of component."""
        pass

    def create_default_xml_file(self, xml_filepath=None):
        """
        Pretty-print the XML document to a file.  If no filepath is
        given, it will be written to the default label filepath for
        the component.
        """
        if xml_filepath is None:
            xml_name = self.default_xml_name()
            xml_filepath = os.path.join(self.component.directory_filepath(),
                                        xml_name)
        with open(xml_filepath, 'w') as output_file:
            pretty = True
            if pretty:
                output_file.write(self.document.toprettyxml(indent='  ',
                                                            newl='\n',
                                                            encoding='utf-8'))
            else:
                output_file.write(self.document.toxml(encoding='utf-8'))


def xml_schema_check(filepath):
    """
    Test the XML label at the filepath against the PDS4 v1.5 XML
    schema, returning true iff it passes.
    """

    failures = XmlSchema.xml_schema_failures(filepath)
    return failures is None

    # xmllint breaks the Unix Rule of Silence and says 'foo.xml
    # verifies' when the command succeeds instead of remaining silent.
    # To suppress (possible hundreds of thousands of) these messages
    # without suppressing real error messages, we perform the
    # following incantation in Old Enochian to the tentacled horrific
    # Elder Ones to suppress only lines including '.xml verifies' only
    # in stderr. Ph'nglui mglw'nafh Cthulhu R'lyeh wgah'nagl fhtagn.
    cmd_template = "xmllint --noout --schema %s %s 3>&1 1>&2 2>&3 3>&- | " + \
        "sed '/\\.xml validates/d'"
    exit_code = os.system(cmd_template %
                          ('./xml/PDS4_PDS_1500.xsd.xml', filepath))
    return exit_code == 0


def schematron_check(filepath):
    """
    Test the XML label at the filepath against the PDS4 v1.5
    Schematron schema, returning true iff it passes.
    """
    cmd_template = 'java -jar probatron.jar %s %s' + \
        ' | xmllint -format -' + \
        ' | python Schematron.py'
    exit_code = os.system(cmd_template %
                          (filepath, './xml/PDS4_PDS_1500.sch.xml'))
    return exit_code == 0
