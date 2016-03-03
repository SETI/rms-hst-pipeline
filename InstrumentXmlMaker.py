import abc

import XmlUtils


class InstrumentXmlMaker(XmlUtils.XmlUtils):
    """
    A class to build an Observing_System node in a PDS4 product label.
    """
    def __init__(self, document, parent):
        """
        Create the XML corresponding to an Observing_System node,
        given the XML document and the parent node to which the new
        XML will be added.
        """
        assert document
        assert parent
        self.parent = parent
        super(InstrumentXmlMaker, self).__init__(document)
        self.create_default_xml()

    @abc.abstractmethod
    def observing_system_name(self):
        pass

    @abc.abstractmethod
    def observing_system_component_name(self):
        pass

    @abc.abstractmethod
    def observing_system_component_lid(self):
        pass

    def create_default_xml(self):
        # At XPath 'Observing_System'
        observing_system = self.create_child(self.parent, 'Observing_System')
        name, observing_system_component_hst, \
            observing_system_component_inst = \
            self.create_children(observing_system,
                                 ['name',
                                  'Observing_System_Component',
                                  'Observing_System_Component'])
        self.set_text(name, self.observing_system_name())

        # At XPath
        # 'Observing_System/Observing_System_Component[0]'
        name, type, internal_reference = \
            self.create_children(observing_system_component_hst,
                                 ['name', 'type', 'Internal_Reference'])
        self.set_text(name, 'Hubble Space Telescope')
        self.set_text(type, 'Spacecraft')

        # At XPath
        # 'Observing_System/Observing_System_Component[0]/Internal_Reference'
        lid_reference, reference_type = \
            self.create_children(internal_reference,
                                 ['lid_reference', 'reference_type'])
        self.set_text(lid_reference,
                      'urn:nasa:pds:context:instrument_host:spacecraft.hst')
        self.set_text(reference_type, 'is_instrument_host')

        # At XPath
        # 'Observing_System/Observing_System_Component[1]'
        name, type, internal_reference = \
            self.create_children(observing_system_component_inst,
                                 ['name', 'type', 'Internal_Reference'])
        self.set_text(name, self.observing_system_component_name())
        self.set_text(type, 'Instrument')

        # At XPath
        # 'Observing_System/Observing_System_Component[1]/Internal_Reference'
        lid_reference, reference_type = \
            self.create_children(internal_reference,
                                 ['lid_reference', 'reference_type'])
        self.set_text(lid_reference,
                      self.observing_system_component_lid())
        self.set_text(reference_type, 'is_instrument')


class AcsXmlMaker(InstrumentXmlMaker):
    def __init__(self, document, parent):
        super(AcsXmlMaker, self).__init__(document, parent)

    def observing_system_name(self):
        return 'Hubble Space Telescope Advanced Camera for Surveys'

    def observing_system_component_name(self):
        return 'Advanced Camera for Surveys'

    def observing_system_component_lid(self):
        return 'urn:nasa:pds:context:instrument:insthost.acs.hst'


class Wfc3XmlMaker(InstrumentXmlMaker):
    def __init__(self, document, parent):
        super(Wfc3XmlMaker, self).__init__(document, parent)

    def observing_system_name(self):
        return 'Hubble Space Telescope Wide Field Camera 3'

    def observing_system_component_name(self):
        return 'Wide Field Camera 3'

    def observing_system_component_lid(self):
        return 'urn:nasa:pds:context:instrument:insthost.wfc3.hst'


class Wfpc2XmlMaker(InstrumentXmlMaker):
    def __init__(self, document, parent):
        super(Wfpc2XmlMaker, self).__init__(document, parent)

    def observing_system_name(self):
        return 'Hubble Space Telescope Wide-Field Planetary Camera 2'

    def observing_system_component_name(self):
        return 'Wide-Field Planetary Camera 2'

    def observing_system_component_lid(self):
        return 'urn:nasa:pds:context:instrument:insthost.wfpc2.hst'
