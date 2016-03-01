import XmlUtils


class InstrumentXmlMaker(XmlUtils.XmlUtils):
    """
    A class to build an Observing_System node in a PDS4 product label.
    """

    def __init__(self, document, parent, info, instrument):
        """
        Create the XML corresponding to an Observing_System node,
        given the XML document, the parent node to which the new XML
        will be added, and the instrument name.
        """
        assert document
        assert parent
        self.parent = parent
        assert info
        self.info = info
        super(InstrumentXmlMaker, self).__init__(document)
        self.create_default_xml(instrument)

    def create_default_xml(self, instrument):
        # At XPath 'Observing_System'
        observing_system = self.create_child(self.parent, 'Observing_System')
        name, observing_system_component_hst, \
            observing_system_component_acs = \
            self.create_children(observing_system,
                                 ['name',
                                  'Observing_System_Component',
                                  'Observing_System_Component'])
        if instrument == 'acs':
            self.set_text(name,
                          'Hubble Space Telescope Advanced Camera for Surveys')

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
                          'urn:nasa:pds:context:investigation:mission.hst')
            self.set_text(reference_type, 'is_instrument_host')

            # At XPath
            # 'Observing_System/Observing_System_Component[1]'
            name, type, internal_reference = \
                self.create_children(observing_system_component_acs,
                                     ['name', 'type', 'Internal_Reference'])
            self.set_text(name, 'Advanced Camera for Surveys')
            self.set_text(type, 'Instrument')

            # At XPath
            # 'Observing_System/Observing_System_Component[1]/Internal_Reference'
            lid_reference, reference_type = \
                self.create_children(internal_reference,
                                     ['lid_reference', 'reference_type'])
            self.set_text(lid_reference,
                          'urn:nasa:pds:context:investigation:mission.hst_acs')
            self.set_text(reference_type, 'is_instrument')

        else:
            # At XPath 'Observing_System/name'
            self.set_text(name, self.info.observing_system_name())

            # At XPath
            # 'Observing_System/Observing_System_Component'
            name, type = self.create_children(observing_system_component_hst,
                                              ['name', 'type'])
            self.set_text(name, self.info.observing_system_component_name())
            self.set_text(type, self.info.observing_system_component_type())

            name, type = self.create_children(observing_system_component_acs,
                                              ['name', 'type'])
            self.set_text(name, self.info.observing_system_component_name())
            self.set_text(type, self.info.observing_system_component_type())
