from pdart.xml.Templates import *

_observing_system = interpret_template("""<Observing_System>
      <name><NODE name="name"/></name>
      <Observing_System_Component>
        <name>Hubble Space Telescope</name>
        <type>Spacecraft</type>
        <Internal_Reference>
          <lid_reference>urn:nasa:pds:context:instrument_host:spacecraft.hst</lid_reference>
          <reference_type>is_instrument_host</reference_type>
        </Internal_Reference>
      </Observing_System_Component>
      <Observing_System_Component>
        <name><NODE name="component_name"/></name>
        <type>Instrument</type>
        <Internal_Reference>
          <lid_reference>urn:nasa:pds:context:instrument:insthost.acs.\
<NODE name="abbreviation"/></lid_reference>
          <reference_type>is_instrument</reference_type>
        </Internal_Reference>
      </Observing_System_Component>
    </Observing_System>""")


acs_observing_system = _observing_system({
        'name': 'Hubble Space Telescope Advanced Camera for Surveys',
        'component_name': 'Advanced Camera for Surveys',
        'abbreviation': 'acs'
        })

wfc3_observing_system = _observing_system({
        'name': 'Hubble Space Telescope Wide Field Camera 3',
        'component_name': 'Wide Field Camera 3',
        'abbreviation': 'wfc3'
        })

wfpc2_observing_system = _observing_system({
        'name': 'Hubble Space Telescope Wide-Field Planetary Camera 2',
        'component_name': 'Wide-Field Planetary Camera 2',
        'abbreviation': 'wfpc2'
        })


def observing_system(instrument):
    return {'acs': acs_observing_system,
            'wfc3': wfc3_observing_system,
            'wfpc2': wfpc2_observing_system,
            }[instrument]
