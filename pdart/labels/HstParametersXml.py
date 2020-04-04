"""
Templates to create a XML ``<hst:HST />`` node.
"""

from pdart.xml.Templates import interpret_document_template, interpret_template

from typing import Dict
from pdart.xml.Templates import DocTemplate, NodeBuilderTemplate

hst: NodeBuilderTemplate = interpret_template(
    """<hst:HST>
<NODE name="parameters_general"/>
<NODE name="parameters_instrument"/>
</hst:HST>"""
)
"""
An interpreted fragment template to create an ``<hst:HST />``
XML element.
"""

parameters_general: NodeBuilderTemplate = interpret_template(
    """<hst:Parameters_General>
  <hst:mast_observation_id><NODE name="mast_observation_id" /></hst:mast_observation_id>
  <hst:hst_proposal_id><NODE name="hst_proposal_id" /></hst:hst_proposal_id>
  <hst:hst_pi_name><NODE name="hst_pi_name" /></hst:hst_pi_name>
  <hst:exposure_duration unit="s"><NODE name="exposure_duration" />\
</hst:exposure_duration>
  <hst:hst_target_name><NODE name="hst_target_name" /></hst:hst_target_name>
  <hst:aperture_name><NODE name="aperture_name" /></hst:aperture_name>
  <hst:exposure_type><NODE name="exposure_type" /></hst:exposure_type>
  <hst:filter_name><NODE name="filter_name" /></hst:filter_name>
  <hst:fine_guidance_system_lock_type>\
<NODE name="fine_guidance_system_lock_type" />\
</hst:fine_guidance_system_lock_type>
  <hst:instrument_mode_id><NODE name="instrument_mode_id" />\
</hst:instrument_mode_id>
  <hst:moving_target_flag><NODE name="moving_target_flag" />\
</hst:moving_target_flag>
</hst:Parameters_General>"""
)
"""
An interpreted fragment template to create an ``<hst:Parameters_General />``
XML element.
"""

parameters_acs: NodeBuilderTemplate = interpret_template(
    """<hst:Parameters_ACS>
<hst:detector_id><NODE name="detector_id" /></hst:detector_id>
<hst:gain_mode_id><NODE name="gain_mode_id" /></hst:gain_mode_id>
<hst:observation_type><NODE name="observation_type" /></hst:observation_type>
<hst:repeat_exposure_count><NODE name="repeat_exposure_count" />\
</hst:repeat_exposure_count>
<hst:subarray_flag><NODE name="subarray_flag" /></hst:subarray_flag>
</hst:Parameters_ACS>"""
)
"""
An interpreted fragment template to create an ``<hst:Parameters_ACS />``
XML element.
"""

parameters_wfc3: NodeBuilderTemplate = interpret_template(
    """<hst:Parameters_WFC3>
<hst:detector_id><NODE name="detector_id" /></hst:detector_id>
<hst:observation_type><NODE name="observation_type" /></hst:observation_type>
<hst:repeat_exposure_count><NODE name="repeat_exposure_count" />\
</hst:repeat_exposure_count>
<hst:subarray_flag><NODE name="subarray_flag" /></hst:subarray_flag>
</hst:Parameters_WFC3>"""
)
"""
An interpreted fragment template to create an ``<hst:Parameters_WFC3 />``
XML element.
"""

parameters_wfpc2: NodeBuilderTemplate = interpret_template(
    """<hst:Parameters_WFPC2>
<hst:bandwidth unit="nm"><NODE name="bandwidth" /></hst:bandwidth>
<hst:center_filter_wavelength unit="nm"><NODE name="center_filter_wavelength" />\
</hst:center_filter_wavelength>
<hst:targeted_detector_id><NODE name="targeted_detector_id" />\
</hst:targeted_detector_id>
<hst:gain_mode_id><NODE name="gain_mode_id" /></hst:gain_mode_id>
</hst:Parameters_WFPC2>"""
)
"""
An interpreted fragment template to create an ``<hst:Parameters_WFPC2 />``
XML element.
"""

wrapper: DocTemplate = interpret_document_template("""<NODE name="wrapped" />""")


# Not XML, but placeholder code common to both the database- and the
# read-and-parse-FITS-file code.


def get_targeted_detector_id(aperture: str) -> str:
    """
    Return text for the ``<targeted_detector_id />`` XML element.
    """
    general_cases = {
        "PC1": "PC1",
        "WF2": "WF2",
        "WF3": "WF3",
        "WF4": "WF4",
        "ALL": "WF3",
        "W2": "WF2",
        "W3": "WF3",
        "W4": "WF4",
    }
    for k, v in list(general_cases.items()):
        if k in aperture:
            return v

    # quad or polarizing filters
    special_cases = {
        "FQUVN33": "WF2",
        "POLQN33": "WF2",
        "POLQN18": "WF2",
        "POLQP15P": "PC1",
        "POLQP15W": "WF2",
        "FQCH4N33": "WF2",
        "FQCH4N15": "PC1",
        "FQCH4P15": "PC1",
        "FQCH4N15": "WF3",
        "F160BN15": "WF3",
    }

    for k, v in list(special_cases.items()):
        if k in aperture:
            return v

    raise ValueError(f"get_targeted_detector_id({aperture!r})")
