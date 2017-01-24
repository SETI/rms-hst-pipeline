"""
Templates to create a XML ``<hst:HST />`` node.
"""
from pdart.pds4labels.Placeholders import *
from pdart.xml.Templates import *


hst = interpret_template("""<hst:HST>
<NODE name="parameters_general"/>
<NODE name="parameters_instrument"/>
</hst:HST>""")
"""
An interpreted fragment template to create an ``<hst:HST />``
XML element.
"""
# type: NodeBuilderTemplate


parameters_general = interpret_template("""<hst:Parameters_General>
  <hst:stsci_group_id><NODE name="stsci_group_id" /></hst:stsci_group_id>
  <hst:hst_proposal_id><NODE name="hst_proposal_id" /></hst:hst_proposal_id>
  <hst:hst_pi_name><NODE name="hst_pi_name" /></hst:hst_pi_name>
  <hst:hst_target_name><NODE name="hst_target_name" /></hst:hst_target_name>
  <hst:aperture_type><NODE name="aperture_type" /></hst:aperture_type>
  <hst:exposure_duration><NODE name="exposure_duration" />\
</hst:exposure_duration>
  <hst:exposure_type><NODE name="exposure_type" /></hst:exposure_type>
  <hst:filter_name><NODE name="filter_name" /></hst:filter_name>
  <hst:fine_guidance_system_lock_type>\
<NODE name="fine_guidance_system_lock_type" />\
</hst:fine_guidance_system_lock_type>
  <hst:gyroscope_mode><NODE name="gyroscope_mode" /></hst:gyroscope_mode>
  <hst:instrument_mode_id><NODE name="instrument_mode_id" />\
</hst:instrument_mode_id>
  <hst:moving_target_flag><NODE name="moving_target_flag" />\
</hst:moving_target_flag>
</hst:Parameters_General>""")
"""
An interpreted fragment template to create an ``<hst:Parameters_General />``
XML element.
"""
# type: Dict[str, Any] -> NodeBuilder

parameters_acs = interpret_template("""<hst:Parameters_ACS>
<hst:detector_id><NODE name="detector_id" /></hst:detector_id>
<hst:gain_mode_id><NODE name="gain_mode_id" /></hst:gain_mode_id>
<hst:observation_type><NODE name="observation_type" /></hst:observation_type>
<hst:repeat_exposure_count><NODE name="repeat_exposure_count" />\
</hst:repeat_exposure_count>
<hst:subarray_flag><NODE name="subarray_flag" /></hst:subarray_flag>
</hst:Parameters_ACS>""")
"""
An interpreted fragment template to create an ``<hst:Parameters_ACS />``
XML element.
"""
# type: Dict[str, Any] -> NodeBuilder

parameters_wfc3 = interpret_template("""<hst:Parameters_WFC3>
<hst:detector_id><NODE name="detector_id" /></hst:detector_id>
<hst:observation_type><NODE name="observation_type" /></hst:observation_type>
<hst:repeat_exposure_count><NODE name="repeat_exposure_count" />\
</hst:repeat_exposure_count>
<hst:subarray_flag><NODE name="subarray_flag" /></hst:subarray_flag>
</hst:Parameters_WFC3>""")
"""
An interpreted fragment template to create an ``<hst:Parameters_WFC3 />``
XML element.
"""
# type: Dict[str, Any] -> NodeBuilder

parameters_wfpc2 = interpret_template("""<hst:Parameters_WFPC2>
<hst:bandwidth><NODE name="bandwidth" /></hst:bandwidth>
<hst:center_filter_wavelength><NODE name="center_filter_wavelength" />\
</hst:center_filter_wavelength>
<hst:targeted_detector_id><NODE name="targeted_detector_id" />\
</hst:targeted_detector_id>
<hst:gain_mode_id><NODE name="gain_mode_id" /></hst:gain_mode_id>
<hst:pc1_flag><NODE name="pc1_flag" /></hst:pc1_flag>
<hst:wf2_flag><NODE name="wf2_flag" /></hst:wf2_flag>
<hst:wf3_flag><NODE name="wf3_flag" /></hst:wf3_flag>
<hst:wf4_flag><NODE name="wf4_flag" /></hst:wf4_flag>
</hst:Parameters_WFPC2>""")
"""
An interpreted fragment template to create an ``<hst:Parameters_WFPC2 />``
XML element.
"""
# type: NodeBuilderTemplate

wrapper = interpret_document_template("""<NODE name="wrapped" />""")
# type: DocTemplate


# Not XML, but placeholder code common to both the database- and the
# read-and-parse-FITS-file code.

def get_targeted_detector_id(product_id, instrument, header):
    # type: (unicode, unicode, unicode) -> unicode
    """
    Return placeholder text for the ``<targeted_detector_id />`` XML
    element.
    """
    return placeholder(product_id, 'targeted_detector_id')


def get_pc1_flag(product_id, instrument, header):
    # type: (unicode, unicode, unicode) -> unicode
    """
    Return a placeholder integer for the ``<pc1_flag />`` XML element,
    noting the problem.
    """
    return placeholder_int(product_id, 'pc1_flag')


def get_wf2_flag(product_id, instrument, header):
    # type: (unicode, unicode, unicode) -> unicode
    """
    Return a placeholder integer for the ``<wf2_flag />`` XML element,
    noting the problem.
    """
    return placeholder_int(product_id, 'wf2_flag')


def get_wf3_flag(product_id, instrument, header):
    # type: (unicode, unicode, unicode) -> unicode
    """
    Return a placeholder integer for the ``<wf3_flag />`` XML element,
    noting the problem.
    """
    return placeholder_int(product_id, 'wf3_flag')


def get_wf4_flag(product_id, instrument, header):
    # type: (unicode, unicode, unicode) -> unicode
    """
    Return a placeholder integer for the ``<wf4_flag />`` XML element,
    noting the problem.
    """
    return placeholder_int(product_id, 'wf4_flag')
