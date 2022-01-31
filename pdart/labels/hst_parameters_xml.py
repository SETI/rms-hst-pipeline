from pdart.xml.Templates import NodeBuilderTemplate, interpret_template

hst_parameters: NodeBuilderTemplate = interpret_template(
    """<hst:HST_Parameters>
  <NODE name="program_parameters"/>
  <NODE name="instrument_parameters"/>
  <NODE name="pointing_parameters"/>
  <NODE name="tracking_parameters"/>
  <NODE name="exposure_parameters"/>
  <NODE name="wavelength_filter_grating_parameters"/>
  <NODE name="operational_parameters"/>
</hst:HST_Parameters>"""
)

exposure_parameters: NodeBuilderTemplate = interpret_template(
    """<hst:Exposure_Parameters>
  <hst:exposure_duration unit="s"><NODE name="exposure_duration"/></hst:exposure_duration>
  <hst:exposure_type><NODE name="exposure_type"/></hst:exposure_type>
</hst:Exposure_Parameters>"""
)

detector_id: NodeBuilderTemplate = interpret_template(
    """<hst:detector_id><NODE name="detector_id"/></hst:detector_id>"""
)

instrument_parameters: NodeBuilderTemplate = interpret_template(
    """<hst:Instrument_Parameters>
  <hst:instrument_id><NODE name="instrument_id"/></hst:instrument_id>
  <hst:channel_id><NODE name="channel_id"/></hst:channel_id>
  <FRAGMENT name="detector_ids"/>
  <hst:observation_type><NODE name="observation_type"/></hst:observation_type>
</hst:Instrument_Parameters>"""
)

operational_parameters: NodeBuilderTemplate = interpret_template(
    """<hst:Operational_Parameters>
  <hst:instrument_mode_id><NODE name="instrument_mode_id"/></hst:instrument_mode_id>
  <hst:gain_setting><NODE name="gain_setting"/></hst:gain_setting>
  <hst:coronagraph_flag><NODE name="coronagraph_flag"/></hst:coronagraph_flag>
  <hst:cosmic_ray_split_count><NODE name="cosmic_ray_split_count"/>"""
    """</hst:cosmic_ray_split_count>
  <hst:repeat_exposure_count><NODE name="repeat_exposure_count"/>"""
    """</hst:repeat_exposure_count>
  <hst:subarray_flag><NODE name="subarray_flag"/></hst:subarray_flag>
  <hst:binning_mode><NODE name="binning_mode"/></hst:binning_mode>
  <hst:plate_scale><NODE name="plate_scale"/></hst:plate_scale>
</hst:Operational_Parameters>"""
)

moving_target_keyword: NodeBuilderTemplate = interpret_template(
    """<hst:moving_target_keywords><NODE name="moving_target_keyword"/>"""
    """</hst:moving_target_keywords>"""
)

moving_target_description: NodeBuilderTemplate = interpret_template(
    """<hst:moving_target_description><NODE name="moving_target_description"/>"""
    """</hst:moving_target_description>"""
)

targeted_detector_id: NodeBuilderTemplate = interpret_template(
    """<hst:targeted_detector_id><NODE name="targeted_detector_id"/>"""
    """</hst:targeted_detector_id>"""
)

pointing_parameters: NodeBuilderTemplate = interpret_template(
    """<hst:Pointing_Parameters>
  <hst:hst_target_name><NODE name="hst_target_name"/></hst:hst_target_name>
  <hst:moving_target_flag><NODE name="moving_target_flag"/></hst:moving_target_flag>
  <FRAGMENT name="moving_target_keywords"/>
  <FRAGMENT name="moving_target_descriptions"/>
  <hst:aperture_name><NODE name="aperture_name"/></hst:aperture_name>
  <hst:proposed_aperture_name><NODE name="proposed_aperture_name"/>"""
    """</hst:proposed_aperture_name>
  <FRAGMENT name="targeted_detector_ids"/>
</hst:Pointing_Parameters>"""
)

program_parameters: NodeBuilderTemplate = interpret_template(
    """<hst:Program_Parameters>
  <hst:mast_observation_id><NODE name="mast_observation_id"/></hst:mast_observation_id>
  <hst:hst_proposal_id><NODE name="hst_proposal_id"/></hst:hst_proposal_id>
  <hst:hst_pi_name><NODE name="hst_pi_name"/></hst:hst_pi_name>
</hst:Program_Parameters>"""
)

tracking_parameters: NodeBuilderTemplate = interpret_template(
    """<hst:Tracking_Parameters>
  <hst:fine_guidance_sensor_lock_type><NODE name="fine_guidance_sensor_lock_type"/>"""
    """</hst:fine_guidance_sensor_lock_type>
  <hst:gyroscope_mode><NODE name="gyroscope_mode"/></hst:gyroscope_mode>
</hst:Tracking_Parameters>"""
)

wavelength_filter_grating_parameters: NodeBuilderTemplate = interpret_template(
    """<hst:Wavelength_Filter_Grating_Parameters>
  <hst:filter_name><NODE name="filter_name"/></hst:filter_name>
  <hst:center_filter_wavelength unit="micrometer">"""
    """<NODE name="center_filter_wavelength"/></hst:center_filter_wavelength>
  <hst:bandwidth unit="micrometer"><NODE name="bandwidth"/></hst:bandwidth>
  <hst:spectral_resolution unit="micrometer"><NODE name="spectral_resolution"/>"""
    """</hst:spectral_resolution>
</hst:Wavelength_Filter_Grating_Parameters>"""
)
