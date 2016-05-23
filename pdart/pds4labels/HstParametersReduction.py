from pdart.reductions.Reduction import *
from pdart.xml.Templates import *


hst_parameters = interpret_template("""<General_HST_Parameters>
<stsci_group_id><NODE name="stsci_group_id" /></stsci_group_id>
<hst_proposal_id><NODE name="hst_proposal_id" /></hst_proposal_id>
<hst_pi_name><NODE name="hst_pi_name" /></hst_pi_name>
<instrument_id><NODE name="instrument_id" /></instrument_id>
<detector_id><NODE name="detector_id" /></detector_id>
<observation_type><NODE name="observation_type" /></observation_type>
<product_type><NODE name="product_type" /></product_type>
<exposure_duration><NODE name="exposure_duration" /></exposure_duration>
<hst_target_name><NODE name="hst_target_name" /></hst_target_name>
<filter_name><NODE name="filter_name" /></filter_name>
<center_filter_wavelength><NODE name="center_filter_wavelength" />\
</center_filter_wavelength>
<bandwidth><NODE name="bandwidth" /></bandwidth>
<wavelength_resolution><NODE name="wavelength_resolution" />\
</wavelength_resolution>
<maximum_wavelength><NODE name="maximum_wavelength" /></maximum_wavelength>
<minimum_wavelength><NODE name="minimum_wavelength" /></minimum_wavelength>
<aperture_type><NODE name="aperture_type" /></aperture_type>
<exposure_type><NODE name="exposure_type" /></exposure_type>
<fine_guidance_system_lock_type><NODE name="fine_guidance_system_lock_type" />\
</fine_guidance_system_lock_type>
<gain_mode_id><NODE name="gain_mode_id" /></gain_mode_id>
<instrument_mode_id><NODE name="instrument_mode_id" /></instrument_mode_id>
<lines><NODE name="lines" /></lines>
<line_samples><NODE name="line_samples" /></line_samples>
</General_HST_Parameters>""")

wrapper = interpret_document_template("""<NODE name="wrapped" />""")


class HstParametersReduction(Reduction):
    def reduce_product(self, archive, lid, get_reduced_fits_files):
        # return String
        wrapped = get_reduced_fits_files()[0]
        return wrapper({'wrapped': wrapped}).toxml()

    def reduce_fits_file(self, file, get_reduced_hdus):
        # returns (Doc -> Node)
        return get_reduced_hdus()[0]

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        # returns (Doc -> Node) or None
        if n == 0:
            params = {
                'stsci_group_id': '### placeholder for stsci_group_id ###',
                'hst_proposal_id': '### placeholder for hst_proposal_id ###',
                'hst_pi_name': '### placeholder for hst_pi_name ###',
                'instrument_id': '### placeholder for instrument_id ###',
                'detector_id': '### placeholder for detector_id ###',
                'observation_type': '### placeholder for observation_type ###',
                'product_type': '### placeholder for product_type ###',
                'exposure_duration':
                    '### placeholder for exposure_duration ###',
                'hst_target_name': '### placeholder for hst_target_name ###',
                'filter_name': '### placeholder for filter_name ###',
                'center_filter_wavelength':
                    '### placeholder for center_filter_wavelength ###',
                'bandwidth': '### placeholder for bandwidth ###',
                'wavelength_resolution':
                    '### placeholder for wavelength_resolution ###',
                'maximum_wavelength':
                    '### placeholder for maximum_wavelength ###',
                'minimum_wavelength':
                    '### placeholder for minimum_wavelength ###',
                'aperture_type': '### placeholder for aperture_type ###',
                'exposure_type': '### placeholder for exposure_type ###',
                'fine_guidance_system_lock_type':
                    '### placeholder for fine_guidance_system_lock_type ###',
                'gain_mode_id': '### placeholder for gain_mode_id ###',
                'instrument_mode_id':
                    '### placeholder for instrument_mode_id ###',
                'lines': '### placeholder for lines ###',
                'line_samples': '### placeholder for line_samples ###'
                }
            return hst_parameters(params)
        else:
            return None
