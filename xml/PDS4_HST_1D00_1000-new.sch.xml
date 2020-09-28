<?xml version="1.0" encoding="UTF-8"?>
  <!-- PDS4 Schematron for Name Space Id:hst  Version:1.0.0.0 - Tue Aug 25 11:55:45 EDT 2020 -->
  <!-- Generated from the PDS4 Information Model Version 1.13.0.0 - System Build 10a -->
  <!-- *** This PDS4 schematron file is an operational deliverable. *** -->
<sch:schema xmlns:sch="http://purl.oclc.org/dsdl/schematron" queryBinding="xslt2">

  <sch:title>Schematron using XPath 2.0</sch:title>

  <sch:ns uri="http://www.w3.org/2001/XMLSchema-instance" prefix="xsi"/>
  <sch:ns uri="http://pds.nasa.gov/pds4/pds/v1" prefix="pds"/>
  <sch:ns uri="http://pds.nasa.gov/pds4/mission/hst/v1" prefix="hst"/>

		   <!-- ================================================ -->
		   <!-- NOTE:  There are two types of schematron rules.  -->
		   <!--        One type includes rules written for       -->
		   <!--        specific situations. The other type are   -->
		   <!--        generated to validate enumerated value    -->
		   <!--        lists. These two types of rules have been -->
		   <!--        merged together in the rules below.       -->
		   <!-- ================================================ -->
  <sch:pattern>
    <sch:rule context="hst:Exposure_Parameters/hst:exposure_duration">
      <sch:assert test="@unit = ('day', 'hr', 'julian day', 'microseconds', 'min', 'ms', 's', 'yr')">
        The attribute @unit must be equal to one of the following values 'day', 'hr', 'julian day', 'microseconds', 'min', 'ms', 's', 'yr'.</sch:assert>
    </sch:rule>
  </sch:pattern>
  <sch:pattern>
    <sch:rule context="hst:Instrument_Parameters/hst:channel_id">
      <sch:assert test=". = ('CCD', 'FOC', 'FOS', 'FUV', 'FUV-MAMA', 'GHRS', 'HRC', 'HSP', 'IR', 'NIC1', 'NIC2', 'NIC3', 'NUV', 'NUV-MAMA', 'PC', 'SBC', 'UVIS', 'WFC', 'WFPC2')">
        The attribute hst:channel_id must be equal to one of the following values 'CCD', 'FOC', 'FOS', 'FUV', 'FUV-MAMA', 'GHRS', 'HRC', 'HSP', 'IR', 'NIC1', 'NIC2', 'NIC3', 'NUV', 'NUV-MAMA', 'PC', 'SBC', 'UVIS', 'WFC', 'WFPC2'.</sch:assert>
    </sch:rule>
  </sch:pattern>
  <sch:pattern>
    <sch:rule context="hst:Instrument_Parameters/hst:detector_id">
      <sch:assert test=". = ('AMBER', 'BLUE', 'CCD', 'FOC', 'FUV', 'FUV-MAMA', 'GHRS1', 'GHRS2', 'HRC', 'IR', 'NIC1', 'NIC2', 'NIC3', 'NUV', 'NUV-MAMA', 'PC1', 'PC5', 'PC6', 'PC7', 'PC8', 'PMT', 'POL', 'SBC', 'UV1', 'UV2', 'UVIS1', 'UVIS2', 'VIS', 'WF1', 'WF2', 'WF3', 'WF4', 'WFC1', 'WFC2')">
        The attribute hst:detector_id must be equal to one of the following values 'AMBER', 'BLUE', 'CCD', 'FOC', 'FUV', 'FUV-MAMA', 'GHRS1', 'GHRS2', 'HRC', 'IR', 'NIC1', 'NIC2', 'NIC3', 'NUV', 'NUV-MAMA', 'PC1', 'PC5', 'PC6', 'PC7', 'PC8', 'PMT', 'POL', 'SBC', 'UV1', 'UV2', 'UVIS1', 'UVIS2', 'VIS', 'WF1', 'WF2', 'WF3', 'WF4', 'WFC1', 'WFC2'.</sch:assert>
    </sch:rule>
  </sch:pattern>
  <sch:pattern>
    <sch:rule context="hst:Instrument_Parameters/hst:instrument_id">
      <sch:assert test=". = ('ACS', 'COS', 'FOC', 'FOS', 'GHRS', 'HSP', 'NICMOS', 'STIS', 'WF/PC', 'WFC3', 'WFPC2')">
        The attribute hst:instrument_id must be equal to one of the following values 'ACS', 'COS', 'FOC', 'FOS', 'GHRS', 'HSP', 'NICMOS', 'STIS', 'WF/PC', 'WFC3', 'WFPC2'.</sch:assert>
    </sch:rule>
  </sch:pattern>
  <sch:pattern>
    <sch:rule context="hst:Instrument_Parameters/hst:observation_type">
      <sch:assert test=". = ('IMAGING', 'SPECTROGRAPHIC', 'TIME-SERIES')">
        The attribute hst:observation_type must be equal to one of the following values 'IMAGING', 'SPECTROGRAPHIC', 'TIME-SERIES'.</sch:assert>
    </sch:rule>
  </sch:pattern>
  <sch:pattern>
    <sch:rule context="hst:Wavelength_Filter_Grating_Parameters/hst:bandwidth">
      <sch:assert test="@unit = ('AU', 'Angstrom', 'cm', 'km', 'm', 'micrometer', 'mm', 'nm')">
        The attribute @unit must be equal to one of the following values 'AU', 'Angstrom', 'cm', 'km', 'm', 'micrometer', 'mm', 'nm'.</sch:assert>
    </sch:rule>
  </sch:pattern>
  <sch:pattern>
    <sch:rule context="hst:Wavelength_Filter_Grating_Parameters/hst:center_filter_wavelength">
      <sch:assert test="@unit = ('AU', 'Angstrom', 'cm', 'km', 'm', 'micrometer', 'mm', 'nm')">
        The attribute @unit must be equal to one of the following values 'AU', 'Angstrom', 'cm', 'km', 'm', 'micrometer', 'mm', 'nm'.</sch:assert>
    </sch:rule>
  </sch:pattern>
  <sch:pattern>
    <sch:rule context="hst:Wavelength_Filter_Grating_Parameters/hst:spectral_resolution">
      <sch:assert test="@unit = ('AU', 'Angstrom', 'cm', 'km', 'm', 'micrometer', 'mm', 'nm')">
        The attribute @unit must be equal to one of the following values 'AU', 'Angstrom', 'cm', 'km', 'm', 'micrometer', 'mm', 'nm'.</sch:assert>
    </sch:rule>
  </sch:pattern>
</sch:schema>
