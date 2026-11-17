##########################################################################################
# xml_support.py
#
# get_modification_history(xml_content)
#   return the content of the Modification_History from an XML label.
#
# get_target_identifications(xml_content)
#   return the contents of the Target_Identifications from an XML label.
#
# get_citation_information(xml_content)
#   return the contents of the Citation_Information from an XML label.
#
# get_time_coordinates(xml_content)
#   return the contents of the Citation_Information from an XML label.
##########################################################################################

import re

MODIFICATION_HISTORY = re.compile(r'</?Modification_History>')
MODIFICATION_DETAIL = re.compile(r'</?Modification_Detail>')
MODIFICATION_DATE = re.compile(r'</?modification_date>')
VERSION_ID = re.compile(r'</?version_id>')
DESCRIPTION = re.compile(r'</?description>')

def get_modification_history(xml_content):
    """Quickly retrieve the Modification_History from the content of an XML label.

    Input:
        xml_content     the full content of the XML label, as a single character string.

    Return:             a list of tuples (modification_date, version_id, description), one
                        for each Modification_Detail.
    """

    # Isolate the text between "<Modification_Detail>" and "</Modification_Detail>"
    parts = MODIFICATION_DETAIL.split(xml_content)
    texts = parts[1::2]

    # Extract the fields from each Modification_Detail object
    info_list = []
    for text in texts:

        parts = MODIFICATION_DATE.split(text)
        modification_date = parts[1].strip()

        # represent the version ID as a tuple of two ints
        parts = VERSION_ID.split(text)
        version_id = tuple([int(v) for v in parts[1].split('.')])

        parts = DESCRIPTION.split(text)
        description = parts[1].strip()

        info_list.append((modification_date, version_id, description))

    return info_list

TARGET_IDENTIFICATION = re.compile(r'</?Target_Identification>')
NAME = re.compile(r'</?name>')
ALTERNATE_DESIGNATION = re.compile(r'</?alternate_designation>')
TYPE = re.compile(r'</?type>')
LID_REFERENCE = re.compile(r'</?lid_reference>')

def get_target_identifications(xml_content):
    """Quickly retrieve the Target_Identification(s) from the content of an XML label.

    Input:
        xml_content     the full content of the XML label, as a single character string.

    Return:             a list of tuples (name, alternate_designations, type, description,
                        lid_reference)
        name                the preferred name;
        alt_designations    a list of strings indicating alternative names;
        body_type           "Asteroid", "Centaur", etc.;
        description         a list of strings, to be separated by newlines inside the
                            description attribute of the XML Target_Identification object;
        lid                 the LID of the object, omitting "urn:...:target:".
    """

    # Isolate the text between "<Target_Identification>" and "</Target_Identification>"
    parts = TARGET_IDENTIFICATION.split(xml_content)
    texts = parts[1::2]

    # Extract the fields from each Target_Identification object
    info_list = []
    for text in texts:
        info = {}

        parts = NAME.split(text)
        info['name'] = parts[1].strip()

        parts = ALTERNATE_DESIGNATION.split(text)
        alternate_designations = []
        for part in parts[1::2]:
            alternate_designations.append(part.strip())
        info['alternate_designations'] = alternate_designations

        parts = TYPE.split(text)
        info['type'] = parts[1].strip()

        parts = DESCRIPTION.split(text)
        if len(parts) == 1:
            info['description'] = ''
        else:
            info['description'] = parts[1]

        parts = LID_REFERENCE.split(text)[1].strip()
        _, _, lid = parts.partition('target:')
        info['lid'] = lid

        info_list.append(info)

    return info_list

CITATION_INFORMATION = re.compile(r'</?Citation_Information>')
AUTHOR_LIST = re.compile(r'</?author_list>')
EDITOR_LIST = re.compile(r'</?editor_list>')
PUBLICATION_YEAR = re.compile(r'</?publication_year>')
DOI = re.compile(r'</?doi>')
KEYWORD = re.compile(r'</?keyword>')

def get_citation_information(xml_content):
    """Quickly retrieve the Citation_Information from the content of an XML label.

    Input:
        xml_content     the full content of the XML label, as a single character string.

    Return:             a tuple (author_list, editor_list, publication_year, doi,
                        keywords, description)
    """

    # Isolate the text between "<Citation_Information>" and "</Citation_Information>"
    parts = CITATION_INFORMATION.split(xml_content)
    text = parts[1]

    # Extract the fields from each Citation_Information object

    parts = AUTHOR_LIST.split(text)
    author_list = parts[1].strip()
    author_list = ' '.join(author_list.split())

    parts = EDITOR_LIST.split(text)
    if len(parts) > 1:
        editor_list = parts[1].strip()
        editor_list = ' '.join(editor_list.split())
    else:
        editor_list = ''

    parts = PUBLICATION_YEAR.split(text)
    publication_year = parts[1].strip()

    parts = DOI.split(text)
    if len(parts) > 1:
        doi = parts[1].strip()
    else:
        doi = ''

    parts = KEYWORD.split(text)
    keywords = []
    for part in parts[1::2]:
        keywords.append(part.strip())

    parts = DESCRIPTION.split(text)
    description = parts[1].strip()
    description = ' '.join(description.split())
    description = description

    return (author_list, editor_list, publication_year, doi, keywords, description)

TIME_COORDINATES = re.compile(r'</?Time_Coordinates>')
START_DATE_TIME = re.compile(r'</?start_date_time>')
STOP_DATE_TIME = re.compile(r'</?stop_date_time>')

def get_time_coordinates(xml_content):
    """Quickly retrieve the Time_Coordinates from the content of an XML label.

    Input:
        xml_content     the full content of the XML label, as a single character string.

    Return:             a tuple (start_date_time, stop_date_time)
    """

    # Isolate the text between "<Time_Coordinates>" and "</Time_Coordinates>"
    parts = TIME_COORDINATES.split(xml_content)
    text = parts[1]

    # Extract the fields from each Time_Coordinates object
    parts = START_DATE_TIME.split(text)
    start_date_time = parts[1].strip()

    parts = STOP_DATE_TIME.split(text)
    stop_date_time = parts[1].strip()

    return (start_date_time, stop_date_time)

PRIMARY_RESULT_SUMMARY = re.compile(r'</?Primary_Result_Summary>')
PURPOSE = re.compile(r'</?purpose>')
PROCESSING_LEVEL = re.compile(r'</?processing_level>')
WAVELENGTH_RANGE = re.compile(r'</?wavelength_range>')
DOMAIN = re.compile(r'</?domain>')

def get_primary_result_summary(xml_content):
    """Quickly retrieve the Primary_Result_Summary from the content of an XML label.

    Input:
        xml_content     the full content of the XML label, as a single character string.

    Return:             a tuple (purposes, processing_levels, wavelength_ranges, domains)
    """

    # Isolate the text between "<Primary_Result_Summary>" and "</Primary_Result_Summary>"
    parts = PRIMARY_RESULT_SUMMARY.split(xml_content)
    text = parts[1]

    # Extract the fields
    purposes = [p.strip() for p in PURPOSE.split(text)[1::2]]
    processing_levels = [p.strip() for p in PROCESSING_LEVEL.split(text)[1::2]]
    wavelength_ranges = [p.strip() for p in WAVELENGTH_RANGE.split(text)[1::2]]
    domains = [p.strip() for p in DOMAIN.split(text)[1::2]]

    return (purposes, processing_levels, wavelength_ranges, domains)

INST_PARAMS = re.compile(r'</?hst:Instrument_Parameters>')
INST_ID = re.compile(r'</?hst:instrument_id>')
CHANNEL_ID = re.compile(r'</?hst:channel_id>')
DETECTOR_ID = re.compile(r'</?hst:detector_id>')
OBS_TYPE = re.compile(r'</?hst:observation_type>')

def get_instrument_params(xml_content):
    """Quickly retrieve the instrument params from the content of an XML label.

    Input:
        xml_content     the full content of the XML label, as a single character string.

    Return:             a tuple (inst_id, channel_id, detector_id, obs_type)
    """

    # Isolate the text between "<Time_Coordinates>" and "</Time_Coordinates>"
    parts = INST_PARAMS.split(xml_content)
    text = parts[1]

    # Extract the fields from each Time_Coordinates object
    parts = INST_ID.split(text)
    inst_id = parts[1].strip()

    parts = CHANNEL_ID.split(text)
    channel_id = parts[1].strip()

    parts = DETECTOR_ID.split(text)
    detector_id = parts[1].strip()

    parts = OBS_TYPE.split(text)
    obs_type = parts[1].strip()

    return (inst_id, channel_id, detector_id, obs_type)

# Probably not needed
# def labels_are_equivalent(new_content, old_content):
#     """Compare the content of two XML labels and return True if they are functionally
#     equivalent.
#
#     "Functionally equivalent" here means that all the content is the same, with the
#     possible exception of the XML header text, the version number, the modification
#     details, and any white space.
#
#     Input:
#         new_content     the full content of the new XML label.
#         old_content     the full content of the old XML label.
#     """
#
#     def to_words(content):
#         """Return the XML content as a list of words, skipping any ignored content.
#         """
#
#         # Ignore everything before the <Identification_Area>
#         content = content.partition('<Identification_Area>')[2]
#
#         # Strip out the Modification_History and the version_id
#         parts = MODIFICATION_HISTORY.split(content)
#         subparts = VERSION_ID.split(parts[0])
#         content = subparts[0] + subparts[2] + parts[2]
#
#         # Put a space around every "<", ">", and "="
#         content = content.replace('<', ' < ')
#         content = content.replace('>', ' > ')
#         content = content.replace('=', ' = ')
#
#         # Now reduce the text to a long list of individual words
#         return content.split()
#
#
#     new_words = to_words(new_content)
#     old_words = to_words(old_content)
#
#     return new_words == old_words

##########################################################################################
