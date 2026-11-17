##########################################################################################
# target_support.py
#
# Target identification for HST observations, delegated to the rms-hst-targets package.
#
# This module replaces the former hst_pipeline/target_identifications/ subpackage. It
# keeps that subpackage's public contract -- hst_target_identifications() with the same
# signature and the same tuple return -- so the label-building code above it is unchanged,
# while the identification itself is now done by rms-hst-targets.
##########################################################################################

import pdslogger

from targets import identify_target_dicts
# _complete_target is the step that fills in the PDS fields (title, lid, type_name,
# alt_titles, description) that identify_target_dicts leaves out; identify_targets()
# applies it before rendering a context product. It is private to rms-hst-targets, so if
# that package ever promotes it, switch to the public name.
from targets.target_xml_support import _complete_target

__all__ = ['hst_target_identifications']


def hst_target_identifications(spt_header0, filepath, logger=None):
    """Return a list of target identifications as tuples of the form:
          (name, alt_designations, type, description, lid)
    where:
          name                the preferred name
          alt_designations    a list of strings indicating alternative names.
          type                "Planet", "Satellite", etc.
          description         a list of strings, to be separated by newlines
                              inside the XML Target_Identification object.
          lid                 the full LID of the object.

    Note that the old subpackage put a standard body's NAIF ID in `description`
    ("NAIF ID: 799"); rms-hst-targets carries it as an alternate title ("NAIF ID 799"),
    which is where PDS4 puts it, so it now arrives in `alt_designations` instead.

    Inputs:
        spt_header0    the primary header of the SPT/SHF support file, as an
                       astropy.io.fits.Header or a plain dictionary.
        filepath       path to the SPT file, used only to identify it in log messages.
        logger         pdslogger to use; None for a NullLogger.

    Raises:
        ValueError     if no target can be identified, or the observation is not
                       planetary. rms-hst-targets raises TargetIdentificationFailure and
                       NotPlanetaryError, both subclasses of ValueError, so callers
                       already handling ValueError need no change.
    """

    logger = logger or pdslogger.NullLogger()

    targets = identify_target_dicts([spt_header0], logger=logger)
    records = [_record_from_target(_complete_target(target)) for target in targets]

    logger.debug(f'Identified {len(records)} target(s) in {filepath}')
    return records


def _record_from_target(target):
    """Convert a completed rms-hst-targets body dictionary to a pipeline tuple.

    Inputs:
        target    a body dictionary that has been through _complete_target(), so it
                  carries 'title', 'alt_titles', 'type_name', 'description', and 'lid'.

    Returns:
        the (name, alt_designations, type, description, lid) tuple used throughout the
        label-building code.
    """

    return (target['title'],
            list(target['alt_titles']),
            target['type_name'],
            list(target['description']),
            target['lid'])

##########################################################################################
