#!/usr/bin/env python3
##########################################################################################
# target_cache/update_target_cache.py
##########################################################################################
"""
.. update_target_cache:

###################
update_target_cache
###################

This is a stand-alone program that can be used to manage a local cache of all the
currently defined target context products. Type::

    update_target_cache --help

for more information.
"""

import argparse
import os
import pathlib
import pickle
import re
import sys

import lxml.etree
import pdslogger
import requests

from remote_listdir import remote_listdir

TARGET_URL = 'https://pds.nasa.gov/data/pds4/context-pds4/target/'
TARGET_CACHE = pathlib.Path(os.path.dirname(__file__)) / 'CACHE'
TARGET_DICT_BASENAME = '$LOOKUP.pickle'

BASENAME_SPLITTER = re.compile(r'(.*)_(\d\.\d)(|_local)\.xml')
MINOR_PLANET_NUMBER = re.compile(r'.*?\((\d+)\).*')

# Set up parser
PARSER = argparse.ArgumentParser(
    description='Update the local cache of all the target context products, which is '
                'found inside the `target_cache/CACHE` subdirectory. It compares the '
                'local cache contents to what is at the Engineering Node and retrieves '
                'any new XML files that are missing from the cache. It also deletes any '
                'superseded or deprecated versions of context products from the local '
                'cache.',
    epilog = 'NOTE: to add or update a context product locally, create a properly named '
             'file but append the suffix "_local" after the version number. This file '
             'will be preserved in the cache until a file of the same name and version, '
             'but no suffix, appears at the Engineering Node.')

PARSER.add_argument('--debug', '-d', action='store_true',
                    help='see warnings about duplicated aliases.')

PARSER.add_argument('--rebuild', '-r', action='store_true',
                    help='rebuild the index even if the cache is up to date.')

PARSER.add_argument('--quiet', '-q', action='store_true',
                    help='Do not log to the terminal.')

PARSER.add_argument('--log', '-l', type=str,
                    help='Path to a log file, if any.')


def main():

    args = PARSER.parse_args()

    logger = pdslogger.PdsLogger('pds.update_target_cache',
                                 roots=[TARGET_CACHE], lognames=False, indent=False,
                                 timestamps=True, digits=3,
                                 level='debug' if args.debug else 'info')

    logger.add_handler(pdslogger.NULL_HANDLER)  # suppress automatic logging to stdout
    if not args.quiet:
        logger.add_handler(pdslogger.STDOUT_HANDLER)
    if args.log:
        logger.add_handler(pdslogger.file_handler(args.log, rotation='none'))

    _update_target_cache(logger=logger, rebuild=args.rebuild)


def _update_target_cache(logger=None, rebuild=False):
    """Update the target cache.

    Parameters:
        rebuild (bool, optional): True to rebuild the index even if no files have changed.
        logger (pdslogger.PdsLogger, optional): Logger to use.
    """

    # Check the local context products
    logger and logger.info(f'Checking local targets: {TARGET_CACHE}')
    local_basenames = set(p.name for p in TARGET_CACHE.iterdir()) - {TARGET_DICT_BASENAME}

    # Get the list of remote context products
    logger and logger.info(f'Checking remote targets: {TARGET_URL}')
    remote_info = remote_listdir(TARGET_URL, logger=logger, verbose=False)
    remote_basenames = set(t[0] for t in remote_info)

    # Delete deprecated local files
    sorted_basenames = list(local_basenames)
    sorted_basenames.sort()
    for basename in sorted_basenames:
        if basename.endswith('_local.xml'):
            continue
        if basename not in remote_basenames:
            if basename.endswith('.xml'):
                logger and logger.info(f'Deprecated file removed', basename)
            else:
                logger and logger.info(f'Extraneous file removed', basename)
            (TARGET_CACHE / basename).unlink()

    # Replace files updated remotely
    updates = 0
    remote_basenames = _latest_basenames(remote_basenames)
    remote_basenames.sort()
    for basename in remote_basenames:
        if basename in local_basenames:
            continue

        updates += 1

        # Retrieve remote content
        url = TARGET_URL + basename
        logger and logger.info(f'Retrieving', basename)
        request = requests.get(url, allow_redirects=True)
        if request.status_code != 200:
            logger and logger.error(f'Response {request.status_code} received', basename)
            continue

        # Remove old versions of this target from local cache
        # This will also remove the _local.xml file if the remote copy is the same version
        lid, vid, _ = BASENAME_SPLITTER.match(basename).groups()
        for old_version in TARGET_CACHE.glob(lid + '_*.xml'):
            _, old_vid, suffix = BASENAME_SPLITTER.match(old_version.name).groups()
            if old_vid > vid and suffix:    # if local update is still newer
                continue
            logger and logger.debug(f'Superseded file removed', old_version)
            old_version.unlink()

        # Save the local version to the cache
        (TARGET_CACHE / basename).write_bytes(request.content)

    # Summarize local updates
    for basename in sorted_basenames:
        if basename.endswith('_local.xml') and (TARGET_CACHE / basename).exists():
            logger and logger.info(f'Local update retained', basename)

    if not updates:
        logger and logger.info(f'Target cache is up to date')

    if not (updates or rebuild):
        logger and logger.blankline()
        return

    # Re-index...
    logger and logger.info(f'Rebuilding index', TARGET_DICT_BASENAME)

    # Determine which local copies still supersede the remote versions
    local_updates = {b for b in local_basenames if b.endswith('_local.xml')}
    local_updates = {b for b in local_updates if (TARGET_CACHE / b).exists()}
    # ^This filters out the local versions that were just superseded

    local_update_dict = {BASENAME_SPLITTER.match(b).group(1):b for b in local_updates}

    # lookup_by_name[title, alias, or lid] -> context file basename or list if multiple
    lookup_by_name = {}
    print(88888, local_update_dict)
    for basename in remote_basenames + list(local_updates):
        if 'moontwo' in basename:
            print(77777, basename)
        key, version, suffix = BASENAME_SPLITTER.match(basename).groups()
        basename = local_update_dict.get(key, basename)  # use "_local" basename if any
        if 'moontwo' in basename:
            print(77700, basename)

        tree = _get_etree(TARGET_CACHE / basename)
        title = tree.xpath('//title')[0].text
        alts = {node.text for node in tree.xpath('//alternate_title')}
        lid = tree.xpath('//logical_identifier')[0].text
        keys = _lookup_keys(title, alts, lid)
        if 'MoonTwo' in keys:
            print('===============')
        for key in keys:
            if key in lookup_by_name:
                other = lookup_by_name[key]
                if isinstance(other, list):
                    other.append(basename)
                    first = other[0]
                else:
                    lookup_by_name[key] = [other, basename]
                    first = other
                logger.debug(f'Duplicated target "{key}": {first}, {basename}')
            else:
                lookup_by_name[key] = basename

    # Include keys in lower case
    keys = list(lookup_by_name.keys())
    for key in keys:
        key_lower = key.lower()
        if key == key_lower:
            continue
        if key_lower in lookup_by_name:
            if lookup_by_name[key_lower] != lookup_by_name[key]:
                logger.warn(f'Case-sensitive lookup key "{key}"')
        else:
            lookup_by_name[key_lower] = lookup_by_name[key]

    # Include integer keys for minor planets
    lookup_by_number = {}
    for key in keys:
        basename = lookup_by_name[key]
        if isinstance(basename, str) and basename.startswith('satellite'):
            continue
        match = MINOR_PLANET_NUMBER.match(key)
        if match:
            number = int(match.group(1))
            if number in lookup_by_number:
                if lookup_by_number[number] != lookup_by_name[key]:
                    logger.warn(f'Duplicated integer key {number}: '
                                f'{lookup_by_name[key]}, {lookup_by_number[number]}')
            else:
                lookup_by_number[number] = lookup_by_name[key]

    # Save the dictionary as a pickle file
    dict_file = TARGET_CACHE / TARGET_DICT_BASENAME
    with dict_file.open('wb') as f:
        pickle.dump((lookup_by_name, lookup_by_number), f)

    logger and logger.info(f'Index rebuilt', TARGET_DICT_BASENAME)
    logger and logger.blankline()


def get_index():
    """Read and return the index as two dictionaries, lookups by name and by number."""

    # Save the dictionary as a pickle file
    dict_file = TARGET_CACHE / TARGET_DICT_BASENAME
    with dict_file.open('rb') as f:
        lookup_by_name, lookup_by_number = pickle.load(f)

    return (lookup_by_name, lookup_by_number)


def _latest_basenames(basenames):
    """Filter out all but the latest version of each target context file.

    Also remove any "collection_target" files, which can appear in the online directory.
    """

    # Remove "collection_target" files, deprecated files, and anything not ending in .xml
    basenames = [b for b in basenames if b.endswith('.xml')]
    basenames = [b for b in basenames if not b.endswith('_deprecated.xml')]
    basenames = [b for b in basenames if not b.startswith('collection_target')]
    basenames = [b for b in basenames if not b.startswith('Collection_target')]

    # Example basename: dwarf_planet.136108_haumea_1.2.xml
    version_dict = {}
    for basename in basenames:
        lid = BASENAME_SPLITTER.match(basename).group(1)
        version_dict.setdefault(lid, []).append(basename)

    latest = []
    for lid, version_list in version_dict.items():
        version_list.sort()
        latest.append(version_list[-1])

    return latest


MP_NUMBERED = re.compile(r'\((\d+)\) (19[6-9]\d|20[0-4]\d) ([A-Z][A-Z]?\d+)$')
MP_UNNUMBERED = re.compile(r'(19[6-9]\d|20[0-4]\d) ([A-Z][A-Z]?\d+)$')
MP_NAMED = re.compile(r'\((\d+)\) ([A-Z].*)')

COMET_PERIODIC = re.compile(rf'(\d+P)/([A-Z].*[^\d])(| \d+)(-[a-z]\d?|)$')
COMET_NUMBERED = re.compile(r'([A-Z]/(?:19[6-9]\d|20[0-4]\d) [A-Z][A-Z]?\d+)'
                            r'(-[a-z]\d?|)$')
COMET_NAMED = re.compile(COMET_NUMBERED.pattern[:-1] + r' \((.*)\)$')

_ROMAN = r'(?:CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})'
_NAME = r'[A-Z][a-z]+'
SATELLITE_FULL_NAME = re.compile(rf'(\(\d+\) |)({_NAME}) ({_ROMAN}) \(({_NAME})\)$')
SATELLITE_TEMPORARY = re.compile(r'S/(\d\d\d\d) ([JSUN]) (\d+)$')
COMET_FULL_NAME = re.compile(r'(\d+[A-Z])/

def _lookup_keys(title, alts, lid):

    lid_tail = lid.rsplit(':')[-1]
    category = lid_tail.split('.')[0]

    keys = {title, lid, lid_tail} | alts

    new_keys = set()
    if category in {'asteroid', 'centaur', 'comet', 'dwarf_planet',
                    'trans-neptunian_object'}:
        for key in keys:
            match = MP_NUMBERED.match(key):
            if match:
                (number, year, code) = match.groups()
                new_keys |= {f'{number} {year} {code}', f'{number} {year}{code}',
                             f'({number}) {year}{code}', f'{number} ({year} {code})',
                             f'{number} ({year}{code})', f'{year} {code}',
                             f'{year}{code}'}
                if code[1].isalpha():
                    new_keys.add(code)
                continue

            match = MP_UNNUMBERED.match(key)
            if match:
                (year, code) = match.groups()
                new_keys.add(year + code)
                if code[1].isalpha():
                    new_keys.add(code)

            match = MP_NAMED.match(key)
            if match:
                (number, name) = match.groups()
                new_keys.add(name)

    if category in {'comet', 'centaur'}:
        for key in keys:
            match = COMET_PERIODIC.match(key):
            if match:
                (code, name, number, fragment) = match.groups()
                new_keys |= {code, name, name + number, name + number + fragment,
                             code + name, code + name + number}
                if number == '':
                    new_keys |= {name + ' 1', name + ' 1' + fragment,
                                 code + name + ' 1', code + name + ' 1' + fragment}


        elif category == 'satellite':


2010199

    # Example: "(486958) 2014 MU69" -> "2014MU69", "MU69"
    for key in keys:
        match = MINOR_PLANET_CODE.match(key)
        if match:
            (year, code) = match.groups()
            new_keys.add(match.group(1))
            break

    if lid_tail.startswith('satellite'):

        # Examples: "(243) Ida I (Dactyl)"; "Saturn XVIII (Pan)"
        for key in keys:

            match = SATELLITE_FULL_NAME.match(key)
            if match:
                (number, parent, roman, name) = match.groups()
                parent_roman = f'{parent} {roman}'
                new_keys |= {number + parent_roman, parent_roman, name,
                             f'{parent_roman} ({name})'}
                if not number:
                    intval = _roman_to_int(roman)
                    new_keys |= {f'{parent[0]}{roman} ({name})', f'{parent[0]}{roman}',
                                 f'{parent[0]}{intval} ({name})', f'{parent[0]}{intval}'}
                break

        # Example: "S/1981 S 13"
        for key in keys:
            match = SATELLITE_TEMPORARY.match(key)
            if match:
                (year, planet, number) = match.groups()
                new_keys |= {f'S/{year}{planet}{number}', f'S{year}{planet}{number}',
                             f'S/{year} {planet}{number}', f'S{year} {planet}{number}'}
                break

    keys |= new_keys
    print(lid_tail, keys)
    return keys


XMLNS = re.compile(r'\s*(?:xmlns|xsi)(?:|:\w+)\s*=\s*"[^"]+"')

def _get_etree(xml_path):
    """The content of the given XML file as an etree; header and namespaces stripped."""

    content = pathlib.Path(xml_path).read_text()
    content = content.rpartition('?>')[-1].lstrip()
    content = ''.join(XMLNS.split(content))
    return lxml.etree.fromstring(content)


ROMAN_MAP = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

def _roman_to_int(s: str) -> int:  # from Google Gemini
    """Convert a Roman numeral string to its integer value."""

    # Iterate through the string up to the second-to-last character
    total = 0
    for i in range(len(s) - 1):
        if ROMAN_MAP[s[i]] < ROMAN_MAP[s[i+1]]:
            total -= ROMAN_MAP[s[i]]
        else:
            total += ROMAN_MAP[s[i]]

    # Add the value of the last character, which is always added
    total += ROMAN_MAP[s[-1]]
    return total

############################################

if __name__ == '__main__':
    main()

##########################################################################################
