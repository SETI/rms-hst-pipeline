################################################################################
# fix_title.py
################################################################################

import re
from typing import Any, List

DEBUG = False

# Replacements for known weirdness in titles
#
# These are applied to the title string before any further modifications, so
# they must match the exact capitlization of the title. The changes can be
# applied to the interiors of words, so be careful about that.
TRANSLATIONS = [
    (r'\\cotwo\\', 'CO₂'),
    ('Mid?IR', 'Mid-IR'),
    ('WISE?selected', 'WISE-selected'),
    ('Mrk~', 'Mrk '),
    ('NGC~', 'NGC '),
    ('*.t23', ''),
    ('H_2', 'H₂'),
    (r'\\h2', 'H₂'),
    ('H_o', 'H₀'),
    ('H_0', 'H₀'),
    ('C_2', 'C₂'),
    ('CH_2', 'CH₂'),
    ('M_BH', 'Mbh'),
    (r'\?', '?'),
    (r'\PKS', 'PKS'),
    (r'\apl', '~'),
    (r'\{', '{'),
    (r'\}', '}'),
    ('*all*', 'all'),
    ('*best*', 'best'),
    ('*always*', 'always'),
    ('SGRA', 'SGR A'),
    ('^10B', '¹⁰B'),
    ('^11B', '¹¹B'),
    ('^12C', '¹²C'),
    ('^13C', '¹³C'),
    ('^1 ', '¹ '),
    ('^2 ', '² '),
    ('^3 ', '³ '),
    ('^4 ', '⁴ '),
    ('^5 ', '⁵ '),
    ('^6 ', '⁶ '),
    ('^7 ', '⁷ '),
    ('^8 ', '⁸ '),
    ('^9 ', '⁹ '),
    ('^-1 ', '⁻¹ '),
    ('^-2 ', '⁻² '),
    ('^-3 ', '⁻³ '),
    ('^-4 ', '⁻⁴ '),
    ('^-5 ', '⁻⁵ '),
    ('^-6 ', '⁻⁶ '),
    ('^-7 ', '⁻⁷ '),
    ('^-8 ', '⁻⁸ '),
    ('^-9 ', '⁻⁹ '),
    ('^-3)', '⁻³)'),
    ('>=', '≥'),
    ('<=', '≤'),
    ('>~', '≳'),
    ('<~', '≲'),
    ('1998_WW31', '1998 WW31'),
    (r'\\ ', ' '),
    (r'\ ', ' '),
    ('_', '-'),
    ('MGII ', 'MgII '),
    ('SDO ', 'sdO '),
    ('FEII', 'FeII'),
    ('HEII', 'HeII'),
    ('SIVI', 'SiVI'),
    ('QSOS', 'QSOs'),
    ('YSOS', 'YSOs'),
    ('SNRS', 'SNRs'),
    ('CM DRA', 'CM Dra'),
    ('MSPS', 'MSPs'),
    ('RAQR', 'RAqr'),
    ('AGNS', 'AGNs'),
]

# These are words that should always be fully capitalized when they appear in a
# title. This only affects isolated words of more than two letters. For the
# change to be applied, the word must appear in lower case here.
ALLCAPS_WORDS = [
    'afm',
    'agb',
    'agn',
    'ast',
    'bal',
    'ccd',
    'civ',
    'cno',
    'cos',
    'costar',
    'cte',
    'dob',
    'dqe',
    'euv',
    'fgs',
    'fhst',
    'foc',
    'fos',
    'fsc',
    'fsm',
    'fuv',
    'ghrs',
    'gimp',
    'grb',
    'grs',
    'gto',
    'hdf',
    'hii',
    'hopr',
    'hst',
    'iii',
    'imf',
    'iras',
    'ism',
    'lbds',
    'lmc',
    'lmxb',
    'lrf',
    'mama',
    'msmt',
    'mwc',
    'ngc',
    'nicmos',
    'npn',
    'ofad',
    'ota',
    'ovv',
    'pagb',
    'pans',
    'pdr',
    'psf',
    'qso',
    'sed',
    'sins',
    'smc',
    'smov',
    'snr',
    'sofa',
    'ssa',
    'std',
    'stis',
    'tacq',
    'uvis',
    'wfc',
    'wfpc',
    'yso',
    'zams',
]

# These are the only two-letter words not necessarily capitalized, other than
# the NOCAPS_WORDS below. All other two-letter words will be capitalized
# Ly = Lyman
# Zw = Zwicky
# Ia, Ib are "subparts" of Roman Numeral I
TWO_LETTER_NOT_ALL_CAPS = ['ia', 'ib', 'io', 'is', 'ly', 'no', 'up', 'vs', 'zw']

# These are words that are not capitalized unless at the beginning of the title
# or after a colon, semicolon, dash or period.
NOCAPS_WORDS = [
    'a',
    'an',
    'and',
    'as',
    'at',
    'but',
    'by',
    'for',
    'from',
    'in',
    'into',
    'like',
    'nor',
    'of',
    'on',
    'or',
    'over',
    'so',
    'the',
    'to',
    'upon',
    'with',
    'yet',
    'cm',
    'km',
    'nm',
]

# This is used to address weirdness in some titles during Cycles 3, 4, and 5.
CYCLE345_REGEX = re.compile(r'c(ycl?e?)([345])(|medium|high)', re.I)


def fix_title(title: str) -> str:
    """Standardize case and punctuation in titles. If the title is all upper
    case, it gets converted to mixed case.
    """

    def capitalize1(word: str) -> str:
        """Fix capitalization of words from titles all in lower case.
        We start with everything in lower case. Characters get capitalized
        as appropriate.
        """

        if len(word) == 0:
            return word

        if not word.isalpha():

            # Handle possessives
            if word.endswith("'s"):
                return capitalize1(word[:-2]) + "'s"

            # Handle other common punctuation
            for punc in ".,;:-/()'?":
                if punc in word:
                    wordlets = word.split(punc)
                    wordlets = [capitalize1(w) for w in wordlets]
                    return punc.join(wordlets)

            # The text 'Cycle' + [345] is a special case
            match = CYCLE345_REGEX.match(word)
            if match:
                replacement = (
                    'C'
                    + match.group(1)
                    + ' '
                    + match.group(2)
                    + ' '
                    + match.group(3).capitalize()
                )
                return replacement.rstrip()

            # Probably some sort of acronym then
            return word.upper()

        if word in NOCAPS_WORDS:
            return word

        if word in ALLCAPS_WORDS:
            return word.upper()

        if len(word) == 2 and word not in TWO_LETTER_NOT_ALL_CAPS:
            return word.upper()

        if word[0].islower():
            return word[0].upper() + word[1:]

        return word

    #### End of internal function

    # Clean up extra whitespace
    title = title.strip()
    words = title.split()
    title = ' '.join(words)

    # Fix known weirdness
    for (before, after) in TRANSLATIONS:
        title = title.replace(before, after)

    # Strip surrounding quotes
    if title[0] == ''' and title[-1] == ''':
        title = title[1:-1]

    # No space before semicolon or question mark
    title = title.replace(' ;', ';')
    title = title.replace(' ?', '?')

    # Ensure space after a semicolon
    title = title.replace(';', '; ')

    # Repair dashes and quotes
    title = title.replace('--', '-')
    title = title.replace('--', '-')  # handle triple-dashes too
    title = title.replace('``', "'")
    title = title.replace('`', "'")
    title = title.replace("''", "'")
    title = title.replace('"', "'")

    # No space before a period, comma or colon; space after unless it's a number
    # (Avoid "30,000" -> "30, 000", "2:1" -> "2: 1", "3.14" -> "3. 14")
    for punc in ('.', ',', ':'):
        if punc in title:
            words = title.split(punc)
            for k, word in enumerate(words):
                word = word.rstrip()
                if word and not word[0].isdigit():
                    word = ' ' + word
                words[k] = word

            title = punc.join(words)

    # Remove any other duplicated spaces and prepare to process individual words
    words = title.split()

    # Standardize capitalization if necessary
    if title.isupper():
        recapitalize = len(words) * [True]
        if DEBUG:
            print('Re-capitalizing upper-case:', title)

    elif title.islower():
        recapitalize = len(words) * [True]
        if DEBUG:
            print('Re-capitalizing lower-case:', title)

    else:  # already mixed case
        recapitalize = []

        # If most words are lower-case after the first letter, there's no need
        # to re-capitalize
        uppers = 0
        lowers = 0
        for k, word in enumerate(words):
            if word.isupper():
                uppers += 1
                recapitalize.append(True)
            elif len(word) == 1:
                recapitalize.append(True)
            elif word[1:].islower():
                lowers += 1
                recapitalize.append(False)
            else:  # don't re-capitalize if it's already mixed case
                recapitalize.append(False)

        if lowers >= uppers - 2:  # already mixed-case; don't worry about up to
            # two uppercase words--that's common
            return ' '.join(words)

        elif DEBUG:
            print('Re-capitalizing mixed-case:', title)

    # Re-capitalize
    new_words = []
    for k, word in enumerate(words):
        if recapitalize[k]:
            new_words.append(capitalize1(word.lower()))
        else:
            new_words.append(word)

    title = ' '.join(new_words)

    # Capitalize first word and anything after punctuation
    title = title[0].upper() + title[1:]
    for punc in ('-', '- ', ': ', '; ', '. '):
        if punc in title:
            parts = title.split(punc)
            parts = [parts[0]] + [(p[0].upper() + p[1:]) for p in parts[1:]]
            title = punc.join(parts)

    # This can happen: capitalize "A" just before a comma because it's not an
    # article in this context
    title = title.replace(' a,', ' A,')

    if DEBUG:
        print('Case repaired:', title)

    return title


################################################################################
