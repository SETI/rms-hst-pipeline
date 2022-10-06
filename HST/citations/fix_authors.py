################################################################################
# fix_authors.py
################################################################################

import re

# Possible titles to omit from author names
TITLES = {
    'Prof.',
    'PROF. ',
    'Prof',
    'PROF',
    'Assoc.',
    'ASSOC.',
    'Assoc',
    'ASSOC',
    'Dr.',
    'DR.',
    'Dr',
    'DR',
    'Drs.',
    'DRS.',
    'Drs',
    'DRS',
    'Dra.',
    'DRA.',
    'Dra',
    'DRA',
    'Mr.',
    'MR.',
    'Mr',
    'MR',
    'Pr.',
    'PR.',
    'Pr',
    'PR',
    'Ms.',
    'MS.',
    'Ms',
    'MS',
    'Mrs.',
    'MRS.',
    'Mrs',
    'MRS',
    'Lcda.',
    'LCDA.',
    'Lcda',
    'LCDA',
    'A/Prof',
    'Co-I',
    '(Esa)',
    'Col.',
}

PI_FLAGS = ['P.I.', 'PI ', 'Pi ']

# Various TeX-isms in names need to be fixed
NAME_TRANSLATIONS = [
    (r'\-', ''),
    (r'\ n', 'ñ'),
    (r'\~n', 'ñ'),
    (r"'\i ", 'í'),
    (r"'\i", 'í'),
    (r'\^e', 'ê'),
    (r'\^o', 'ô'),
    (r"\'e", 'é'),
    (r'\o ', 'ø'),
    (r"e\'", 'é'),
    (r'\.z', 'ż'),
    (r'\c ', 'ç'),
    ("'o ", 'ó'),
    (r"'O", 'Ó'),
    (r'.\ ', '. '),
    (r'``', '"'),
    (r"''", '"'),
    (r'^\dag', ''),
    ('M?rcio', 'Márcio'),
    ('GR"U N', 'Grün'),
    ("Zolt'a n", 'Zoltán'),
    ('Andre'', 'André'),
    ('Rene'', 'René'),
    ('Marti n', 'Martín'),
    ("Jos'e", 'José'),
    ("Jose'", 'José'),
    ('Mo/ller', 'Møller'),
    ('CHRISTPPHER', 'Christopher'),
    ('Gonz?lez', 'González'),
    ('Jes?s', 'Jesús'),
    ('Ma?z-Apell?niz', 'Maíz-Apellániz'),
    ('Ro?kar', 'Roškar'),
    ('MÃ¼ller', 'Müller'),
    ('MÃ\x83Â¼ller', 'Müller'),
    ('Andr|*|s', 'Andrés'),
    ("Dall' Aglio", "Dall'Aglio"),
    ("Maccio'", 'Macciò'),
    ("Lame'e'", "Lame'e"),
    ("Ken'Ichi", "Ken'ichi"),
]


def fix_authors(authors):
    """Standardize a list of names"""

    # Fix this weirdness first
    for k in range(len(authors)):
        if 'Alerts-Distribution' in authors[k]:
            _ = authors.pop(k)
            break

    pi_index = 0  # Index of a name that begins with "Pi "
    for k, author in enumerate(authors):

        # General cleanup
        author = author.strip()

        for (before, after) in NAME_TRANSLATIONS:
            author = author.replace(before, after)

        for flag in PI_FLAGS:
            if author.startswith(flag):
                author = author[len(flag) :].lstrip()

        author = author.replace('.', '. ')      # ensure spaces after periods
        author = author.replace('. -', '.-')    # but not just before a dash
        author = author.replace('--', '-')      # remove double-dashes

        # Remove titles and roles
        words = author.split()
        if words and words[0].upper() in PI_FLAGS:
            words = words[1:]
            pi_index = k

        for j, word in enumerate(words):
            if word in TITLES:
                _ = words.pop(j)

        # Convert to mixed case if necessary
        for j, word in enumerate(words):
            if word.isupper() and word not in ('II', 'III', 'IV'):
                # Lower case any letter preceeded by an uppercase letter.
                # This handles "A'Hearn", "O'Dell" and hyphenated names.
                chars = list(word)
                for i in range(len(word) - 1):
                    if word[i].isupper():
                        chars[i + 1] = chars[i + 1].lower()
                words[j] = ''.join(chars)

        # Ensure periods after initials
        for j, word in enumerate(words):
            if len(word) == 1 and word.isalpha() and word != 'Ó':
                words[j] = word + '.'

        # Ensure no space, then a capital after "O'", "D'", "A'", "Mc"
        for prefix in ('O'', 'D'', 'A'', 'Mc'):
            if prefix in words:
                author = ' '.join(words)
                author = author.replace(prefix + ' ', prefix)
                words = author.split(' ')

            for j, word in enumerate(words):
                if word.startswith(prefix):
                    words[j] = prefix + word[2].upper() + word[3:]

        author = ' '.join(words)

        # Swap last, first
        if ',' in author:
            parts = author.partition(',')

            # Don't remove the comma before a suffix, but anything after the
            # suffix has to move.
            #   Examples, "Augustus, Jr. Oemler" -> "Oemler Augusts, Jr."
            words = parts[2].split()
            head_words = []
            tail_words = []
            for word in words:
                if word in ('Jr.', 'Sr.', 'II', 'III', 'IV'):
                    tail_words.append(word)
                else:
                    head_words.append(word)

            if tail_words:
                words = head_words + [parts[0] + ','] + tail_words
            else:
                words = head_words + [parts[0]]

            author = ' '.join(words)

        authors[k] = author

    # If a PI was found, put that name first in list
    if pi_index:
        pi_name = authors.pop(pi_index)
        authors = [pi_name] + authors

    return authors


################################################################################
