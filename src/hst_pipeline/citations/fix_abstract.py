##########################################################################################
# citations/fix_abstract.py
##########################################################################################

import re

def fix_abstract(abstract):
    """Standardize punctuation in abstracts. Input is one string, a list containing one
    string per embedded line break (i.e., paragraph) is returned.
    """

    # Replace multiple newlines with one
    abstract = abstract.replace('\r', '\n')
    parts = abstract.split('\n')
    parts = [p for p in parts if p.strip()]
    abstract = '\n'.join(parts)

    # Operate on each line separately
    new_abstract = []
    for rec in abstract.splitlines():

        # Clean up extra whitespace
        rec = rec.strip()
        words = rec.split()
        rec = ' '.join(words)

        # Correct space after comma inside a number
        rec = re.sub(r'(\d), (\d\d\d)([^\d])', r'\1,\2\3', rec)

        # Correct extraneous "! "
        rec = re.sub(r'! ([^A-Z])', r'\1', rec)

        # The .pro files often have a "?" where a special character should be
        # In some cases, we can guess what's missing
        rec = re.sub(r'(\d+)\?(C|K)', r'\1°\2', rec)
        rec = re.sub(r'([^\w])\?(\w[\w -]*\.?)\?([^\w])', r'\1“\2”\3', rec)
        rec = re.sub(r'([^\w])\?“(\w[\w -]*\.?)”\?([^\w])', r'\1“\2”\3', rec)
        rec = re.sub(r'(\w)\?s([^\w])', r'\1’s\2', rec)
        rec = re.sub(r's\? ([a-z])', r's’ \1', rec)
        rec = re.sub(r'([A-Za-z]+)\?([A-Za-z]+)', r'\1-\2', rec)

        # Fix non-ASCII and inconsistencies
        rec = rec.replace('r?gime', 'regime')

        rec = rec.replace('\x93', '“')
        rec = rec.replace('\x94', '”')
        rec = rec.replace('``', '“')
        rec = rec.replace("''", '”')
        rec = rec.replace('‘‘', '“')
        rec = rec.replace('’’', '”')
        rec = rec.replace('ÔÔ', '“')
        rec = rec.replace('ÕÕ', '”')
        rec = rec.replace('Ò', '“')
        rec = rec.replace('Ó', '”')
        rec = rec.replace('â\x80\x9c', '“')
        rec = rec.replace('â\x80\x9d', '”')

        rec = rec.replace('\x92', "’")
        rec = rec.replace('Õ', "’")
        rec = rec.replace('â\x80\x99', "’")

        rec = rec.replace('â\x88\x92', '-')  # standard en-dash
        rec = rec.replace('â\x88’', '-')
        rec = rec.replace('â\x80\x90', '-')
        rec = rec.replace('−', '-')
        rec = rec.replace('‐', '-')

        rec = rec.replace('\x96', '—')  # long dash
        rec = rec.replace('–', '—')
        rec = rec.replace('â\x80\x99', '—')
        rec = rec.replace('â\x80"', '—')
        rec = rec.replace('â\x80“', '—')
        rec = rec.replace('Ð', '—')

        rec = rec.replace('â\x88¼', '~')
        rec = rec.replace('∼', '~')

        rec = rec.replace('\x81', 'Å')
        rec = rec.replace('Ã\x83Â', 'Å')
        rec = rec.replace('|*|', 'Å')
        rec = rec.replace('Ã ', 'Å')
        rec = rec.replace('Ã', 'Å')

        rec = rec.replace('ﬁ', 'fi')
        rec = rec.replace('ï¬\x81', 'fi')
        rec = rec.replace('ï¬Å', 'fi')
        rec = rec.replace(' ?rst', ' first')
        rec = rec.replace(' ?ve ', ' five ')
        rec = rec.replace('?eld', 'field')
        rec = rec.replace('ef-cient', 'efficient')

        rec = rec.replace(' out?ow', ' outflow')
        rec = rec.replace(' out-ow', ' outflow')
        rec = rec.replace(' outﬂow', ' outflow')

        rec = rec.replace('Âµ', 'µ')

        rec = rec.replace('Î±', 'α')

        rec = rec.replace('Ã\x97', '×')
        rec = rec.replace('Ã\x97', '×')

        rec = rec.replace('É', '…')
        rec = rec.replace('…', '…')

        rec = rec.replace('â\x80©', '')
        rec = rec.replace(' Â ', ' ')

        # Append to abstract with an extra blank line after each paragrapn
        new_abstract += [rec, '']

    new_abstract = new_abstract[:-1]  # Strip extra blank line at the end

    # Sometimes paragraph breaks are incorrect. Fix the most obvious ones.
    merged = '\n'.join(new_abstract)
    merged = re.sub(r'\n([a-z])', r' \1', merged)
    merged = re.sub(r'([,\w])\n', r'\1 ', merged)
    merged = merged.replace('  ', ' ')
    new_abstract = merged.splitlines()

    return new_abstract

##########################################################################################
