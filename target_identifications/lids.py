# type: ignore
################################################################################
# lids.py - Tools for LID creation
################################################################################

import unidecode

PREFIX = "urn:nasa:pds:context:target:"


def clean(text):

    # Remove diacritics
    text = unidecode.unidecode(text)

    # Replace spaces with underscores
    text = text.replace(" ", "_")

    # Convert to lower case
    text = text.lower()

    # Filter out disallowed characters
    chars = [c for c in text if c in "abcdefghijklmnopqrstuvwxyz0123456789_-.:"]
    text = "".join(chars)

    # Insert prefix if necessary
    if "target:" in text:
        text = text.split("target:")[1]

    return PREFIX + text
