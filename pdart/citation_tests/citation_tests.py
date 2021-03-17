# This file contains a sequence of scripts intended to run interactively.
# The output of each run is included in this directory.
# If changes are made to one of the Citation_Information functions, one should
# re-run the scripts to see if the output has changed.
#
# The input directory, containing about 10,000 .pro and .apt files, is on
# Dropbox. It's 3.8 GB so not suitable for checking into GitHub.
#
# Each test is written as independent code suitable for quick cut-and-paste into
# an iPython shell. Each section stands alone and can be executed independently.
# Note that the path to the Dropbox directory, defined by PREFIX, will need to
# be modified first via a global find-and-replacement.
#
# Several of the scripts identify mismatches between the results of reading a
# .apt file and reading the .pro file from the same program. It appears to be
# the case that the only mismatches are cases where there is genuinely different
# information in the two files. When that happens, the differences are
# usually but not always trivial. Regardless, the information in the .apt file
# should take precedence.
#
# --Mark Showalter, June 30, 2020

################################################################################
# Print every description
################################################################################

import os
from citations import Citation_Information

PREFIX = "/Users/mark/Desktop/HST-pro-apt-files/"

for k in range(5000, 16101):
    filename1 = PREFIX + str(k) + ".pro"
    filename2 = PREFIX + str(k) + ".apt"

    if os.path.exists(filename1):
        try:
            c1 = Citation_Information.create_from_file(filename1)
            print(k, "PRO", c1.description)
        except Exception as e:
            print("*****", k, "*****", e)

    if os.path.exists(filename2):
        try:
            c2 = Citation_Information.create_from_file(filename2)
            print(k, "APT", c2.description)
        except Exception as e:
            print("*****", k, "*****", e)

################################################################################
# Print every unique title
################################################################################

import os
from citations import Citation_Information

PREFIX = "/Users/mark/Desktop/HST-pro-apt-files/"

titles = set()
for k in range(5000, 16101):
    filename1 = PREFIX + str(k) + ".pro"
    filename2 = PREFIX + str(k) + ".apt"

    if os.path.exists(filename1):
        try:
            c1 = Citation_Information.create_from_file(filename1)
            if c1.title not in titles:
                print(c1.title)
                titles.add(c1.title)

        except Exception as e:
            print("*****", k, "*****", e)

    if os.path.exists(filename2):
        try:
            c2 = Citation_Information.create_from_file(filename2)
            if c2.title not in titles:
                print(c2.title)
                titles.add(c2.title)

        except Exception as e:
            print("*****", k, "*****", e)

################################################################################
# Print every unique author
################################################################################

import os
from citations import Citation_Information

PREFIX = "/Users/mark/Desktop/HST-pro-apt-files/"

authors = set()
for k in range(5000, 16101):
    filename1 = PREFIX + str(k) + ".pro"
    filename2 = PREFIX + str(k) + ".apt"

    if os.path.exists(filename1):
        try:
            c1 = Citation_Information.create_from_file(filename1)
            for author in c1.authors:
                if author not in authors:
                    print(author)
                    authors.add(author)

        except Exception as e:
            print("*****", k, "*****", e)

    if os.path.exists(filename2):
        try:
            c2 = Citation_Information.create_from_file(filename2)
            for author in c2.authors:
                if author not in authors:
                    print(author)
                    authors.add(author)

        except Exception as e:
            print("*****", k, "*****", e)

################################################################################
# Check that titles match from .pro and .apt files
################################################################################

import os
from citations import Citation_Information

PREFIX = "/Users/mark/Desktop/HST-pro-apt-files/"

for k in range(9730, 16101):
    filename1 = PREFIX + str(k) + ".pro"
    filename2 = PREFIX + str(k) + ".apt"
    if not os.path.exists(filename1):
        continue
    if not os.path.exists(filename2):
        continue

    try:
        c1 = Citation_Information.create_from_file(filename1)
        c2 = Citation_Information.create_from_file(filename2)

        if c1.title != c2.title:
            print(k)
            print(c1.title)
            print(c2.title)
    except Exception as e:
        print("*****", k, "*****", e)

################################################################################
# Check that sorted lists of authors match from .pro and .apt files
################################################################################

import os
from citations import Citation_Information

PREFIX = "/Users/mark/Desktop/HST-pro-apt-files/"

for k in range(9730, 16101):
    filename1 = PREFIX + str(k) + ".pro"
    filename2 = PREFIX + str(k) + ".apt"
    if not os.path.exists(filename1):
        continue
    if not os.path.exists(filename2):
        continue

    try:
        c1 = Citation_Information.create_from_file(filename1)
        c2 = Citation_Information.create_from_file(filename2)

        authors1 = list(c1.authors)
        authors2 = list(c2.authors)
        authors1.sort()
        authors2.sort()

        if authors1 != authors2:
            print(k)
            print(authors1)
            print(authors2)
    except Exception as e:
        print("*****", k, "*****", e)

################################################################################
# Check that first authors match from .pro and .apt files
################################################################################

import os
from citations import Citation_Information

PREFIX = "/Users/mark/Desktop/HST-pro-apt-files/"

for k in range(9730, 16101):
    filename1 = PREFIX + str(k) + ".pro"
    filename2 = PREFIX + str(k) + ".apt"
    if not os.path.exists(filename1):
        continue
    if not os.path.exists(filename2):
        continue

    try:
        c1 = Citation_Information.create_from_file(filename1)
        c2 = Citation_Information.create_from_file(filename2)

        if c1.authors[0] != c2.authors[0]:
            print(k)
            print(c1.authors[0])
            print(c2.authors[0])
    except Exception as e:
        print("*****", k, "*****", e)

################################################################################
# Check that years match from .pro and .apt files
################################################################################

import os
from citations import Citation_Information

PREFIX = "/Users/mark/Desktop/HST-pro-apt-files/"

for k in range(9730, 16101):
    filename1 = PREFIX + str(k) + ".pro"
    filename2 = PREFIX + str(k) + ".apt"
    if not os.path.exists(filename1):
        continue
    if not os.path.exists(filename2):
        continue

    try:
        c1 = Citation_Information.create_from_file(filename1)
        c2 = Citation_Information.create_from_file(filename2)

        if c1.publication_year != c2.publication_year:
            print(k, c1.publication_year, c2.publication_year)
    except Exception as e:
        print("*****", k, "*****", e)

################################################################################
# Check that abstracts match from .pro and .apt files
################################################################################

import os
from citations import Citation_Information

PREFIX = "/Users/mark/Desktop/HST-pro-apt-files/"

for k in range(9730, 16101):
    if k in (
        10142,
        10282,
        10316,
        10454,
        10652,
        10772,
        10790,
        10890,
        10934,
        10939,
        10972,
        11195,
        11265,
        11275,
        11308,
        11340,
        11478,
        11572,
        11648,
        11671,
        11726,
        11994,
        12049,
        12230,
        12250,
        12303,
        12305,
        12318,
        12329,
        12330,
        12365,
        12366,
        12379,
        12435,
        12550,
        12551,
        12677,
        12703,
        12704,
        12705,
        13162,
        13167,
        13599,
        13602,
        13637,
        13962,
        14405,
        14863,
        15731,
    ):
        continue

    filename1 = PREFIX + str(k) + ".pro"
    filename2 = PREFIX + str(k) + ".apt"
    if not os.path.exists(filename1):
        continue
    if not os.path.exists(filename2):
        continue

    print(k)
    try:
        c1 = Citation_Information.create_from_file(filename1)
        c2 = Citation_Information.create_from_file(filename2)

        abstract1 = c1.abstract
        abstract2 = c2.abstract

        # Ignore spacing, apostrophe, and quote discrepancies
        #         abstract1 = abstract1.replace(" ","")
        #         abstract2 = abstract2.replace(" ","")
        #
        #         abstract1 = abstract1.replace("’","'")
        #         abstract2 = abstract2.replace("’","'")
        #
        #         abstract1 = abstract1.replace("“",'"').replace("”",'"')
        #         abstract2 = abstract2.replace("“",'"').replace("”",'"')

        if abstract1 != abstract2:
            print(k)
            print(c1.abstract)
            print(c2.abstract)
    except Exception as e:
        print("*****", k, "*****", e)

################################################################################
