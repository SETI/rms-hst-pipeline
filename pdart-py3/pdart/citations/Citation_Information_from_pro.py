import re
from typing import List, Tuple

################################################################################
# This is the top of a sample .PRO file after c. 1995
################################################################################
#                                                 6803( 13) - 10/12/98 09:54  - [  1]
#
#     PROPOSAL FOR HUBBLE SPACE TELESCOPE OBSERVATIONS   ST ScI Use Only
#                                                        ID:      6803
#                                                        Version: 13
#                                                        Check-in Date: 12-Oct-1998
#                                                                       09:52:41
#
# 1.Proposal Title:
# Disk-resolved Spectrophotometry of the Dark Side of Iapetus
# ------------------------------------------------------------------------------------
# 2. Proposal For  3. Cycle
# GO               6
# ------------------------------------------------------------------------------------
# 4. Investigators
#                                                                                      Contact?
#     PI: Tilmann Denk                      Deutsches Zentrum fuer Luft- und Raumfahrt
#    CoI: Keith S. Noll                     Space Telescope Science Institute             N
#    CoI: Dale P. Cruikshank                NASA Ames Research Center                     N
# ...
# ------------------------------------------------------------------------------------
# 5. Abstract
#
# With this HST observation, we will obtain the first spectrum of pure dark material
# ...
# ------------------------------------------------------------------------------------
#                                                  6803( 13) - 10/12/98 09:54  - [  2]
# ...

################################################################################
# This is the top of an earlier .PRO file
################################################################################
#                                                                   Page   1
#     PROPOSAL FOR HUBBLE SPACE TELESCOPE OBSERVATIONS   ST ScI Use Only
#                                                        ID 5215
#                                                        Report Date: 18-Jul-95:17:04
#                                                        Version: **********
#                                                        Check-in Date: **********
#
# 1.Proposal Title:
# THE MARTIAN SURFACE AND ATMOSPHERE
# ------------------------------------------------------------------------------------
# 2. Scientific Category   3. Proposal For  4. Proposal Type        5. Continuation ID
# SOLAR SYSTEM                GTO/WF2          2 Long Term yrs         <none>
# Sub Category
# INNER PLANETS
# ------------------------------------------------------------------------------------
# 6. Principal Investigator   Institution                    Country    Telephone
# Dr. John T. Trauger         2370                           USA        (818) 354-9594
# PROJECT SCIENTIST
# ------------------------------------------------------------------------------------
# 7. Abstract
# We propose to use WF/PC-II to make high-resolution observations of the surface and
# atmosphere of Mars.  These observations will employ the new capabilities the WF/PC-
# ...
# ------------------------------------------------------------------------------------
# 8. Scientific Key Words:   MARS
# ------------------------------------------------------------------------------------
# 9. Est obs time (hours) pri: 5.17     par: 0       10. Num targs pri: 1      par: 0
# ------------------------------------------------------------------------------------
# 11. Instruments requested:  WF/PC
# ------------------------------------------------------------------------------------
# 12. Special sched req:  Time Critical obs.
# ------------------------------------------------------------------------------------
#                                                                    Page   2
# I. GENERAL FORM    Proposal 5215
# PI: Dr. John T. Trauger
# Proposal Title:
# THE MARTIAN SURFACE AND ATMOSPHERE
# ------------------------------------------------------------------------------------
#
# 1. Proposers:
# Proposers                        Institution                       Country       ESA
# ------------------------------------------------------------------------------------
# Pi John T. Trauger               2370                              USA
# Con David Crisp                  2370                              USA
# Con John T. Clarke               2660                              USA
# ------------------------------------------------------------------------------------
# ...

# This pattern matches the proposal ID
PROPNO_PATTERN: re.Pattern = re.compile(r" +ID:? +([0-9]{4})\s*")

# Sometimes "2. Scientific Category" appears in front of the category and cycle
# number, sometimes not
INFO_HEADER1: re.Pattern = re.compile(
    r"2\. *Scientific Category +3\. *Proposal For +4\. *Cycle\s*"
)
INFO_PATTERN1: re.Pattern = re.compile(r".{22} *(GO|GTO)[^ ]* +([1-9])\s*")
INFO_HEADER2: re.Pattern = re.compile(r"2\.  *Proposal For +3\. *Cycle\s*")
INFO_PATTERN2: re.Pattern = re.compile(r" *(GO|GTO)[^ ]* +([1-9])\s*")
INFO_HEADER3: re.Pattern = re.compile(
    r"2\. *Scientific Category +3\. *Proposal For +4\. *Proposal Type\s*"
)
INFO_PATTERN3: re.Pattern = re.compile(r".{22} *(GO|GTO).*")

CYCLE_PATTERN: re.Pattern = re.compile(r".*CYCLE ([0-9]).*")

# Authors are always one PI followed by zero or more CoI lines
AUTHOR_PATTERN1: re.Pattern = re.compile(r" *PI?i?:? +(.*?)($|  .*)")
AUTHOR_PATTERN2: re.Pattern = re.compile(r" *CoI?n?:? +(.*?)   .*")

# The title is always after this line
TITLE_HEADER: re.Pattern = re.compile(r"1\. *Proposal Title:\s*")

# The first option below usually provides the check-in date of the Phase II
# program. However, sometimes it fails, in which case the check-in date is in
# the header line for page 2. For very old files, there's a "Report Date"
# instead.
YEAR1_PATTERN: re.Pattern = re.compile(r" *Check-in Date: .*?-.*?-([0-9]{4})\s*")
YEAR2_PATTERN: re.Pattern = re.compile(
    r".* [01][0-9]/[0-3][0-9]/([0-9]{2}) .*\[  2\]\s*"
)
YEAR3_PATTERN: re.Pattern = re.compile(
    r" *Report Date: [0-9]{2}-...-([0-9]{2})[^0-9].*"
)

# This pattern matches any year used in a timing constraint, where the format is
# dd-MON-yy.
REQ_PATTERN: re.Pattern = re.compile(
    r".*[0-9]{1,2}-(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)-([0-9]{2})[^0-9].*"
)

# Ideally, the year returned should be the year of the last observation obtained
# in an HST program. Unfortunately, there is no guaranteed way to get this
# information from a .PRO file. Instead we return the check-in year or the year
# of the latest timing constraint, whichever is greater. I think this is the
# best we can do.

################################################################################


def Citation_Information_from_pro(
    filename: str,
) -> Tuple[int, str, int, List[str], str, str]:

    # Quick and dirty function to standardize program titles
    def fix_title(title: str) -> str:

        # Fix known weirdness
        title = title.replace("\\\\cotwo\\\\", "CO2")

        # Fix double-blanks
        title = title.replace("  ", " ")

        # Standardize capitalization if necessary
        if not title.isupper():
            return title

        words = title.split()
        title = " ".join([w.capitalize() for w in words])

        for test in (
            "a",
            "an",
            "the",
            "for",
            "and",
            "or",
            "at",
            "by",
            "from",
            "of",
            "on",
            "to",
            "with",
        ):
            title = title.replace(" " + test.capitalize() + " ", " " + test + " ")

        return title

    # A quick and dirty function to standardize author names
    def fix_authors(authors: List[str]) -> List[str]:

        for k, author in enumerate(authors):

            # Strip titles if any
            author = author.replace("Prof. ", "")
            author = author.replace("Dr. ", "")
            author = author.replace("Mr. ", "")
            author = author.replace("Ms. ", "")

            # Fix known weirdness
            author = author.replace('GR"U N', "Gruen")

            # Standarize case of last names, which are sometimes all caps
            if author[-1].isupper():
                words = author.split()
                for w, word in enumerate(words):  # handles "A'Hearn" correctly!
                    chars = list(word)
                    c = len(chars) - 1
                    while c > 0 and chars[c].isupper() and chars[c - 1].isupper():
                        chars[c] = chars[c].lower()
                        c -= 1
                    words[w] = "".join(chars)

                # But then there are Roman numerals. Yikes!
                if words[-1] in ("Ii", "Iii", "Iv"):
                    words[-1] = words[-1].upper()

                author = " ".join(words)

            authors[k] = author

        return authors

    # Read file
    with open(filename) as f:
        recs = f.readlines()

    # Initialize the info we seek
    propno = 0
    category = ""
    authors: List[str] = []
    title = ""
    year = 0
    needed = 5  # tracks when to stop searching for citation info

    cycle = None  # cycle number sometimes needs to be handled separately

    # Loop through records, allowing for skipped records
    k = -1
    while needed > 0:
        k += 1

        try:
            rec = recs[k]
        except IndexError:  # we got to the end and some info wasn't found
            if not propno:
                raise ValueError("missing proposal number in " + filename)
            elif not category:
                raise ValueError("missing proposal category in " + filename)
            elif not authors:
                raise ValueError("missing authors in " + filename)
            elif not title:
                raise ValueError("missing title in " + filename)
            else:
                raise ValueError("missing year in " + filename)

        # Try to get proposal number if still needed
        if not propno:
            match = re.match(PROPNO_PATTERN, rec)
            if match:
                propno = int(match.group(1))
                needed -= 1
                next

        # Try to get proposal type and cycle number if still needed
        if not category:
            if re.match(INFO_HEADER1, rec):
                match = re.match(INFO_PATTERN1, recs[k + 1])
            elif re.match(INFO_HEADER2, rec):
                match = re.match(INFO_PATTERN2, recs[k + 1])
            elif re.match(INFO_HEADER3, rec):
                match = re.match(INFO_PATTERN3, recs[k + 1])
            else:
                match = None

            if match:
                category = match.group(1)
                try:
                    cycle = int(match.group(2))
                except IndexError:
                    pass  # no cycle value in INFO_PATTERN3
                k += 1
                needed -= 1
                next

        # Try to get author list if still needed
        if not authors:
            match = re.match(AUTHOR_PATTERN1, rec)
            if match:
                authors.append(match.group(1))

                match = re.match(AUTHOR_PATTERN2, recs[k + 1])
                while match:
                    authors.append(match.group(1))
                    k += 1
                    match = re.match(AUTHOR_PATTERN2, recs[k + 1])

                needed -= 1
                next

        # Try to get title if still needed
        if not title:
            match = re.match(TITLE_HEADER, rec)
            if match:
                title = recs[k + 1].strip()
                k += 1
                if recs[k + 1][:4] != "----":  # in case title has a second line
                    title += " " + recs[k + 1].strip()
                    k += 1

                needed -= 1
                next

        # Try to get year if still needed
        if not year:
            match = re.match(YEAR1_PATTERN, rec)
            if match:
                year = int(match.group(1))
                needed -= 1
                next

            match = re.match(YEAR2_PATTERN, rec)
            if match:
                year = 1900 + int(match.group(1))
                if year < 1950:
                    year += 100
                needed -= 1
                next

            match = re.match(YEAR3_PATTERN, rec)
            if match:
                year = 1900 + int(match.group(1))
                if year < 1950:
                    year += 100
                needed -= 1
                next

    # Now we update the year with the latest date from any timing constraint
    # We also fill in a cycle number if it is still missing
    for rec in recs[k + 1 :]:
        match = re.match(REQ_PATTERN, rec)
        if match:
            req_year = 1900 + int(match.group(1))
            if year < 1950:
                year += 100
            year = max(year, 1900 + int(match.group(1)))

        if not cycle:
            match = re.match(CYCLE_PATTERN, rec)
            if match:
                cycle = int(match.group(1))

    if cycle is None:
        raise ValueError("missing cycle number in " + filename)

    # Misc. cleanup
    authors = fix_authors(authors)
    title = fix_title(title)

    return (propno, category, cycle, authors, title, str(year))
