import re
from typing import List, Tuple

DEBUG = False  # set True for useful debugging info printed to stdout.

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

CATEGORIES = r"(GO|GTO|gto|AUG|SMC|SME|SNAP|CAL|RPT|AR|ENG|SM2|SM3|NASA)"

# This pattern matches the proposal ID
PROPNO_REGEX1 = re.compile(r" +ID:? +([0-9]{4,5})\s*")
PROPNO_REGEX2 = re.compile(r" *HUBBLE SPACE TELESCOPE OBSERVING PROGRAM ([0-9]+)\s*")

# Sometimes "2. Scientific Category" appears in front of the category and cycle
# number, sometimes not
INFO_HEADER1 = re.compile(r"2\. *Scientific Category +3\. *Proposal For +4\. *Cycle\s*")
INFO_REGEX1 = re.compile(
    r".{22} *" + CATEGORIES + r"(?:|/[\w/]+) +([0-9]+)\s*",
)
INFO_HEADER2 = re.compile(r"2\.  *Proposal For +3\. *Cycle\s*")
INFO_HEADER2a = re.compile(r"Type +Cycle +.*")
INFO_REGEX2 = re.compile(r" *" + CATEGORIES + r"(?:|/[\w/]+) +([0-9]+)\s*")
INFO_HEADER3 = re.compile(
    r"2\. *Scientific Category +3\. *Proposal For +4\. *Proposal Type   .*"
)
INFO_REGEX3 = re.compile(r".{22} *" + CATEGORIES + ".*")

CYCLE_HEADER1 = re.compile(r" *Type +Cycle.*")
CYCLE_REGEX1 = re.compile(r".*   ([0-9]+).*")

CYCLE_REGEX2 = re.compile(r".*CYCLE ([0-9]+).*")

CYCLE_IN_TITLE = re.compile(".*(?:CYCLE|CYC.|CYC) *([0-9]+).*", re.I)

# Authors are usually one PI/Pi followed by zero or more CoI/Con lines
PI_REGEX = re.compile(r" *PI: *(.*?)(?:\n|   .*)")
COI_REGEX = re.compile(r" *CoI: *(.*?)(?:\n|   .*)")
LONGNAME_REGEX = re.compile(r" {8}([^ ].*?)(?:\n|   .*)")

# However, first check for a "Proposers" section
AUTHOR_HEADER = re.compile(r" *Proposers +Institution +.*")
AUTHOR_REGEX = re.compile(r"(.*?)   .*")

# The title is always after this line
TITLE_HEADER = re.compile(r"(1\. *Proposal |)Title:?\s*")

# The first option below usually provides the check-in date of the Phase II
# program. However, sometimes it fails, in which case the check-in date is in
# the header line for page 2. For very old files, there's a "Report Date"
# instead.
YEAR1_REGEX = re.compile(r" *Check-in Date: .*?-.*?-([0-9]{4})\s*")
YEAR2_REGEX = re.compile(r".* [01][0-9]/[0-3][0-9]/([0-9]{2}) .*\[  2\]\s*")
YEAR3_REGEX = re.compile(r" *Report Date: [0-9]{2}-...-([0-9]{2})[^0-9].*")
YEAR4_REGEX = re.compile(r" *Check-in Time: [0-9]{2}-...-([0-9]{2})[^0-9].*")

# This pattern matches any year used in a timing constraint, where the format is
# dd-MON-yy or yyyy-MON-dd
MONTHS = r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)"
YMD_REGEX = re.compile(
    r"(?:^|.*[^0-9])((?:19|20)[0-9]{2})-" + MONTHS + r"-[0-9]{1,2}(?:\n|[^0-9].*)"
)
DMY_REGEX = re.compile(
    r"(?:^|.*[^0-9])[0-9]{1,2}-" + MONTHS + r"-(?:|19|20)([0-9]{2})(?:$|[^0-9].*)"
)

# Target Names                 PLUTO, PLUTO-STYX-KERBEROS
TARG_REGEX = re.compile(r".*Target Names\s+(.*)(?:\r\n|\r|\n)")

# Configurations               NIC1 NIC2
INSTRUMENT_REGEX = re.compile(r".*Configurations\s+(.*)(?:\r\n|\r|\n)")

################################################################################

MISSING_CYCLES = {
    5211: 4,
    6141: 4,
    6806: 6,
}

################################################################################


def Citation_Information_from_pro(
    filename: str,
) -> Tuple[int, str, int, List[str], str, int, int, str, str]:

    # A quick and dirty function to merge author lists
    # Sometimes the PI is in the author list, sometimes not!
    def merge_authors(authors: List[str], pi_author: str, cois: List[str]) -> List[str]:
        def letters_only(author: str) -> str:
            letters = []
            for c in author:
                if c.isalpha():
                    letters.append(c)
            return "".join(letters).upper()

        # Author list found
        if authors and not pi_author:
            return authors

        # No author list found
        if not authors:
            return [pi_author] + cois

        # See if PI is first in the author list; if so, return
        pi_letters = letters_only(pi_author)
        letters = letters_only(authors[0])
        if pi_letters in letters or letters in pi_letters:
            return authors

        # See if the PI is further down the list; if so, move it to top
        pi_found_in_authors = False
        for k, author in enumerate(authors):
            letters = letters_only(author)
            if letters in pi_letters or pi_letters in letters:
                pi_found_in_authors = True
                pi_name = authors.pop(k)
                break

        if pi_found_in_authors:
            if DEBUG:
                print("PI moved to top: " + pi_name)

            return [pi_name] + authors

        # PI is not in author list, but other authors are OK
        if DEBUG:
            print("PI %s inserted before co-I %s" % (pi_author, authors[0]))

        return [pi_author] + authors

    # Read file
    with open(filename, "r", encoding="latin-1") as f:
        recs = f.readlines()

    # Get proposal number
    propno = 0
    for rec in recs:
        match = PROPNO_REGEX1.match(rec)
        if match:
            propno = int(match.group(1))
            break

    if not propno:
        for rec in recs:
            match = PROPNO_REGEX2.match(rec)
            if match:
                propno = int(match.group(1))
                break

    # Get proposal type and cycle number
    category = ""
    cycle = 0
    for k, rec in enumerate(recs):
        if INFO_HEADER1.match(rec):
            match = INFO_REGEX1.match(recs[k + 1])
        elif INFO_HEADER2.match(rec):
            match = INFO_REGEX2.match(recs[k + 1])
        elif INFO_HEADER2a.match(rec):
            match = INFO_REGEX2.match(recs[k + 1])
        elif INFO_HEADER3.match(rec):
            match = re.match(INFO_REGEX3, recs[k + 1])
        else:
            match = None

        if match:
            category = match.group(1).upper()
            try:
                cycle = int(match.group(2))
            except IndexError:
                pass  # no cycle value in INFO_REGEX3

            break

    # Get title
    title = ""
    for k, rec in enumerate(recs):
        match = TITLE_HEADER.match(rec)
        if match:
            title = recs[k + 1].strip()
            if recs[k + 2][:4] != "----":  # title has a second line
                # If there's no space before a final dash, don't put one after
                if title.endswith("-") and not title.endswith(" -"):
                    title += recs[k + 2].strip()
                else:
                    title += " " + recs[k + 2].strip()

                if recs[k + 3][:4] != "----":  # title has a third line
                    if title.endswith("-") and not title.endswith(" -"):
                        title += recs[k + 3].strip()
                    else:
                        title += " " + recs[k + 3].strip()

            break

    # Get the authors from a "Proposers" section
    authors = []
    for k, rec in enumerate(recs):
        match = AUTHOR_HEADER.match(rec)
        if match and "----" in recs[k + 1]:
            for next_rec in recs[k + 2 :]:
                match = AUTHOR_REGEX.match(next_rec)
                if match:
                    author = match.group(1).strip()
                    if author:
                        authors.append(author)
                elif "----" in next_rec:
                    break

    # Also look for PI/CoI prefixes
    pi_author = ""
    for k, rec in enumerate(recs):
        rec = rec[:42] + "   "  # wipe out the Institution
        match = PI_REGEX.match(rec)
        if match:
            pi_author = match.group(1).strip()
            match = LONGNAME_REGEX.match(recs[k + 1][:42])
            if match:
                pi_author += " " + match.group(1).strip()

            break

    cois = []
    coi_found = False
    for k, rec in enumerate(recs):
        rec = rec[:42] + "   "  # wipe out the Institution
        match = COI_REGEX.match(rec)
        if match:
            coi_found = True
            name = match.group(1).strip()
            if not name:  # sometimes it's empty. Weird.
                continue

            match = LONGNAME_REGEX.match(recs[k + 1][:42])
            if match:
                name += " " + match.group(1).strip()

            cois.append(name)
        elif coi_found:
            if "----" in rec:  # end of Co-Is
                break

    authors = merge_authors(authors, pi_author, cois)

    # Get the submission year
    submission_year = 0
    for rec in recs:
        match = (
            YEAR1_REGEX.match(rec)
            or YEAR2_REGEX.match(rec)
            or YEAR3_REGEX.match(rec)
            or YEAR4_REGEX.match(rec)
        )
        if match:
            submission_year = int(match.group(1))
            if submission_year < 50:
                submission_year += 2000
            elif submission_year < 100:
                submission_year += 1900

            break

    # Get the timing year, containing the latest date from any timing constraint
    timing_year = 0
    for rec in recs:
        match = DMY_REGEX.match(rec) or YMD_REGEX.match(rec)
        if match:
            alt_year = int(match.group(1))
            if alt_year < 50:
                alt_year += 2000
            elif alt_year < 100:
                alt_year += 1900

            timing_year = max(alt_year, timing_year)

    # Get the target name
    instruments = ""
    for rec in recs:
        match = INSTRUMENT_REGEX.match(rec)
        if match:
            instruments = match.group(1).strip()

    # Get the target name
    target_names = ""
    for rec in recs:
        match = TARG_REGEX.match(rec)
        if match:
            target_names = match.group(1).strip()

    # Fill in a cycle number if it is still missing
    if not cycle:
        for k, rec in enumerate(recs):
            match = CYCLE_HEADER1.match(rec)
            if match:
                match = CYCLE_REGEX1.match(recs[k + 1])
                if match:
                    cycle = int(match.group(1))

                break

    if not cycle:
        for rec in recs:
            match = CYCLE_REGEX2.match(rec)
            if match:
                cycle = int(match.group(1))
                break

    # Sometimes the cycle is embedded in the title
    if not cycle:
        match = CYCLE_IN_TITLE.match(title)
        if match:
            cycle = int(match.group(1))

    # Deal with known cases of missing cycle
    if not cycle:
        if propno in MISSING_CYCLES:
            cycle = MISSING_CYCLES[propno]

    # Check for complete results
    if not propno:
        raise ValueError("missing proposal number in " + filename)
    elif not authors:
        raise ValueError("missing authors in " + filename)
    elif not title:
        raise ValueError("missing title in " + filename)
    elif not instruments:
        raise ValueError("missing instruments in " + filename)
    elif not target_names:
        raise ValueError("missing target names in " + filename)
    elif not cycle:
        raise ValueError("missing cycle number in " + filename)

    return (
        propno,
        category,
        cycle,
        authors,
        title,
        submission_year,
        timing_year,
        instruments,
        target_names,
    )
