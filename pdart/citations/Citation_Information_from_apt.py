import re
import xml.dom.minidom as md
from typing import List, Tuple

# This is how we find a date embedded inside a comment
DATE_REGEX1 = re.compile(r" *--.*Submission.*[^0-9](19[7-9][0-9]|20[0-3][0-9])[^0-9].*")

DATE_REGEX2 = re.compile(r".*<!-.*Date.*[^0-9](19[7-9][0-9]|20[0-3][0-9])[^0-9].*")

DATE_REGEX3 = re.compile(r" *:date .*?-(20[0-3][0-9]) .*")

################################################################################


def Citation_Information_from_apt(
    filename: str,
) -> Tuple[int, str, int, List[str], str, int, int]:

    # Read file
    doc = md.parse(filename)

    # Get proposal number
    nodes = doc.getElementsByTagName("HSTProposal")
    propno = int(nodes[0].getAttribute("Phase2ID"))

    # Get category, cycle
    nodes = doc.getElementsByTagName("ProposalInformation")
    category = nodes[0].getAttribute("Category")
    cycle = int(nodes[0].getAttribute("Cycle"))

    # Get authors
    authors = []
    for key in ("PrincipalInvestigator", "CoInvestigator"):
        nodes = doc.getElementsByTagName(key)
        for node in nodes:
            first = node.getAttribute("FirstName")
            middle = node.getAttribute("MiddleInitial")
            last = node.getAttribute("LastName").strip()
            suffix = node.getAttribute("Suffix").strip()

            author = " ".join([first, middle, last, suffix])
            authors.append(author)

    # Get title
    nodes = doc.getElementsByTagName("Title")
    title = nodes[0].childNodes[0].data

    # Try to get the year from the SubmissionLog
    nodes = doc.getElementsByTagName("SubmissionLog")
    submission_year = 0
    for node in nodes:
        if node.childNodes:
            logtext = node.childNodes[0].data
            recs = logtext.split("\n")
            for rec in recs:
                match = DATE_REGEX1.match(rec)
                if match:
                    submission_year = max(submission_year, int(match.group(1)))

    # Update the submission_year from a "Date" comment or alternative
    # submission line
    if submission_year == 0:
        with open(filename, encoding="latin-1") as f:
            recs = f.readlines()

        for rec in recs:
            match = (
                DATE_REGEX1.match(rec)
                or DATE_REGEX2.match(rec)
                or DATE_REGEX3.match(rec)
            )
            if match:
                submission_year = max(submission_year, int(match.group(1)))

    # Find a year based on BEFORE/AFTER/BETWEEN constraint
    timing_year = 0
    for key in ("Start", "End", "Date"):
        nodes = doc.getElementsByTagName(key)
        for node in nodes:
            year_str = node.getAttribute("Year")
            if year_str:
                timing_year = max(timing_year, int(year_str))

    return (propno, category, cycle, authors, title, submission_year, timing_year)
