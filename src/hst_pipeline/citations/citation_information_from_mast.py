##########################################################################################
# citations/citation_information_from_mast.py
#
# Extract citation information from a MAST proposal search page, the fallback for early
# programs that have no Phase 2 file (.apt or .pro) at STScI.
##########################################################################################

import html
import re

# The page for a proposal, e.g., https://archive.stsci.edu/proposal_search.php?id=1086&...
# contains (on separate lines, among others):
#   <h2 align='center'>DO NEPTUNE AND PLUTO HAVE RINGS?<br> HST Proposal 1086</h2>
#   <center><H3>Robert Bless <span ...>(...)</span><br>University of ...</H3></center>
#   <b>Cycle: </b>1<br>
#   <B>Proposal type:</B>   GTO/HSP<br>
#   <HR><H2>Proposal Abstract</H2>
#   <p> The origin of planetary ring systems ... </p><hr/>

TITLE_REGEX = re.compile(r"<h2 align='center'>(.*?)<br>\s*HST Proposal (\d+)\s*</h2>",
                         re.I | re.S)
PI_REGEX = re.compile(r'<center>\s*<h3>(.*?)<span', re.I | re.S)
CYCLE_REGEX = re.compile(r'<b>Cycle:\s*</b>\s*(\d+)', re.I)
TYPE_REGEX = re.compile(r'<b>Proposal type:</b>\s*([^<]*?)\s*<br>', re.I)
ABSTRACT_REGEX = re.compile(r'<h2>Proposal Abstract</h2>\s*<p>(.*?)</p>', re.I | re.S)

# Early abstracts end with the proposal's revision history, which is not part of it
REVISION_HISTORY_REGEX = re.compile(r'\s*Revision History:.*', re.S)

def _text(fragment):
    """Return the plain text of an HTML fragment, with tags removed, entities decoded,
    and whitespace collapsed.

    Input:
        fragment    a string of HTML.

    Returns:    the plain text.
    """
    text = html.unescape(re.sub(r'<[^>]*>', ' ', fragment))
    return ' '.join(text.split())

def citation_information_from_mast(filename):
    """Return tuple (propno, category, cycle, authors, title, submission_year,
    timing_year, abstract) from a saved MAST proposal search page.

    The page names only the PI, so the PI is the only author. It gives no submission or
    timing year, so both are returned as zero and the publication year falls back to
    the end of the cycle.

    Input:
        filename    path to the saved HTML page.

    Returns:    the tuple of citation information.

    Raises:     ValueError if the page lacks the proposal number, PI, title, or cycle,
                e.g., because MAST has no proposal with that ID.
    """
    with open(filename, encoding='utf-8', errors='replace') as f:
        page = f.read()

    match = TITLE_REGEX.search(page)
    if not match:
        raise ValueError('missing title and proposal number in ' + filename)
    title = _text(match.group(1))
    propno = int(match.group(2))

    match = PI_REGEX.search(page)
    pi = _text(match.group(1)) if match else ''
    if not pi:
        raise ValueError('missing PI in ' + filename)

    match = CYCLE_REGEX.search(page)
    if not match:
        raise ValueError('missing cycle number in ' + filename)
    cycle = int(match.group(1))

    match = TYPE_REGEX.search(page)
    category = _text(match.group(1)) if match else ''

    match = ABSTRACT_REGEX.search(page)
    abstract = _text(match.group(1)) if match else ''
    abstract = REVISION_HISTORY_REGEX.sub('', abstract)

    return (propno, category, cycle, [pi], title, 0, 0, abstract)
