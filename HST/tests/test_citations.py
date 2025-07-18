import pytest
import tempfile
import os
from HST.citations.citation_information_from_apt import citation_information_from_apt
from HST.citations.fix_abstract import fix_abstract
from HST.citations.fix_authors import fix_authors
from HST.citations.fix_title import fix_title
import xml.dom.minidom as md

def write_xml_file(contents):
    fd, path = tempfile.mkstemp(suffix='.xml')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(contents)
    return path

def test_citation_information_from_apt_basic():
    xml = '''<?xml version="1.0"?>
<HSTProposal Phase2ID="12345">
  <ProposalInformation Category="GO" Cycle="27" />
  <PrincipalInvestigator FirstName="Jane" MiddleInitial="A." LastName="Doe" Suffix="Jr." />
  <CoInvestigator FirstName="John" MiddleInitial="B." LastName="Smith" Suffix="" />
  <Title>Test Proposal Title</Title>
  <Abstract>This is the abstract.\nWith a line break.</Abstract>
  <SubmissionLog>-- Submission Date: 2021</SubmissionLog>
  <Start Year="2022" />
</HSTProposal>
'''
    path = write_xml_file(xml)
    result = citation_information_from_apt(path)
    os.remove(path)
    assert result[0] == 12345
    assert result[1] == 'GO'
    assert result[2] == 27
    assert 'Jane' in result[3][0]
    assert 'John' in result[3][1]
    assert 'Test Proposal Title' in result[4]
    # The code does not extract submission year from SubmissionLog unless in a comment, so expect 0
    assert result[5] == 0
    assert result[6] == 2022
    assert 'This is the abstract.' in result[7]

def test_citation_information_from_apt_no_abstract():
    xml = '''<?xml version="1.0"?>
<HSTProposal Phase2ID="12345">
  <ProposalInformation Category="GO" Cycle="27" />
  <PrincipalInvestigator FirstName="Jane" MiddleInitial="A." LastName="Doe" Suffix="Jr." />
  <Title>Test Proposal Title</Title>
</HSTProposal>
'''
    path = write_xml_file(xml)
    result = citation_information_from_apt(path)
    os.remove(path)
    assert result[7] == ''

def test_fix_abstract_basic():
    text = 'This is a test.\n\nSecond paragraph! not capitalized.'
    result = fix_abstract(text)
    assert isinstance(result, list)
    assert 'This is a test.' in result[0]
    assert any('Second paragraph' in r for r in result if r)

def test_fix_abstract_unicode_and_punctuation():
    text = 'r?gime\n\x93quoted\x94\n out?ow\n out-ow\n outﬂow\n'
    result = fix_abstract(text)
    joined = ' '.join(result)
    assert 'r-gime' in joined  # The code replaces 'r?gime' with 'r-gime'
    assert '“quoted”' in joined
    assert 'out-ow' in joined or 'outﬂow' in joined

def test_fix_authors_basic():
    authors = ['Dr. John Doe', 'Prof. Jane Roe', 'PI Alan Smithee']
    result = fix_authors(authors.copy())
    assert 'John Doe' in result[0]
    assert 'Jane Roe' in result[1]
    assert 'Alan Smithee' in result[2]

def test_fix_authors_titles_and_case():
    authors = ['DR. JOHN DOE', 'MR. JAMES SMITH', 'PI JANE ROE']
    result = fix_authors(authors.copy())
    # Check that all names are present and properly formatted
    names = ' '.join(result)
    assert 'John Doe' in names
    assert 'James Smith' in names
    assert 'Jane Roe' in names

def test_fix_title_basic():
    title = 'A STUDY OF THE HST COSMOS'
    result = fix_title(title)
    assert result.startswith('A Study')
    assert 'HST' in result

def test_fix_title_weirdness():
    title = 'Mid?IR and H_2 Emission'
    result = fix_title(title)
    assert 'Mid-IR' in result
    assert 'H₂' in result
