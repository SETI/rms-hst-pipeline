"""
Append the path to the pds-tools package directory to the system path.

This is a temporary hack until I figure out the correct way to
generally combine my packages with the pds-tools modules.
"""
import os.path
import sys


_PDS_TOOLS_DIR = os.path.join(os.path.expanduser('~'), 'pds-tools')

sys.path.append(_PDS_TOOLS_DIR)
