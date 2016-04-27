import os.path
import sys

# This is a hack until I figure out the correct way to generally
# combine my packages with the pds-tools modules.

_PDS_TOOLS_DIR = os.path.join(os.path.expanduser('~'), 'pds-tools')

sys.path.append(_PDS_TOOLS_DIR)
