##########################################################################################
# tests/conftest.py
#
# Test-session setup shared by every test module.
##########################################################################################

import os

# Limit MAST queries to data released before 2025, so that the result counts the
# network tests assert don't grow with the archive. This runs before any pipeline module
# is imported, which is when hst_helper reads it, and the pipeline subprocesses that
# test_pipeline_run starts inherit it. Set HST_END_DATE yourself to override it.
os.environ.setdefault('HST_END_DATE', '2025-01-01')
