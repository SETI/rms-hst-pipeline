#
# Required setup before running HST pipeline tasks
- Go to `pds-hst-pipeline/HST`, setup an virtual environment and run `pip install -r requirements.txt`
- Setup these environment variables:
    - 'HST_STAGING' (for downloaded files)
    - 'HST_PIPELINE' (for logs and program info)
    - 'HST_BUNDLES' (for final bundles)
    - 'PDS_HST_PIPELINE' (path of pds-hst-pipeline repo, this is where we execute the shell commands of each task)