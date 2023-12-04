#
# Required setup before running HST pipeline tasks
- Go to `pds-hst-pipeline/HST`, setup an virtual environment and run `pip install -r requirements.txt`
- Setup these environment variables:
    - `HST_STAGING` (for downloaded files)
    - `HST_PIPELINE` (for logs and program info)
    - `HST_BUNDLES` (for final bundles)
    - `PDS_HST_PIPELINE` (path of pds-hst-pipeline repo, this is where we execute the shell commands of each task)
#
# Example commands
- Example Commands to run the full pipeline with all tasks:
    - Run with the full ids (pre-fetch from MAST with True moving target flag):
        - `python pipeline/pipeline_run.py`
    - Query MAST with True moving target flag to get the latest ids, and then run pipeline with them:
        - `python pipeline/pipeline_run.py --get-ids`
    - Run with one proposal id:
        - `python pipeline/pipeline_run.py --proposal-ids 07885`
    - Run with multiple proposal ids:
        - `python pipeline/pipeline_run.py --proposal-ids 13736 05167 10341 14930 06679`
    - Run with specific number of subprocesses and max allowed running time for each task (in sec):
        - `python pipeline/pipeline_run.py --proposal-ids 07885 13736 --max-subproc 30 --max-time 1860`
- Example commands to run each specific task: (use `7885` as an example)
    - These are the commands being run when executing `pipeline_run.py`
        ```
        python pipeline/pipeline_query_hst_moving_targets.py --proposal-ids 7885
        python pipeline/pipeline_query_hst_products.py --proposal-id 7885
        python pipeline/pipeline_get_program_info.py --proposal-id 7885
        python pipeline/pipeline_retrieve_hst_visit.py --proposal-id 7885 --vi 01
        python pipeline/pipeline_retrieve_hst_visit.py --proposal-id 7885 --vi 02
        python pipeline/pipeline_retrieve_hst_visit.py --proposal-id 7885 --vi 03
        python pipeline/pipeline_label_hst_products.py --proposal-id 7885 --vi 01
        python pipeline/pipeline_label_hst_products.py --proposal-id 7885 --vi 02
        python pipeline/pipeline_label_hst_products.py --proposal-id 7885 --vi 03
        python pipeline/pipeline_prepare_browse_products.py --proposal-id 7885 --vi 01
        python pipeline/pipeline_prepare_browse_products.py --proposal-id 7885 --vi 02
        python pipeline/pipeline_prepare_browse_products.py --proposal-id 7885 --vi 03
        python pipeline/pipeline_finalize_hst_bundle.py --proposal-id 7885
        ```