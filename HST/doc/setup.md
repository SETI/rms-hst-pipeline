#
# Required setup before running HST pipeline tasks
- Go to `pds-hst-pipeline/HST`, setup an virtual environment and run `pip install -r requirements.txt`
- Setup these environment variables:
    - 'HST_STAGING' (for downloaded files)
    - 'HST_PIPELINE' (for logs and program info)
    - 'HST_BUNDLES' (for final bundles)
    - 'PDS_HST_PIPELINE' (path of pds-hst-pipeline repo, this is where we execute the shell commands of each task)
#
# Example commands
- Example Commands to run the full pipeline with all tasks:
    - Run with one proposal id:
        - `python pipeline/pipeline_run.py --prog-id 07885`
    - Run with multiple proposal ids:
        - `python pipeline/pipeline_run.py --prog-id 13736 05167 10341 14930 06679`
- Example commands to run each specific task: (use `7885` as an example)
    - These are the commands being run when executing `pipeline_run.py`
        ```
        python pipeline/pipeline_query_hst_moving_targets.py --prog-id 7885
        python pipeline/pipeline_query_hst_products.py --prog-id 7885
        python pipeline/pipeline_get_program_info.py --prog-id 7885
        python pipeline/pipeline_retrieve_hst_visit.py --prog-id 7885 --vi 01
        python pipeline/pipeline_retrieve_hst_visit.py --prog-id 7885 --vi 02
        python pipeline/pipeline_retrieve_hst_visit.py --prog-id 7885 --vi 03
        python pipeline/pipeline_label_hst_products.py --prog-id 7885 --vi 01
        python pipeline/pipeline_label_hst_products.py --prog-id 7885 --vi 02
        python pipeline/pipeline_label_hst_products.py --prog-id 7885 --vi 03
        python pipeline/pipeline_prepare_browse_products.py --prog-id 7885 --vi 01
        python pipeline/pipeline_prepare_browse_products.py --prog-id 7885 --vi 02
        python pipeline/pipeline_prepare_browse_products.py --prog-id 7885 --vi 03
        python pipeline/pipeline_finalize_hst_bundle.py --prog-id 7885
        ```