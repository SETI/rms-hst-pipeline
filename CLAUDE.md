# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Required environment variables

`HST_STAGING`, `HST_PIPELINE`, and `HST_BUNDLES` are read at import time in
`hst_helper/__init__.py` — no defaults. If any is unset or empty, importing any pipeline
module raises `hst_helper.MissingEnvironmentError` (a `RuntimeError`) naming every missing
variable, and pytest collection fails the same way. `import hst_pipeline` on its own does
not read them.

- `HST_STAGING` — where MAST downloads land
- `HST_PIPELINE` — logs, program info, `task_queue.db`
- `HST_BUNDLES` — final deliverable bundles

(`PDS_HST_PIPELINE` is no longer used. The task queue dispatches console scripts by name
rather than locating `.py` files inside the checkout.)

## Setup

```bash
./scripts/setup-venv.sh && source venv/bin/activate
pip install --no-deps -e ../rms-psiops -e ../rms-picmaker -e ../rms-hst-targets
```

Or `pip install -e ".[dev]"` for the first line. Three dependencies aren't on PyPI in the
form needed, so `pyproject.toml` names them by GitHub URL: `rms-psiops` (not on PyPI;
`rms-picmaker` needs it), `rms-picmaker` on branch `bugfix/wfpc2-multi-extension-fits`
(the PyPI release lacks SETI/rms-picmaker PR #33), and `rms-hst-targets` (not on PyPI; its
committed caches need a source checkout). The second line swaps those GitHub copies for
editable clones beside this checkout. It must be `--no-deps`, since pip otherwise reports
a conflict with the URLs, and it must be repeated after any rerun of the first line,
which reinstalls the GitHub copies. Dependencies live in `pyproject.toml`; there is no
requirements lockfile. Installing puts the ten task commands (`hst-pipeline`,
`hst-query-products`, …) on `PATH`; they are declared in `[project.scripts]` and are what
`queue_manager/config.py`'s `TASK_INFO` dispatches. `hst-dedupe-staging` is also declared
there but is a one-time maintenance command, not a task. Finalizing also needs the PDS4
`validate` tool on `PATH` and `JAVA_HOME` set.

## Tests

```bash
pytest -m "not network"       # the fast, deterministic suite — default when iterating
pytest                        # everything, including the MAST tests
./scripts/run-all-checks.sh   # what CI enforces
```

`pythonpath = ["src"]` in `pyproject.toml` means the tests import the package without an
install, and they run from the repo root — no directory-dependent invocation any more.

Don't run the `network` tests unless asked. They query MAST live and assert exact result
counts (which drift), `test_retrieve_hst_visit.py` downloads 135 files, and
`test_pipeline_run.py` takes ~3 minutes and **deletes** the `HST_PIPELINE`/`HST_BUNDLES`
directories for proposal 16167 in `setup_method`. A network-dependent count assertion
failing is usually MAST drifting, not a regression.

Tests write into a parallel `_testing` tree (`get_program_dir_path(..., testing=True)` →
`hst_07885_testing`) so real data is untouched. Label tests write `.lblx` into
`tests/golden_copies/` and clean up in teardown.

`src/hst_pipeline/citation_tests/` holds standalone scripts, not pytest, and depends on
local data that isn't in the repo.

Tests marked `targets_data` exercise real target identification and need the committed
caches in a source checkout of `rms-hst-targets`; they are deselected in CI alongside
`network`.

## Code style

No linter or formatter is enforced. `[tool.ruff]` in `pyproject.toml` and `.flake8`
describe the intended style, and the corresponding CI steps in
`.github/workflows/run-tests.yml` are deliberately commented out until the code is clean
enough to turn them on. The tests, the docs build, the packaging check, and the Markdown
lint *are* enforced. Match the surrounding code:

- 90-column `####...` banner comment at the top of each module, with the module path and a one-line description
- Single quotes for string literals; double quotes for docstrings or strings containing apostrophes
- Docstrings use aligned `Input:` / `Returns:` blocks, not Google or NumPy style
- Soft ~90-character wrap; parenthesized, vertically aligned multi-line imports

## Layout and conventions

`src/hst_pipeline/` is the package. `pipeline/` holds thin CLI entry points, each with a
`main(argv=None)` bound to a console script, that delegate to same-named top-level modules
in the package. `queue_manager/` is a SQLite/SQLAlchemy task queue whose task order,
priorities, and command names all live in `queue_manager/config.py`.

All imports within the package are absolute and `hst_pipeline.`-qualified.

Data directories follow `<root>/hst_<nnnnn>/visit_<ss>/` — proposal ID zero-padded to 5
digits, visit 2 chars. IDs are accepted padded or unpadded (`7885` and `07885`) and
normalized internally, so don't compare them as strings when building paths. Superseded
files move to `backups/` with a `-<ymdhms>` suffix.

Each file is stored once in staging. `hst-retrieve-visit` downloads into
`<HST_STAGING>/hst_<nnnnn>/visit_<ss>/mastDownload/`, which is only an inbox;
`hst-prepare-browse` then moves every file into its collection directory,
`<HST_STAGING>/hst_<nnnnn>/<collection>/visit_<ss>/`, and `hst-label-products` labels
the files there. Run them in that order.

Every image product gets a picmaker preview: `hst-prepare-browse` runs picmaker, always
with `PICMAKER_BROWSE_OPTIONS` (`mosaic=True, footprint=7, percentiles=(0.01, 99.99),
mask_ranges=(0, 0)`), on each FITS file lacking a preview whose suffix is listed for its
instrument in `suffix_info.PICMAKER_BROWSE_SUFFIXES` (ACS, COS imaging, FOC, NICMOS, STIS
imaging, WFC3, WFPC, WFPC2) and that holds an image array. COS and STIS files qualify only
with `OBSTYPE = IMAGING` (`PICMAKER_BROWSE_REQUIREMENTS`), e.g., target acquisitions;
their labels likewise call those arrays `Array_2D_Image`, and all other COS/STIS arrays
`Array_2D_Spectrum` (`hdu_dictionary_support.science_array_class_suffix`). FOC raw images
(`d0f`) and WFPC and WFPC2 images (`c0f`/`d0f`, `c0m`/`d0m`) are also passed `hst_dq`, the
visit directory of their data quality collection (`suffix_info.HST_DQ_SUFFIXES`), since
picmaker otherwise looks for the DQ file beside the image; flagged pixels are left out of
the stretch but drawn normally (picmaker blacks them out only with `desnow`, which isn't
passed). Non-image products, those of the non-imaging instruments (FGS, FOS, GHRS, HSP)
and the COS and STIS spectra (`MAST_SPECTRUM_BROWSE_SUFFIXES`), keep MAST's JPEG/PNG
previews, filed under their MAST names; no other MAST preview is downloaded. This needs
the `rms-picmaker` from SETI/rms-picmaker PR #33, installed from a local checkout (see
Setup); the released version can't read WFPC2 files or take `mask_ranges` or `hst_dq`.
picmaker masks ACS, COS, NICMOS, STIS, and WFC3 images with the `DQ` extension of the same
file, and FOC `d0f`, WFPC, and WFPC2 images with their separate DQ files. FOC's calibrated
`c0f` images have no DQ file and aren't masked (FOC's `c1f` is an image, not a DQ file).
The `footprint` median filter, from `rms-psiops`, leaves masked pixels out of each median.
For NICMOS, picmaker leaves unmasked the pixels whose only DQ flags are 16 (grot), 64
(saturated), 1024 (source), or 2048 (signal in the 0th read), in every NICMOS file type.

Only moving-target observations (MAST's `mtFlag`) are retrieved, except that for a given
proposal `query_mast_slice` also keeps every FGS and HSP observation
(`query_utils.NON_MOVING_INSTRUMENTS`), since MAST never flags those targets as moving.
Broad queries without a proposal ID still ask MAST for moving targets only.

`hst-get-program-info` downloads the Phase 2 files (.apt, .pdf, .pro, .prop) from
STScI, and the citation info comes from the .apt or else the .pro file. Early programs
(roughly Cycles 1–3) have neither, so it falls back to the MAST proposal search page,
saved as `<nnnnn>.html` in the pipeline directory and parsed by
`citations/citation_information_from_mast.py`. That page names only the PI and gives no
year, so the citation has one author and the publication year is the end of the cycle.
`finalize_document.get_document_files` puts the proposal files, or failing those the
`<nnnnn>.html` page, in the bundle's `document/` collection; the page is copied
without the MAST site's header and footer, so it begins with `<hr>` and the title. Its
URL and retrieval date go in an `External_Reference` in the document label. `program-info.txt` (the
cleaned-up citation) stays in the pipeline directory and is not delivered.

`HST_HANDBOOKS` names the HST handbooks bundle (e.g., `/Volumes/Data-SSD/HST_HANDBOOKS`;
veraPDF in `validate` can't open paths with spaces).
It is read only when the document collection is built, not at import, and
`finalize_document.get_handbooks_dir()` raises `HandbooksError` if it is missing. A
handbook in that bundle's `document/` directory goes with every instrument named by a
`<keyword>` in its label's `Citation_Information` (case is ignored; `WFPC` is WF/PC and
`WFPC2` is WFPC2), so a volume of the HST Data Handbook that covers several instruments
lists each of them. For each of the program's instruments, every such `Product_Document`
label and the documents its `file_name`s name are copied into the program's `document/`
collection; file names don't matter, except that every handbook file's name has a dash,
which is how stale copies are recognized. The LIDs of the copied labels are listed in
`collection.csv` as secondary members (`S,<lid>`, no VID). Handbooks no longer selected
are removed on the next finalize. The tests pass `handbooks_dir` explicitly, using
`tests/utils.make_handbooks_dir()`. The bundle, data collection, and product labels refer
to each instrument's two handbooks by LID (`hst_helper.HANDBOOK_LID`, e.g.,
`urn:nasa:pds:hst_handbooks:document:wfpc-data-handbook`); the templates spell that
pattern out, and finalizing warns if a handbook it names is missing from `HST_HANDBOOKS`.

Labels use PDS4 Information Model 1.26.0.0 (`PDS4_PDS_1Q00`), the HST dictionary
`PDS4_HST_1M00_1100` (its latest release), and the Display dictionary
`PDS4_DISP_1Q00_1510`. The version appears in every template, in `hst_helper`
(`INFORMATION_MODEL_VERSION` and the schema collection's `*_LIDVID`s), in `xml/`, and in
`validate-pdart`; change them together. Reference types follow the product class:
`Product_Ancillary` labels use `ancillary_to_*`, and observational products refer to
ancillary ones with `data_to_ancillary_data` and to partially processed ones with
`data_to_partially_processed_product` (`DATA_REF_TYPES` in `PRODUCT_LABEL.lblx`).

Under `HST_BUNDLES` only the program directory has a suffix:
`<HST_BUNDLES>/hst_<nnnnn>-deliverable/` holds the manifests and `validation-report.txt`,
and the bundle itself is `<HST_BUNDLES>/hst_<nnnnn>-deliverable/hst_<nnnnn>/`
(`get_program_dir_path(..., 'bundles')` and `get_deliverable_path()` respectively).

Every label the pipeline writes, and every template in `templates/`, ends in `.lblx`,
not `.xml` (`suffix_info.LABEL_SUFFIX`, re-exported by `hst_helper`); `validate-pdart`
passes `-e lblx` so that validate looks for them. `xml/` keeps its `.xml`, `.xsd`, and
`.sch` schema files.

`templates/` and `xml/` are package data resolved relative to
`__file__`, and are declared in `[tool.setuptools.package-data]`. Anything added there
needs a matching entry or it won't ship in the wheel.

Target identification lives in the separate `rms-hst-targets` package (imported as
`targets`). `target_support.py` is a thin adapter that keeps the old
`hst_target_identifications()` contract over it, so the label-building code above it is
unchanged. That adapter is the only place allowed to touch `targets` internals.

Every task's entry point opens its logger with `hst_helper.log_support.setup_task_logger`
and closes it with `close_task_logger`. Besides the timestamped log, the logger appends
WARNING and worse to `<HST_PIPELINE>/hst_<nnnnn>/WARNINGS.log` (or
`<HST_PIPELINE>/logs/WARNINGS.log` for tasks that belong to no program). Closing a run
with any warning adds a line to `<HST_PIPELINE>/NEEDS_REVIEW.log`. The queue manager adds
lines there too, for tasks that exit nonzero or that it kills. So reserve WARNING for
what someone should look at, and log routine repairs at INFO. `run_validation` logs each
message in validate's report, except that `warning.product_not_registered` goes to INFO.

`run_validation.py` shells out to `validate-pdart`, which needs the PDS4 `validate` tool on
`PATH` and reads schemas from the package's own `xml/` directory. That tool exits 0 even
on failure, hence the `FAIL` grep in the wrapper.

`hst_helper/__init__.py` has several hardcoded placeholders marked `TODO` —
`DOI = '10.1/2'`, a `VID_DICT` that always returns 1.0. Treat them as known gaps, not as
values to trust. MAST queries are limited to data released between `START_DATE` and
`END_DATE`. `END_DATE` is `(2100, 1, 1)`, excluding no released data, unless the optional
`HST_END_DATE` environment variable (`YYYY-MM-DD`) overrides it. `tests/conftest.py`
sets `HST_END_DATE=2025-01-01` (unless it's already set) so the MAST counts the network
tests assert stay fixed; the pipeline subprocesses those tests start inherit it.

## archive/

Frozen. Don't edit it, don't include it in refactors, and don't use it as a style
reference — it follows the old PDART project's black/mypy conventions, which don't apply
to `src/`.

## Contributing

All changes go through GitHub pull requests. Code changes must come with new or updated
tests, and docstrings must be kept current. Issues are labeled `A-Bug` or `A-Enhancement`.
