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
```

Or `pip install -e ".[dev]"`. Dependencies live in `pyproject.toml`; there is no
requirements lockfile. Installing puts the ten task commands (`hst-pipeline`,
`hst-query-products`, …) on `PATH`; they are declared in `[project.scripts]` and are what
`queue_manager/config.py`'s `TASK_INFO` dispatches. `hst-dedupe-staging` is also declared
there but is a one-time maintenance command, not a task.

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
`hst_07885_testing`) so real data is untouched. Label tests write `.xml` into
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
instrument in `suffix_info.PICMAKER_BROWSE_SUFFIXES` (ACS, FOC, NICMOS, STIS imaging,
WFC3, WFPC, WFPC2). Non-image products, those of the non-imaging instruments (COS, FGS,
FOS, GHRS, HSP) and the STIS spectra, keep MAST's JPEG/PNG previews, filed under their
MAST names; no other MAST preview is downloaded. This needs the `rms-picmaker` from
SETI/rms-picmaker PR #33 (or a local checkout, installed with `pip install --no-deps -e
../rms-picmaker`); the released version can't read WFPC2 files or take `mask_ranges`.

Only moving-target observations (MAST's `mtFlag`) are retrieved, except that for a given
proposal `query_mast_slice` also keeps every FGS and HSP observation
(`query_utils.NON_MOVING_INSTRUMENTS`), since MAST never flags those targets as moving.
Broad queries without a proposal ID still ask MAST for moving targets only.

Under `HST_BUNDLES` only the program directory has a suffix:
`<HST_BUNDLES>/hst_<nnnnn>-deliverable/` holds the manifests and `validation-report.txt`,
and the bundle itself is `<HST_BUNDLES>/hst_<nnnnn>-deliverable/hst_<nnnnn>/`
(`get_program_dir_path(..., 'bundles')` and `get_deliverable_path()` respectively).

`templates/` and `xml/` are package data resolved relative to
`__file__`, and are declared in `[tool.setuptools.package-data]`. Anything added there
needs a matching entry or it won't ship in the wheel.

Target identification lives in the separate `rms-hst-targets` package (imported as
`targets`). `target_support.py` is a thin adapter that keeps the old
`hst_target_identifications()` contract over it, so the label-building code above it is
unchanged. That adapter is the only place allowed to touch `targets` internals.

`run_validation.py` shells out to `validate-pdart`, which needs the PDS4 `validate` tool on
`PATH` and reads schemas from the package's own `xml/` directory. That tool exits 0 even
on failure, hence the `FAIL` grep in the wrapper.

`hst_helper/__init__.py` has several hardcoded placeholders marked `TODO` —
`END_DATE = (2025, 1, 1)`, `DOI = '10.1/2'`, a `VID_DICT` that always returns 1.0. Treat
them as known gaps, not as values to trust.

## archive/

Frozen. Don't edit it, don't include it in refactors, and don't use it as a style
reference — it follows the old PDART project's black/mypy conventions, which don't apply
to `src/`.

## Contributing

All changes go through GitHub pull requests. Code changes must come with new or updated
tests, and docstrings must be kept current. Issues are labeled `A-Bug` or `A-Enhancement`.
