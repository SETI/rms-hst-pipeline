| PyPI Release | Test Status | Code Coverage |
| ------------- | ------------ | -------------- |
| [![PyPI version](https://badge.fury.io/py/rms-hst-pipeline.svg)](https://badge.fury.io/py/rms-hst-pipeline) | [![Build status](https://img.shields.io/github/actions/workflow/status/SETI/rms-hst-pipeline/run-tests.yml?branch=main)](https://github.com/SETI/rms-hst-pipeline/actions) | [![Code coverage](https://img.shields.io/codecov/c/github/SETI/rms-hst-pipeline/main?logo=codecov)](https://codecov.io/gh/SETI/rms-hst-pipeline) |

# 🚀 HST Pipeline

## 🧭 Overview

The **HST Pipeline** automates the end-to-end processing of Hubble Space Telescope (HST) data for the Planetary Data System (PDS).
It supports querying data from MAST, retrieving and labeling products, preparing browse products, and generating final PDS-compliant bundles.
This tool is designed for reproducible, efficient, and configurable data pipeline execution.

---

## 📦 Installation

### Prerequisites

| Requirement | Notes |
| ----------- | ----- |
| Python 3.12 or later | `rms-picmaker` requires 3.12 |
| git | Three of the dependencies are cloned from GitHub (below) |
| Java 17 or later | Runs the PDS4 `validate` tool, e.g., `brew install openjdk` |
| PDS4 [`validate`](https://github.com/NASA-PDS/validate/releases) 5.0.0 or later | `hst-finalize-bundle` runs it on every bundle; see [Installing validate](#installing-validate) |

### Repositories cloned from GitHub

Most dependencies come from PyPI and are installed automatically, but three must be
cloned from GitHub and installed from those checkouts:

| Repository | Why it is not installed from PyPI |
| ---------- | --------------------------------- |
| [`rms-psiops`](https://github.com/SETI/rms-psiops) | Not published on PyPI. `rms-picmaker` needs it for the median filter behind its `footprint` stretch. |
| [`rms-picmaker`](https://github.com/SETI/rms-picmaker), branch `bugfix/wfpc2-multi-extension-fits` | The PyPI release (2.2.0) cannot read multi-extension WFPC2 files, does not take `mask_ranges` or `hst_dq`, and lacks the data quality masking of HST images (SETI/rms-picmaker PR #33). |
| [`rms-hst-targets`](https://github.com/SETI/rms-hst-targets) | Not published on PyPI. Target identification, and the tests marked `targets_data`, use the caches committed to its repository, which only a source checkout has. |

`pyproject.toml` names all three by their GitHub URLs, so `pip install` fetches copies of
them by itself, but those copies are not editable and, in the case of `rms-hst-targets`,
lack its caches. Clone all four repositories side by side:

```bash
mkdir -p ~/GitHub && cd ~/GitHub
git clone https://github.com/SETI/rms-hst-pipeline.git
git clone https://github.com/SETI/rms-psiops.git
git clone -b bugfix/wfpc2-multi-extension-fits https://github.com/SETI/rms-picmaker.git
git clone https://github.com/SETI/rms-hst-targets.git
```

### Python environment

Create a virtual environment in the pipeline checkout and install the packages into it
in two steps:

```bash
cd ~/GitHub/rms-hst-pipeline
python3 -m venv venv
source venv/bin/activate                  # or `venv\Scripts\activate` on Windows
pip install --upgrade pip

# 1. The pipeline, with all its dependencies and the dev and docs extras
pip install -e ".[dev]"

# 2. Replace the GitHub copies that step 1 fetched with the local checkouts
pip install --no-deps -e ../rms-psiops -e ../rms-picmaker -e ../rms-hst-targets

pip check                                 # should report no broken requirements
pip list --editable                       # should list all four packages
```

Step 1 can be replaced by `./scripts/setup-venv.sh`, which creates the venv and runs it.
Step 2 needs `--no-deps`: without it pip insists on the GitHub URLs in `pyproject.toml`
and reports a conflict. Repeat step 2 whenever step 1 is rerun, since step 1 replaces the
local checkouts with GitHub copies again. After pulling new commits in any of the four
checkouts, nothing needs reinstalling, since all four are editable; rerun both steps only
if a `pyproject.toml` has gained a dependency.

Installing the pipeline puts the task commands described under *Running the Pipeline*,
below, on your `PATH`.

### Installing validate

Download the `validate` release archive from
[NASA-PDS/validate](https://github.com/NASA-PDS/validate/releases), unpack it, and put
its `bin/` directory on your `PATH`. Its launcher needs `JAVA_HOME`. For example, on macOS
with Homebrew's OpenJDK:

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk
export PATH="$HOME/opt/validate-5.0.0/bin:$JAVA_HOME/bin:$PATH"
validate --version
```

`validate` checks the PDF/A documents with veraPDF, which cannot open a path containing a
space, so keep `HST_BUNDLES` and `HST_HANDBOOKS` on paths without spaces.

### Environment variables

Set these, e.g., in `~/.zshrc` or `~/.bashrc`:

| Variable | Required | Description |
| -------- | -------- | ----------- |
| `HST_STAGING` | Yes, at import | Where MAST downloads land and are filed into collections |
| `HST_PIPELINE` | Yes, at import | Logs, program information, and the task queue, `task_queue.db` |
| `HST_BUNDLES` | Yes, at import | The finished bundles, in `<HST_BUNDLES>/hst_<nnnnn>-deliverable/` |
| `HST_HANDBOOKS` | Yes, when finalizing | The HST handbooks bundle, e.g., `/Volumes/Data-SSD/HST_HANDBOOKS` |
| `HST_END_DATE` | No | `YYYY-MM-DD`; limits MAST queries to data released before that date. Unset, no released data is excluded |
| `JAVA_HOME` | Yes, when finalizing | Needed by the `validate` launcher |

`HST_STAGING`, `HST_PIPELINE`, and `HST_BUNDLES` are read at import time with no
defaults; if any is unset or empty, importing a pipeline module fails immediately with a
`MissingEnvironmentError` naming the ones that are missing. `HST_HANDBOOKS` is read only
when a bundle is finalized, which fails with a `HandbooksError` if it is unset or names no
directory.

```bash
export HST_STAGING=/Volumes/Data-SSD/HST_STAGING
export HST_PIPELINE=/Volumes/Data-SSD/HST_PIPELINE
export HST_BUNDLES=/Volumes/Data-SSD/HST_BUNDLES
export HST_HANDBOOKS=/Volumes/Data-SSD/HST_HANDBOOKS
```

### The HST handbooks bundle

Every program's `document/` collection includes copies of the handbooks for its
instruments, taken from the `document/` directory of the `HST_HANDBOOKS` bundle and
listed in the collection inventory as secondary members, by LID. A handbook is selected
by the `<keyword>` elements in the `Citation_Information` of its label, not by its file
name: it goes into every bundle containing data from an instrument that one of its
keywords names (ignoring case). The label is copied along with each document named by its
`<file_name>` elements. So, for example, `hst-data-handbook-volume-1.lblx`, whose only
instrument keyword is `FOC`, goes into bundles with FOC data, and
`hst-data-handbook-volume-2.lblx`, with `FOS`, `GHRS`, `HSP`, and `WFPC`, into bundles
with data from any of those.

Keywords must use the pipeline's instrument ids, which are the FITS `INSTRUME` values:
`ACS`, `COS`, `FGS`, `FOC`, `FOS`, `GHRS`, `HSP`, `NICMOS`, `STIS`, `WFC3`, `WFPC` (for
WF/PC, the original Wide Field and Planetary Camera), and `WFPC2`. The bundle, data
collection, and product labels also refer to each instrument's two handbooks by the LIDs
`urn:nasa:pds:hst_handbooks:document:<inst>-data-handbook` and
`…:<inst>-instrument-handbook`, with `<inst>` in lower case (e.g., `wfpc-data-handbook`),
so those labels must exist. Finalizing warns if one is missing. Name every handbook file
with a dash (`WFPC-Data-Handbook-1997-v3-final.pdf`, `wfpc-data-handbook.lblx`); a file
with a dash in a program's `document/` collection that is no longer selected is removed
the next time the bundle is finalized.

### Verifying the installation

```bash
pytest -m "not network"                   # about 365 tests; takes seconds
```

See [Running the Tests](#running-the-tests) for the other test categories.

---

## ▶️ Running the Pipeline

Every task is a console script that can be run on its own; `hst-pipeline` runs them all
through a task queue. Every command takes `--help`, `--log <path>` (to choose the log
file), and `--quiet` (to log only to the file). Proposal IDs can be given with or without
leading zeros (`7885` or `07885`).

### The full pipeline

`hst-pipeline` keeps a queue of tasks in `<HST_PIPELINE>/task_queue.db`, runs each as a
separate subprocess, and exits when the queue is empty.

- Run with all proposal IDs (pre-fetched from MAST with the *True moving target* flag):

  ```bash
  hst-pipeline
  ```

- Run with one or more proposal IDs:

  ```bash
  hst-pipeline --proposal-ids 07885
  hst-pipeline --proposal-ids 13736 05167 10341 14930 06679 05837
  ```

- Limit the number of tasks running at once (default 100) and the time each may run
  before it is killed, in seconds (default 1800; downloading a large visit can take
  longer):

  ```bash
  hst-pipeline --proposal-ids 07885 13736 --max-subproc 4 --max-time 7200
  ```

- Start over with an empty task queue:

  ```bash
  hst-pipeline --proposal-ids 07885 --recreate-queue
  ```

- Keep running after the queue empties, re-querying every 30 days:

  ```bash
  hst-pipeline --proposal-ids 07885 --run-forever
  ```

- Keep the staged files. Once the queue is empty, `hst-pipeline` deletes each program's
  `mastDownload/` and collection directories under `HST_STAGING`, since their files are
  now in the bundle. `--nocleanup` keeps them, which is needed to relabel or rebuild a
  bundle later from staging:

  ```bash
  hst-pipeline --proposal-ids 07885 --nocleanup
  ```

`finalize-bundle` is queued with a one-day delay after the visits of a program are
queued, so that all of them can finish first.

### Individual tasks

These are the commands `hst-pipeline` runs, in order, for each program (here proposal
`7885`, whose visits are `01`, `02`, and `03`):

```bash
hst-query-moving-targets --proposal-ids 7885   # find the programs with moving targets
hst-query-products --proposal-id 7885          # list the accepted products of each visit
hst-get-program-info --proposal-id 7885        # Phase 2 files and citation information
hst-retrieve-visit --proposal-id 7885 --vi 01  # download a visit from MAST
hst-prepare-browse --proposal-id 7885 --vi 01  # file the downloads; make the previews
hst-label-products --proposal-id 7885 --vi 01  # label a visit's files
hst-finalize-bundle --proposal-id 7885         # build and validate the bundle
```

Run `hst-retrieve-visit`, `hst-prepare-browse`, and `hst-label-products` for each visit
before `hst-finalize-bundle`. `hst-update-program --proposal-id 7885 --visits 01 02 03`
and `hst-update-visit --proposal-id 7885 --vi 01` only add these tasks to the task queue,
for a running `hst-pipeline` to execute.

`hst-prepare-browse` moves each visit's downloads out of `visit_<ss>/mastDownload/` into
their collection directories, `<HST_STAGING>/hst_<nnnnn>/<collection>/visit_<ss>/`, so
it must run before `hst-label-products`, which labels the files there. It also makes a
picmaker preview of every image product that lacks one; to remake a preview, delete it
and rerun `hst-prepare-browse` for that visit.

To rebuild a bundle from what is still staged (see `--nocleanup`), for example after a
change to the label templates, the handbooks, or the previews, relabel every visit and
then finalize:

```bash
for vi in 01 02 03; do hst-label-products --proposal-id 7885 --vi $vi; done
hst-finalize-bundle --proposal-id 7885
```

Staging directories built by older versions of the pipeline hold a second copy of every
file under `mastDownload/`; `hst-dedupe-staging --proposal-id 7885` reports those copies,
and `--delete` removes each visit's copy once every file in it is verified to be
identical to its filed copy.

### Outputs and logs

| What | Where |
| ---- | ----- |
| Staged files and their labels | `<HST_STAGING>/hst_<nnnnn>/<collection>/visit_<ss>/` |
| The bundle | `<HST_BUNDLES>/hst_<nnnnn>-deliverable/hst_<nnnnn>/` |
| Manifests and `validation-report.txt` | `<HST_BUNDLES>/hst_<nnnnn>-deliverable/` |
| Program-level logs | `<HST_PIPELINE>/hst_<nnnnn>/logs/<task>-<yyyy-mm-ddThh-mm-ss>.log` |
| Visit-level logs | `<HST_PIPELINE>/hst_<nnnnn>/visit_<ss>/logs/` |

`validate` exits successfully even when products fail, so check the summary at the end
of `validation-report.txt`, or the log of `hst-finalize-bundle`, which reports
`Validation contains failures.` On macOS, delete any `.DS_Store` files that Finder writes
into a bundle before validating it; `validate` rejects their names.

### Limiting processes and threads

The pipeline's parallelism comes from three places, all of which can be limited.

1. **Concurrent tasks.** `hst-pipeline` runs up to `--max-subproc` tasks at once
   (default 100). Each task is a separate process, so this is the main control:

   ```bash
   hst-pipeline --proposal-ids 07885 --max-subproc 4
   ```

   Running the individual commands by hand runs one task at a time.

2. **Numerical library threads.** NumPy and SciPy may run their linear algebra libraries
   (OpenBLAS, Accelerate, or MKL) on every core in every process. Setting these before
   running `hst-pipeline` or any task limits each process to one thread; the task
   subprocesses inherit them:

   ```bash
   export OMP_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1
   export MKL_NUM_THREADS=1
   export VECLIB_MAXIMUM_THREADS=1
   export NUMEXPR_NUM_THREADS=1
   ```

3. **Java threads in `validate`.** The JVM sizes its garbage-collection and compiler
   thread pools by the number of cores. Limit it with `JAVA_TOOL_OPTIONS`:

   ```bash
   export JAVA_TOOL_OPTIONS="-XX:ActiveProcessorCount=2"
   ```

Memory matters as much as threads: each `validate` run reserves a 2–4 GB Java heap, and
picmaker's `footprint` filter (from `rms-psiops`) sizes its working buffers by up to half
of the system's memory. When raising `--max-subproc`, allow for several finalize and
prepare-browse tasks running at once.

---

## 🧪 Development

### Repository Layout

| Path | Contents |
| ---- | -------- |
| `src/hst_pipeline/` | The pipeline package: task modules, the `pipeline/` entry points, the `queue_manager/` task queue, label templates, browse recipes, and the PDS4 schemas under `xml/`. Target identification is delegated to the separate [`rms-hst-targets`](https://github.com/SETI/rms-hst-targets) package via `target_support.py` |
| `tests/` | The pytest suite |
| `scripts/` | `setup-venv.sh` and `run-all-checks.sh` |
| `docs/` | Sphinx documentation, including the architecture spec |
| `archive/` | Frozen legacy PDART code, kept for reference only |

### Running the Checks

```bash
./scripts/run-all-checks.sh          # what CI enforces
./scripts/run-all-checks.sh --help   # individual checks
```

### Running the Tests

```bash
pytest -m "not network"              # the fast, deterministic suite
pytest                               # everything, including the MAST tests
```

Tests marked `network` query MAST live: they download real products, assert result counts
that drift as the archive grows, and in the case of `test_pipeline_run.py` delete and
rebuild the program directories under `HST_PIPELINE` and `HST_BUNDLES`. To keep those
counts stable, `tests/conftest.py` sets `HST_END_DATE=2025-01-01` unless it is already
set, so the tests see only data released before 2025. They are excluded from the pull
request gate and run weekly by `.github/workflows/network-tests.yml`.

### Continuous Integration

| Workflow | Trigger | What it does |
| -------- | ------- | ------------ |
| `run-tests.yml` | Push, PR, weekly | Non-network tests on Linux and macOS across Python 3.12–3.15, plus coverage, docs, and packaging checks |
| `network-tests.yml` | Weekly, manual | The MAST-dependent tests |
| `audit.yml` | Weekly, manual | `pip-audit` against the dependency set |
| `publish_to_pypi.yml` | GitHub release | Builds and publishes to PyPI |
| `publish_to_test_pypi.yml` | Manual | Builds and publishes to Test PyPI |

Linting is configured but not yet enforced; see the staged-in steps in `run-tests.yml`.
The documentation build, the packaging check, and the Markdown lint are enforced today.

### Releasing

The version is derived from git tags by `setuptools_scm`; there is no version string to
edit. An untagged checkout builds as a development version such as `0.1.dev1160`, and
`pip install rms-hst-pipeline` will **not** resolve to a development version, since pip
skips pre-releases by default. A release therefore needs a tag:

1. Tag the commit (`git tag v1.0.0 && git push --tags`).
2. Publish a GitHub release for that tag. `publish_to_pypi.yml` builds and uploads it.

To rehearse without touching PyPI, run `publish_to_test_pypi.yml` manually. To check the
artifacts locally:

```bash
python -m build && python -m twine check dist/*
```

Both publish workflows check out with `fetch-depth: 0`, which `setuptools_scm` needs in
order to see the tags.

---
