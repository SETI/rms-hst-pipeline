| PyPI Release | Test Status | Code Coverage |
| ------------- | ------------ | -------------- |
| [![PyPI version](https://badge.fury.io/py/rms-hst-pipeline.svg)](https://badge.fury.io/py/rms-hst-pipeline) | [![Build status](https://img.shields.io/github/actions/workflow/status/SETI/rms-hst-pipeline/run-tests.yml?branch=main)](https://github.com/SETI/rms-hst-pipeline/actions) | [![Code coverage](https://img.shields.io/codecov/c/github/SETI/rms-hst-pipeline/main?logo=codecov)](https://codecov.io/gh/SETI/rms-hst-pipeline) |

# 🚀 HST Pipeline

## 🧭 Overview

The **HST Pipeline** automates the end-to-end processing of Hubble Space Telescope (HST) data for the Planetary Data System (PDS).
It supports querying data from MAST, retrieving and labeling products, preparing browse products, and generating final PDS-compliant bundles.
This tool is designed for reproducible, efficient, and configurable data pipeline execution.

---

## ⚙️ Required Setup Before Running HST Pipeline Tasks

1. Set up a virtual environment and install the package:

   ```bash
   ./scripts/setup-venv.sh
   source venv/bin/activate
   ```

   Or by hand:

   ```bash
   python -m venv venv
   source venv/bin/activate   # or `venv\Scripts\activate` on Windows
   pip install -e ".[dev]"
   ```

   Installing the package puts the task commands below on your `PATH`.

2. Set the following environment variables. All three are read at import time with no
   default; if any is unset or empty, importing a pipeline module fails immediately with
   a `MissingEnvironmentError` naming the ones that are missing:

   | Variable | Description |
   | --------- | ------------ |
   | `HST_STAGING` | Directory for downloaded files |
   | `HST_PIPELINE` | Directory for logs and program info |
   | `HST_BUNDLES` | Directory for final bundles |

---

## 🧩 Example Commands

### ▶️ Run the Full Pipeline (All Tasks)

- Run with all proposal IDs (pre-fetched from MAST with the *True moving target* flag):

  ```bash
  hst-pipeline
  ```

- Run a continuous process for a proposal ID that re-queues every 30 days::

  ```bash
  hst-pipeline --proposal-ids 07885 --run-forever
  ```

- Run with a single proposal ID:

  ```bash
  hst-pipeline --proposal-ids 07885
  ```

- Run with multiple proposal IDs:

  ```bash
  hst-pipeline --proposal-ids 13736 05167 10341 14930 06679 05837
  ```

- Run with a specific number of subprocesses and a maximum allowed runtime for each task (in seconds):

  ```bash
  hst-pipeline --proposal-ids 07885 13736 --max-subproc 30 --max-time 1860
  ```

- Run with a recreated task queue:

  ```bash
  hst-pipeline --proposal-ids 07885 --recreate-queue
  ```

---

### 🧠 Run Individual Tasks (Example: Proposal ID `7885`)

These are the commands executed internally when running `hst-pipeline`. They are
console scripts declared in `pyproject.toml`, and the task queue dispatches them by
name, so they can be run from any directory:

```bash
hst-query-moving-targets --proposal-ids 7885
hst-query-products --proposal-id 7885
hst-get-program-info --proposal-id 7885
hst-retrieve-visit --proposal-id 7885 --vi 01
hst-retrieve-visit --proposal-id 7885 --vi 02
hst-retrieve-visit --proposal-id 7885 --vi 03
hst-prepare-browse --proposal-id 7885 --vi 01
hst-prepare-browse --proposal-id 7885 --vi 02
hst-prepare-browse --proposal-id 7885 --vi 03
hst-label-products --proposal-id 7885 --vi 01
hst-label-products --proposal-id 7885 --vi 02
hst-label-products --proposal-id 7885 --vi 03
hst-finalize-bundle --proposal-id 7885
```

`hst-prepare-browse` moves each visit's downloads out of `visit_<ss>/mastDownload/` into
their collection directories, so it must run before `hst-label-products`, which labels
the files there. Staging directories built by older versions of the pipeline hold a
second copy of every file under `mastDownload/`; `hst-dedupe-staging --proposal-id 7885`
reports those copies, and `--delete` removes each visit's copy once every file in it is
verified to be identical to its filed copy.

---

✅ **Tip:** Use `--help` with any command (e.g., `hst-pipeline --help`) to view all available options and arguments.

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
rebuild the program directories under `HST_PIPELINE` and `HST_BUNDLES`. They are excluded
from the pull request gate and run weekly by `.github/workflows/network-tests.yml`.

### Continuous Integration

| Workflow | Trigger | What it does |
| -------- | ------- | ------------ |
| `run-tests.yml` | Push, PR, weekly | Non-network tests on Linux and macOS across Python 3.11–3.15, plus coverage, docs, and packaging checks |
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
