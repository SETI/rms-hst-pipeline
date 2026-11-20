##########################################################################################
# tests/test_hst_helper_env.py
#
# Tests for the required-environment-variable check in hst_helper.
##########################################################################################

import pytest

from hst_pipeline.hst_helper import (DEFAULT_END_DATE,
                                     MissingEnvironmentError,
                                     _read_end_date,
                                     _read_hst_dir,
                                     _REQUIRED_ENV)

ALL_VARS = [var for _, var, _ in _REQUIRED_ENV]


def set_env(monkeypatch, present):
    """Set every required variable named in `present` and unset the rest.

    Inputs:
        monkeypatch    the pytest monkeypatch fixture.
        present        an iterable of variable names that should be set.
    """
    for var in ALL_VARS:
        if var in present:
            monkeypatch.setenv(var, f'/tmp/{var.lower()}')
        else:
            monkeypatch.delenv(var, raising=False)


def test_all_set_returns_mapping(monkeypatch):
    set_env(monkeypatch, ALL_VARS)
    assert _read_hst_dir() == {'staging':  '/tmp/hst_staging',
                               'pipeline': '/tmp/hst_pipeline',
                               'bundles':  '/tmp/hst_bundles'}


def test_all_missing_names_every_variable(monkeypatch):
    set_env(monkeypatch, [])
    with pytest.raises(MissingEnvironmentError) as exc:
        _read_hst_dir()
    message = str(exc.value)
    for var in ALL_VARS:
        assert var in message
    assert 'variables' in message          # plural
    assert 'export HST_STAGING=' in message


@pytest.mark.parametrize('missing', ALL_VARS)
def test_one_missing_names_only_that_variable(monkeypatch, missing):
    set_env(monkeypatch, [v for v in ALL_VARS if v != missing])
    with pytest.raises(MissingEnvironmentError) as exc:
        _read_hst_dir()
    message = str(exc.value)
    assert missing in message
    for other in ALL_VARS:
        if other != missing:
            assert other not in message
    assert 'variable to be set' in message  # singular


def test_empty_value_counts_as_missing(monkeypatch):
    """An empty string is not a usable directory, so it must not be accepted."""
    set_env(monkeypatch, ALL_VARS)
    monkeypatch.setenv('HST_BUNDLES', '')
    with pytest.raises(MissingEnvironmentError) as exc:
        _read_hst_dir()
    assert 'HST_BUNDLES' in str(exc.value)


def test_error_is_a_runtime_error():
    """Callers that catch RuntimeError should keep working."""
    assert issubclass(MissingEnvironmentError, RuntimeError)

##########################################################################################


def test_end_date_defaults_to_far_future(monkeypatch):
    # Without HST_END_DATE, no released data is excluded.
    monkeypatch.delenv('HST_END_DATE', raising=False)
    assert _read_end_date() == DEFAULT_END_DATE == (2100, 1, 1)
    monkeypatch.setenv('HST_END_DATE', '  ')
    assert _read_end_date() == DEFAULT_END_DATE


def test_end_date_from_environment(monkeypatch):
    monkeypatch.setenv('HST_END_DATE', '2025-01-01')
    assert _read_end_date() == (2025, 1, 1)


@pytest.mark.parametrize('value', ['2025', '2025/01/01', '2025-13-01', 'soon'])
def test_end_date_rejects_bad_values(monkeypatch, value):
    monkeypatch.setenv('HST_END_DATE', value)
    with pytest.raises(ValueError, match='HST_END_DATE'):
        _read_end_date()
