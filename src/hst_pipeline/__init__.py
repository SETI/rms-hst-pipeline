##########################################################################################
# hst_pipeline/__init__.py
#
# Package root. Importing this module is deliberately cheap: the pipeline's environment
# variables are read by hst_pipeline.hst_helper, not here, so `import hst_pipeline` works
# without HST_STAGING, HST_PIPELINE, and HST_BUNDLES being set.
##########################################################################################

try:
    # Written at build time by setuptools_scm, and absent from a source checkout that has
    # never been built.
    from hst_pipeline._version import version as __version__
except ImportError:  # pragma: no cover
    try:
        from importlib.metadata import version as _version
        __version__ = _version('rms-hst-pipeline')
    except Exception:
        __version__ = 'unknown'

__all__ = ['__version__']

##########################################################################################
