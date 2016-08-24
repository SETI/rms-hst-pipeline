"""
This module augments the :class:`pdart.reductions.Reduction.Reduction`
with some type contracts to verify that the use of types in the code
is correct.  Unfortunately, currently we can only check that the
``get_reduced_xxx()`` functions are functions; their arguments are
*not* in turn typechecked.

*This is not currently used (as of 2016-08-24).*
"""
from contracts import ContractsMeta, contract, new_contract

from pdart.pds4.Archive import *
from pdart.pds4.File import *
from pdart.pds4.LID import *
from pdart.reductions.Reduction import *

try:
    new_contract('func', lambda(f): hasattr(f, '__call__'))
except ValueError:
    pass


class DbCReduction(Reduction):
    """
    A :class:`pdart.reductions.Reduction.Reduction` augmented with
    some typechecks.
    """
    __metaclass__ = ContractsMeta

    @contract(archive_root='str', get_reduced_bundles='func')
    def reduce_archive(self, archive_root, get_reduced_bundles):
        pass

    @contract(archive=Archive, lid=LID, get_reduced_collections='func')
    def reduce_bundle(self, archive, lid, get_reduced_collections):
        pass

    @contract(archive=Archive, lid=LID, get_reduced_products='func')
    def reduce_collection(self, archive, lid, get_reduced_products):
        pass

    @contract(archive=Archive, lid=LID, get_reduced_fits_files='func')
    def reduce_product(self, archive, lid, get_reduced_fits_files):
        pass

    @contract(file=File, get_reduced_hdus='func')
    def reduce_fits_file(self, file, get_reduced_hdus):
        pass

    @contract(n='int,>=0', hdu='*',
              get_reduced_header_unit='func',
              get_reduced_data_unit='func')
    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        pass

    @contract(n='int,>=0', header_unit='*')
    def reduce_header_unit(self, n, header_unit):
        pass

    @contract(n='int,>=0', data_unit='*')
    def reduce_data_unit(self, n, data_unit):
        pass
