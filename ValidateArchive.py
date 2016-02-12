import os.path
import re
import sys
import traceback

import pyfits

import FileArchive
import FileArchives
import HstFilename

def __validateFile(file, ctxt):
    ctxt['fileCount'] += 1

    # Extract many properties from the filename using
    # the HstFilename class and validate them.
    hstFile = HstFilename.HstFilename(file)
    assert hstFile.instrumentName() == ctxt['collectionInstrument']
    assert hstFile.visit() == ctxt['productVisit']
    ctxt['hstInternalProposalIds'].add(hstFile.hstInternalProposalId())
    try:
	proposId = pyfits.getval(file, 'PROPOSID')
	# if it exists, ensure it matches the bundleProposalId
	assert int(re.match('\A[0-9]+\Z',
			    proposId)) == ctxt['bundleProposalId']
    except:
	# if it doesn't exist; that's okay
	pass

    fileBasename = os.path.basename(file)
    fileSuffix = re.match('\A[^_]+_([^\.]+)\..*\Z',
			  fileBasename).group(1)
    assert ctxt['collectionSuffix'] == fileSuffix

    if True:
	if ctxt['fileCount'] % 10000 == 0:
	    print 'Seen %dK files.' % (ctxt['fileCount'] / 1000)

def __validateProduct(arch, product, ctxt):
    # Extract the product's visit from the filename.
    productBasename = os.path.basename(product)
    ctxt['productVisit'] = re.match('\Avisit_([a-z0-9]{2})\Z',
				    productBasename).group(1)

    ctxt['hstInternalProposalIds'] = set()
    for file in arch.walkComp(product):
	__validateFile(file, ctxt)

    # TODO It seems that hst_00000 is a grab bag of
    # lost files.  This needs to be fixed.
    # Otherwise...
    if ctxt['bundleProposalId'] != 0:

	# Assert that for any product, all of its
	# files belong to the same project, using the
	# HST internal proposal ID codes.
	assert len(ctxt['hstInternalProposalIds']) == 1, (product,
							  ctxt['hstInternalProposalIds'])

    del ctxt['hstInternalProposalIds']
    del ctxt['productVisit']

def __validateCollection(arch, collection, ctxt):
    # Extract the collection's instrument and suffix from
    # the filename.
    collectionBasename = os.path.basename(collection)
    (ctxt['collectionInstrument'],
     ctxt['collectionSuffix']) = re.match('\Adata_([a-z0-9]+)_([a-z0-9_]+)\Z',
					  collectionBasename).group(1,2)
    ctxt['collectionInstruments'].add(ctxt['collectionInstrument'])

    for product in arch.walkComp(collection):
	__validateProduct(arch, product, ctxt)

    del ctxt['collectionInstrument']
    del ctxt['collectionSuffix']

def __validateBundle(arch, bundle, ctxt):
    # Extract the bundle's proposal id from the filename
    bundleBasename = os.path.basename(bundle)
    ctxt['bundleProposalId'] = int(re.match('\Ahst_([0-9]{5})\Z',
					    bundleBasename).group(1))

    ctxt['collectionInstruments'] = set()
    for collection in arch.walkComp(bundle):
	__validateCollection(arch, collection, ctxt)

    # Assert that for any bundle (equivalently, for any
    # proposal), all its collections use the same instrument
    assert len(ctxt['collectionInstruments']) == 1, ctxt['collectionInstruments']

    del ctxt['collectionInstruments']
    del ctxt['bundleProposalId']

def validateArchive(arch):
    print 'Now validating %s' % repr(arch)
    ctxt = {'fileCount': 0}

    try:
	for bundle in arch.walkComp():
	    __validateBundle(arch, bundle, ctxt)

	print 'Test of %s PASSED (after %d files).' % (repr(arch), ctxt['fileCount'])
    except:
	print 'Test of %s FAILED (after %d files).' % (repr(arch), ctxt['fileCount'])
	print(traceback.format_exc())
	sys.exit(1)

############################################################

def __validateFileOop(file, ctxt):
    ctxt['fileCount'] += 1

    # Extract many properties from the filename using
    # the HstFilename class and validate them.
    hstFile = HstFilename.HstFilename(file)
    assert hstFile.instrumentName() == ctxt['collectionInstrument']
    assert hstFile.visit() == ctxt['productVisit']
    ctxt['hstInternalProposalIds'].add(hstFile.hstInternalProposalId())
    try:
	proposId = pyfits.getval(file, 'PROPOSID')
	# if it exists, ensure it matches the bundleProposalId
	assert int(re.match('\A[0-9]+\Z',
			    proposId)) == ctxt['bundleProposalId']
    except:
	# if it doesn't exist; that's okay
	pass

    fileBasename = os.path.basename(file)
    fileSuffix = re.match('\A[^_]+_([^\.]+)\..*\Z',
			  fileBasename).group(1)
    assert ctxt['collectionSuffix'] == fileSuffix

    if True:
	if ctxt['fileCount'] % 10000 == 0:
	    print 'Seen %dK files.' % (ctxt['fileCount'] / 1000)

def __validateProductOop(product, ctxt):
    ctxt['productVisit'] = product.visit()
    ctxt['hstInternalProposalIds'] = set()

    for file in product.fileFilepaths():
        __validateFileOop(file, ctxt)

    # TODO It seems that hst_00000 is a grab bag of
    # lost files.  This needs to be fixed.
    # Otherwise...
    if ctxt['bundleProposalId'] != 0:

	# Assert that for any product, all of its
	# files belong to the same project, using the
	# HST internal proposal ID codes.
	assert len(ctxt['hstInternalProposalIds']) == 1, (product,
                                                          ctxt['bundleProposalId'],
							  ctxt['hstInternalProposalIds'])

    del ctxt['hstInternalProposalIds']
    del ctxt['productVisit']

def __validateCollectionOop(collection, ctxt):
    ctxt['collectionInstrument'] = inst = collection.instrument()
    ctxt['collectionSuffix'] = collection.suffix()

    ctxt['collectionInstruments'].add(inst)

    for product in collection.products():
        __validateProductOop(product, ctxt)

    del ctxt['collectionInstrument']
    del ctxt['collectionSuffix']

def __validateBundleOop(bundle, ctxt):
    ctxt['bundleProposalId'] = bundle.proposalId()
    ctxt['collectionInstruments'] = set()

    for collection in bundle.collections():
        __validateCollectionOop(collection, ctxt)

    # Assert that for any bundle (equivalently, for any
    # proposal), all its collections use the same instrument
    assert len(ctxt['collectionInstruments']) == 1, ctxt['collectionInstruments']

    del ctxt['collectionInstruments']
    del ctxt['bundleProposalId']

def validateArchiveOop(arch):
    print 'Now validating %s' % repr(arch)
    ctxt = {'fileCount': 0}
    try:
        for bundle in arch.bundles():
            __validateBundleOop(bundle, ctxt)

	print ('Test of %s PASSED (after %d files).'
               % (repr(arch), ctxt['fileCount']))
    except:
	print ('Test of %s FAILED (after %d files).'
               % (repr(arch), ctxt['fileCount']))
	print(traceback.format_exc())
	sys.exit(1)


validateArchiveOop(FileArchives.getAnyArchive())
