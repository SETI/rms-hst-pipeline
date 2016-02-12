import os.path
import re
import sys
import traceback

import pyfits

import FileArchive
import FileArchives
import HstFilename

def validateArchive(arch):
    print 'Now validating %s' % repr(arch)
    fileCount = 0

    try:
	for bundle in arch.walkComp():
	    # Extract the bundle's proposal id from the filename
	    bundleBasename = os.path.basename(bundle)
	    bundleProposalId = int(re.match('\Ahst_([0-9]{5})\Z',
					    bundleBasename).group(1))

	    collectionInstruments = set()
	    for collection in arch.walkComp(bundle):
		# Extract the collection's instrument and suffix from
		# the filename.
		collectionBasename = os.path.basename(collection)
		(collectionInstrument,
		 collectionSuffix) = re.match('\Adata_([a-z0-9]+)_([a-z0-9_]+)\Z',
					      collectionBasename).group(1,2)
		collectionInstruments.add(collectionInstrument)

		for product in arch.walkComp(collection):
		    # Extract the product's visit from the filename.
		    productBasename = os.path.basename(product)
		    productVisit = re.match('\Avisit_([a-z0-9]{2})\Z',
					    productBasename).group(1)

		    hstInternalProposalIds = set()
		    for file in arch.walkComp(product):
			fileCount = fileCount + 1

			# Extract many properties from the filename using
			# the HstFilename class and validate them.
			fileBasename = os.path.basename(file)
			hstFile = HstFilename.HstFilename(file)
			assert hstFile.instrumentName() == collectionInstrument
			assert hstFile.visit() == productVisit
			hstInternalProposalIds.add(hstFile.hstInternalProposalId())
			try:
			    proposId = pyfits.getval(file, 'PROPOSID')
			    # if it exists, ensure it matches the
			    # bundleProposalId
			    assert int(re.match('\A[0-9]+\Z',
						proposId)) == bundleProposalId
			except:
			    # if it doesn't exist; that's okay
			    pass

			fileSuffix = re.match('\A[^_]+_([^\.]+)\..*\Z',
					      fileBasename).group(1)
			assert collectionSuffix == fileSuffix

			if True:
			    if fileCount % 10000 == 0:
				print 'Seen %dK files.' % (fileCount / 1000)

		    # TODO It seems that hst_00000 is a grab bag of
		    # lost files.  This needs to be fixed.
		    # Otherwise...
		    if bundleProposalId != 0:

			# Assert that for any product, all of its
			# files belong to the same project, using the
			# HST internal proposal ID codes.
			assert len(hstInternalProposalIds) == 1, (product,
								  hstInternalProposalIds)

	    # Assert that for any bundle (equivalently, for any
	    # proposal), all its collections use the same instrument
	    assert len(collectionInstruments) == 1, collectionInstruments
	print 'Test of %s PASSED (after %d files).' % (repr(arch), fileCount)
    except:
	print 'Test of %s FAILED (after %d files).' % (repr(arch), fileCount)
	print(traceback.format_exc())
	sys.exit(1)

validateArchive(FileArchives.getAnyArchive())
