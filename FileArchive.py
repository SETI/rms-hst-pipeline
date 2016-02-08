import os
import os.path
import re

import HstFilename

class FileArchive:
    def __init__(self, root):
        assert os.path.exists(root)
        self.root = root

    def __str__(self):
        return repr(self.root)

    def __repr__(self):
        return 'FileArchive(%s)' % repr(self.root)

    #### Verifying parts

    @staticmethod
    def isValidInstrument(inst):
        return inst in ['acs', 'wfc3', 'wfpc2']

    @staticmethod
    def isValidProposal(prop):
        return isinstance(prop, int) and 0 <= prop and prop <= 99999

    @staticmethod
    def isValidVisit(vis):
        try:
            return re.match('\A[a-z0-9][a-z0-9]\Z', vis) is not None
        except:
            return False

    #### Creating filepaths from parts

    def instrumentFilepath(self, inst):
        inst = inst.lower()
        assert FileArchive.isValidInstrument(inst)
        return os.path.join(self.root, inst)

    def proposalFilepath(self, inst, prop):
        assert FileArchive.isValidProposal(prop), prop
        return os.path.join(self.instrumentFilepath(inst), "hst_%05d" % prop)

    def visitFilepath(self, inst, prop, vis):
        vis = vis.lower()
        assert FileArchive.isValidVisit(vis)
        return os.path.join(self.proposalFilepath(inst, prop),
                            "visit_%s" % vis)

    #### Walking parts of the archive

    def walkInstruments(self):
        for d in os.listdir(self.root):
            if (os.path.isdir(os.path.join(self.root, d))
                and FileArchive.isValidInstrument(d)):
                yield d

    def walkProposals(self):
        for inst in self.walkInstruments():
            instFilepath = self.instrumentFilepath(inst)
            for d in os.listdir(instFilepath):
                if os.path.isdir(os.path.join(instFilepath, d)):
                    yield (inst, int(d[4:]))

    def walkVisits(self):
        for (inst, prop) in self.walkProposals():
            propFilepath = self.proposalFilepath(inst, prop)
            for d in os.listdir(propFilepath):
                if (os.path.isdir(os.path.join(propFilepath, d))
                    and FileArchive.isValidVisit(d[6:])):
                    yield (inst, prop, d[6:])

    def walkFiles(self):
        for (inst, prop, vis) in self.walkVisits():
            visFilepath = self.visitFilepath(inst, prop, vis)
            for f in os.listdir(visFilepath):
                if (os.path.isfile(os.path.join(visFilepath, f))
                    and f[0] != '.'):
                    yield (inst, prop, vis, f)

    #### Listing parts of the archive

    def listInstruments(self):
        return sorted([d for d in os.listdir(self.root)
                       if (os.path.isdir(os.path.join(self.root, d))
                           and FileArchive.isValidInstrument(d))])

    def listProposals(self, inst):
        inst = inst.lower()
        instFilepath = self.instrumentFilepath(inst)
        return sorted([int(d[4:]) for d in os.listdir(instFilepath)
                       if os.path.isdir(os.path.join(instFilepath, d))])

    def listVisits(self, inst, prop):
        propFilepath = self.proposalFilepath(inst, prop)
        return sorted([v[6:] for v in os.listdir(propFilepath)
                       if (os.path.isdir(os.path.join(propFilepath, v))
                           and FileArchive.isValidVisit(v[6:]))])

    def listFiles(self, inst, prop, vis):
        visFilepath = self.visitFilepath(inst, prop, vis)
        return sorted(f for f in os.listdir(visFilepath)
                      if (os.path.isfile(os.path.join(visFilepath, f))
                          and f[0] != '.'))

############################################################

import shutil
import tempfile
import unittest

class TestFileArchive(unittest.TestCase):
    def testInit(self):
        with self.assertRaises(Exception):
            FileArchive(None)
        with self.assertRaises(Exception):
            FileArchive("I'm/betting/this/directory/doesn't/exist")

        FileArchive('/')	# guaranteed to exist

        # but try with another directory
        tempdir = tempfile.mkdtemp()
        try:
            FileArchive(tempdir)
        finally:
            shutil.rmtree(tempdir)

    def testStr(self):
        tempdir = tempfile.mkdtemp()
        a = FileArchive(tempdir)
        self.assertEqual(repr(tempdir), str(a))

    def testRepr(self):
        tempdir = tempfile.mkdtemp()
        a = FileArchive(tempdir)
        self.assertEqual('FileArchive(%s)' % repr(tempdir), repr(a))

    def testIsValidInstrument(self):
        self.assertTrue(FileArchive.isValidInstrument('wfc3'))
        self.assertTrue(FileArchive.isValidInstrument('wfpc2'))
        self.assertTrue(FileArchive.isValidInstrument('acs'))
        self.assertFalse(FileArchive.isValidInstrument('Acs'))
        self.assertFalse(FileArchive.isValidInstrument('ACS'))
        self.assertFalse(FileArchive.isValidInstrument('ABC'))
        self.assertFalse(FileArchive.isValidInstrument(None))

    def testIsValidProposal(self):
        self.assertFalse(FileArchive.isValidProposal(-1))
        self.assertTrue(FileArchive.isValidProposal(0))
        self.assertTrue(FileArchive.isValidProposal(1))
        self.assertFalse(FileArchive.isValidProposal(100000))
        self.assertFalse(FileArchive.isValidProposal(3.14159265))
        self.assertFalse(FileArchive.isValidProposal('xxx'))
        self.assertFalse(FileArchive.isValidProposal(None))

    def testIsValidVisit(self):
        self.assertTrue(FileArchive.isValidVisit('01'))
        self.assertFalse(FileArchive.isValidVisit('xxx'))
        self.assertFalse(FileArchive.isValidVisit(01))
        self.assertFalse(FileArchive.isValidVisit(None))

    def testInstrumentFilepath(self):
        tempdir = tempfile.mkdtemp()
        try:
            fa = FileArchive(tempdir)
            self.assertEqual(os.path.join(tempdir, 'acs'),
                             fa.instrumentFilepath('acs'))
            self.assertEqual(os.path.join(tempdir, 'wfpc2'),
                             fa.instrumentFilepath('wfpc2'))
            with self.assertRaises(Exception):
                fa.instrumentFilepath('xxx')
        finally:
            shutil.rmtree(tempdir)

    def testProposalFilepath(self):
        tempdir = tempfile.mkdtemp()
        try:
            fa = FileArchive(tempdir)
            self.assertEqual(os.path.join(tempdir, 'acs/hst_12345'),
                             fa.proposalFilepath('acs', 12345))
            self.assertEqual(os.path.join(tempdir, 'wfpc2/hst_00000'),
                             fa.proposalFilepath('wfpc2', 0))
            with self.assertRaises(Exception):
                fa.proposalFilepath('acs', -1)
        finally:
            shutil.rmtree(tempdir)
        pass

    def testVisitFilepath(self):
        tempdir = tempfile.mkdtemp()
        try:
            fa = FileArchive(tempdir)
            self.assertEqual(os.path.join(tempdir, 'acs/hst_12345/visit_xx'),
                             fa.visitFilepath('acs', 12345, 'xx'))
            self.assertEqual(os.path.join(tempdir, 'wfpc2/hst_00000/visit_01'),
                             fa.visitFilepath('wfpc2', 0, '01'))
            with self.assertRaises(Exception):
                fa.visitFilepath('acs', 1, 'xxx')
        finally:
            shutil.rmtree(tempdir)
        pass

if __name__ == '__main__':
    unittest.main()
