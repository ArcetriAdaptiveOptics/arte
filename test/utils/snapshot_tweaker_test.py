#!/usr/bin/env python
import tempfile
import unittest
import os
from astropy.io import fits as pyfits
import numpy as np
from arte.utils.snapshot_tweaker import SnapshotTweaker
from arte.utils.snapshotable import Snapshotable


class SnapshotTweakerTest(unittest.TestCase):

    def setUp(self):
        self.filename = self._tmp_file_name()
        self._removeFile()
        dicto = {}
        dicto['PIPPA'] = 'pippa'
        dicto['PIPPI'] = 'pippi'
        dicto['PIPPO'] = 'pippo'
        dicto['PLUTO'] = 'pluto'
        dicto['SEI.UNA.PIPPA'] = 'sei una pippa'
        dicto['VERY.LONG.KEYWORD'] = 3.14
        header = Snapshotable.as_fits_header(dicto)
        pyfits.writeto(self.filename, np.zeros(2), header)

    def _tmp_file_name(self):
        temp = tempfile.NamedTemporaryFile(prefix='tweak_', suffix='.fits')
        return temp.name

    def _removeFile(self):
        if os.path.isfile(self.filename):
            os.remove(self.filename)

    def tearDown(self):
        self._removeFile()

    def testDeleteKey(self):
        st = SnapshotTweaker(self.filename)
        st.header_delete_key('pippo')
        hdr = pyfits.getheader(self.filename)
        self.assertTrue(hdr.get('pippo') is None)

    def testRenameSingleKey(self):
        st = SnapshotTweaker(self.filename)
        st.header_rename_key('PIPPO', 'TOPOLINO')
        hdr = pyfits.getheader(self.filename)
        self.assertTrue(hdr['TOPOLINO'] == 'pippo')

    def testRenameMultipleKey(self):
        st = SnapshotTweaker(self.filename)
        st.header_rename_key('PIPP', 'SUPERPIPP')
        hdr = pyfits.getheader(self.filename)
        self.assertTrue(hdr['SUPERPIPPO'] == 'pippo')
        self.assertTrue(hdr['SUPERPIPPA'] == 'pippa')
        self.assertTrue(hdr['SUPERPIPPI'] == 'pippi')
        self.assertTrue(hdr['SEI.UNA.SUPERPIPPA'] == 'sei una pippa')

    def testRenameMultipleKeyShouldSkipIfKeyExists(self):
        st = SnapshotTweaker(self.filename)
        st.header_rename_key('PIPP', 'PLUT')
        hdr = pyfits.getheader(self.filename)
        self.assertTrue(hdr['PLUTO'] == 'pluto')


if __name__ == "__main__":
    unittest.main()
