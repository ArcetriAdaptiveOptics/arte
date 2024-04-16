import glob
from astropy.io import fits as pyfits
import logging


class SnapshotTweaker():

    def __init__(self,
                 pathname,
                 logger=logging.getLogger('SnapshotTweaker')):
        self._pathname = pathname
        self._logger = logger
        self._matchingFiles = glob.glob(self._pathname)

    def header_update_key(self, key, value):
        for filename in self._matchingFiles:
            self._logger.info('update %s: %s= %s' %
                              (filename, key, str(value)))
            pyfits.setval(filename, key, value)

    def header_delete_key(self, key):
        for filename in self._matchingFiles:
            self._logger.info('delete %s: %s' % (filename, key))
            pyfits.delval(filename, key)

    def header_rename_key_2(self, key, newkey):
        for filename in self._matchingFiles:
            self._logger.info('rename %s: %s= %s' % (filename, key, newkey))
            try:
                value = pyfits.getval(filename, key)
                pyfits.setval(filename, newkey, value)
                pyfits.delval(filename, key)
            except Exception:
                pass

    def header_rename_key(self, oldKey, newKey):
        for filename in self._matchingFiles:
            self._logger.info('rename %s: %s= %s' %
                              (filename, oldKey, newKey))
            try:
                hdulist = pyfits.open(filename, 'update')
                hdr = hdulist[0].header
                original_hdr = hdr.copy()
                for k, v in original_hdr.items():
                    newK = k.replace(oldKey, newKey)
                    if newK in hdr:
                        self._logger.debug(
                            '%s already exists. Skipped' % newK)
                        continue
                    if newK != k:
                        self._logger.debug(
                            'Replacing %s with %s. '
                            'New entry is: %s = %s' % (
                                oldKey, newKey, newK, v))
                        del hdr[k]
                        hdr['HIERARCH ' + newK] = v
                hdulist.close(output_verify='fix')
            except Exception as e:
                #self._logger.warning('headerRename failed (%s)') % e
                self._logger.warning(e)
                pass

    def header_get_key(self, key):
        ret = {}
        for filename in self._matchingFiles:
            ret[filename] = pyfits.getval(filename, key)
        return ret
