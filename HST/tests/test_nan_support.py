import os
import tempfile
import numpy as np
import pytest
import astropy.io.fits as pyfits
from product_labels import nan_support

class TestNanSupport:
    def create_hdulist(self, data_list, dtype=np.float32):
        hdulist = []
        for data in data_list:
            hdu = pyfits.PrimaryHDU(data=np.array(data, dtype=dtype))
            hdulist.append(hdu)
        return hdulist

    def test_get_nan_info_no_data(self):
        hdulist = [pyfits.PrimaryHDU()]
        result = nan_support._get_nan_info(hdulist)
        assert result == []

    def test_get_nan_info_int_data(self):
        hdulist = self.create_hdulist([[1, 2, 3]], dtype=np.int32)
        result = nan_support._get_nan_info(hdulist)
        assert result == []

    def test_get_nan_info_with_nans(self):
        arr = np.array([1.0, np.nan, 2.0])
        hdulist = self.create_hdulist([arr])
        result = nan_support._get_nan_info(hdulist)
        assert len(result) == 1
        k, mask, antimask = result[0]
        assert k == 0
        assert mask[1]
        assert antimask[0]

    def test_select_nan_replacement_positive(self):
        arr = np.array([1.0, 2.0, np.nan])
        hdulist = self.create_hdulist([arr])
        nan_info = nan_support._get_nan_info(hdulist)
        rep = nan_support._select_nan_replacement(hdulist, nan_info)
        assert rep == 0.0

    def test_select_nan_replacement_negative(self):
        arr = np.array([-2.0, -1.0, np.nan])
        hdulist = self.create_hdulist([arr])
        nan_info = nan_support._get_nan_info(hdulist)
        rep = nan_support._select_nan_replacement(hdulist, nan_info)
        assert isinstance(rep, str)
        assert rep.startswith('-')
        assert 'e' in rep

    def test_has_nans_true_false(self):
        arr = np.array([1.0, np.nan, 2.0])
        hdulist = self.create_hdulist([arr])
        assert nan_support.has_nans(hdulist) is True
        arr2 = np.array([1.0, 2.0, 3.0])
        hdulist2 = self.create_hdulist([arr2])
        assert nan_support.has_nans(hdulist2) is False

    def test_rewrite_wo_nans_no_nans(self, tmp_path):
        arr = np.array([1.0, 2.0, 3.0])
        hdu = pyfits.PrimaryHDU(arr)
        file_path = tmp_path / "test.fits"
        hdu.writeto(file_path)
        rep, idxs = nan_support.rewrite_wo_nans(str(file_path))
        assert rep is None
        assert idxs == []
        # File should remain unchanged
        arr2 = pyfits.getdata(file_path)
        np.testing.assert_array_equal(arr, arr2)

    def test_rewrite_wo_nans_with_nans(self, tmp_path):
        arr = np.array([1.0, np.nan, 2.0])
        hdu = pyfits.PrimaryHDU(arr)
        file_path = tmp_path / "test_nan.fits"
        hdu.writeto(file_path)
        rep, idxs = nan_support.rewrite_wo_nans(str(file_path))
        assert rep is not None
        assert idxs == [0]
        arr2 = pyfits.getdata(file_path)
        assert not np.isnan(arr2).any()
        assert rep in arr2

    def test_cmp_ignoring_nans_identical(self, tmp_path):
        arr = np.array([1.0, 2.0, 3.0])
        hdu = pyfits.PrimaryHDU(arr)
        file1 = tmp_path / "a.fits"
        file2 = tmp_path / "b.fits"
        hdu.writeto(file1)
        hdu.writeto(file2)
        assert nan_support.cmp_ignoring_nans(str(file1), str(file2)) is True

    def test_cmp_ignoring_nans_nan_vs_replacement(self, tmp_path):
        arr_nan = np.array([1.0, np.nan, 2.0])
        arr_rep = np.array([1.0, -2.0, 2.0])
        hdu_nan = pyfits.PrimaryHDU(arr_nan)
        hdu_rep = pyfits.PrimaryHDU(arr_rep)
        file_nan = tmp_path / "nan.fits"
        file_rep = tmp_path / "rep.fits"
        hdu_nan.writeto(file_nan)
        hdu_rep.writeto(file_rep)
        # fudge file sizes to match for test (overwrite rep with nan's bytes)
        with open(file_nan, 'rb') as fsrc, open(file_rep, 'wb') as fdst:
            fdst.write(fsrc.read())
        # Now rep file has nan bytes, so cmp_ignoring_nans should be True
        assert nan_support.cmp_ignoring_nans(str(file_nan), str(file_rep)) is True

    def test_cmp_ignoring_nans_different(self, tmp_path):
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 4.0])
        hdu1 = pyfits.PrimaryHDU(arr1)
        hdu2 = pyfits.PrimaryHDU(arr2)
        file1 = tmp_path / "f1.fits"
        file2 = tmp_path / "f2.fits"
        hdu1.writeto(file1)
        hdu2.writeto(file2)
        assert nan_support.cmp_ignoring_nans(str(file1), str(file2)) is False

    def test_get_nan_info_none_data(self):
        class DummyHDU:
            data = None
        hdulist = [DummyHDU()]
        assert nan_support._get_nan_info(hdulist) == []

    def test_get_nan_info_non_float(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        hdulist = self.create_hdulist([arr], dtype=np.int32)
        assert nan_support._get_nan_info(hdulist) == []

    def test_get_nan_info_float_no_nans(self):
        arr = np.array([1.0, 2.0, 3.0])
        hdulist = self.create_hdulist([arr])
        assert nan_support._get_nan_info(hdulist) == []

    def test_select_nan_replacement_empty(self):
        arr = np.array([1.0, 2.0, 3.0])
        hdulist = self.create_hdulist([arr])
        assert nan_support._select_nan_replacement(hdulist, []) is None

    def test_select_nan_replacement_multiple_hdus(self):
        arr1 = np.array([1.0, np.nan, 2.0])
        arr2 = np.array([-5.0, np.nan, -2.0])
        hdulist = self.create_hdulist([arr1, arr2])
        nan_info = nan_support._get_nan_info(hdulist)
        rep = nan_support._select_nan_replacement(hdulist, nan_info)
        assert rep != 0.0

    def test_rewrite_wo_nans_no_rewrite(self, tmp_path):
        arr = np.array([1.0, np.nan, 2.0])
        hdu = pyfits.PrimaryHDU(arr)
        file_path = tmp_path / "test_no_rewrite.fits"
        hdu.writeto(file_path)
        rep, idxs = nan_support.rewrite_wo_nans(str(file_path), rewrite=False)
        # File should still have NaNs
        arr2 = pyfits.getdata(file_path)
        assert np.isnan(arr2).any()
        assert rep is not None
        assert idxs == [0]

    def test_rewrite_wo_nans_multi_hdu(self, tmp_path):
        arr1 = np.array([1.0, np.nan, 2.0])
        arr2 = np.array([3.0, 4.0, 5.0])
        hdu1 = pyfits.PrimaryHDU(arr1)
        hdu2 = pyfits.ImageHDU(arr2)
        file_path = tmp_path / "multi_hdu.fits"
        hdulist = pyfits.HDUList([hdu1, hdu2])
        hdulist.writeto(file_path)
        rep, idxs = nan_support.rewrite_wo_nans(str(file_path))
        assert rep is not None
        assert 0 in idxs
        assert 1 not in idxs

    def test_cmp_ignoring_nans_diff_size(self, tmp_path):
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0])
        hdu1 = pyfits.PrimaryHDU(arr1)
        hdu2 = pyfits.PrimaryHDU(arr2)
        file1 = tmp_path / "f1.fits"
        file2 = tmp_path / "f2.fits"
        hdu1.writeto(file1)
        hdu2.writeto(file2)
        assert nan_support.cmp_ignoring_nans(str(file1), str(file2)) is False

    def test_cmp_ignoring_nans_diff_num_hdus(self, tmp_path):
        arr = np.array([1.0, 2.0, 3.0])
        hdu1 = pyfits.PrimaryHDU(arr)
        hdu2 = pyfits.ImageHDU(arr)
        file1 = tmp_path / "f1.fits"
        file2 = tmp_path / "f2.fits"
        pyfits.HDUList([hdu1]).writeto(file1)
        pyfits.HDUList([hdu1, hdu2]).writeto(file2)
        assert nan_support.cmp_ignoring_nans(str(file1), str(file2)) is False

    def test_cmp_ignoring_nans_diff_shape_dtype(self, tmp_path):
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([[1.0, 2.0], [3.0, 4.0]])
        hdu1 = pyfits.PrimaryHDU(arr1)
        hdu2 = pyfits.PrimaryHDU(arr2)
        file1 = tmp_path / "f1.fits"
        file2 = tmp_path / "f2.fits"
        hdu1.writeto(file1)
        hdu2.writeto(file2)
        assert nan_support.cmp_ignoring_nans(str(file1), str(file2)) is False

    def test_cmp_ignoring_nans_non_nan_diff(self, tmp_path):
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 4.0])
        hdu1 = pyfits.PrimaryHDU(arr1)
        hdu2 = pyfits.PrimaryHDU(arr2)
        file1 = tmp_path / "f1.fits"
        file2 = tmp_path / "f2.fits"
        hdu1.writeto(file1)
        hdu2.writeto(file2)
        assert nan_support.cmp_ignoring_nans(str(file1), str(file2)) is False

    def test_cmp_ignoring_nans_nans_in_different_hdus(self, tmp_path):
        arr1 = np.array([1.0, np.nan, 3.0])
        arr2 = np.array([1.0, 2.0, 3.0])
        arr3 = np.array([4.0, 5.0, 6.0])
        hdu1 = pyfits.PrimaryHDU(arr1)
        hdu2 = pyfits.ImageHDU(arr3)
        hdu3 = pyfits.PrimaryHDU(arr2)
        file1 = tmp_path / "f1.fits"
        file2 = tmp_path / "f2.fits"
        pyfits.HDUList([hdu1, hdu2]).writeto(file1)
        pyfits.HDUList([hdu3, hdu2]).writeto(file2)
        assert nan_support.cmp_ignoring_nans(str(file1), str(file2)) is True
