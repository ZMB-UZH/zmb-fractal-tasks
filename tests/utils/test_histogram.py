import dask.array as da
import numpy as np
import pytest

from zmb_fractal_tasks.utils.histogram import (
    Histogram,
    align_histograms,
    anndata_to_histograms,
    histograms_to_anndata,
)


def test_histogram_initialization_with_data():
    data = np.array([1, 2, 3, 4, 5])
    bin_width = 1.0
    zero_offset = 0.0
    hist = Histogram(data, bin_width, zero_offset)

    assert hist.bin_width == bin_width
    assert hist.zero_offset == zero_offset
    assert hist.first_bin_no == 1
    assert hist.last_bin_no == 5
    assert np.array_equal(hist.get_bin_edges(), [1, 2, 3, 4, 5, 6])
    assert np.array_equal(hist.frequencies, [1, 1, 1, 1, 1])


def test_histogram_initialization_with_nonzero_offset():
    data = np.array([1, 2, 3, 4, 5])
    bin_width = 2.0
    zero_offset = 1.5
    hist = Histogram(data, bin_width, zero_offset)

    assert hist.bin_width == bin_width
    assert hist.zero_offset == zero_offset
    assert hist.first_bin_no == -1
    assert hist.last_bin_no == 1
    assert np.array_equal(hist.get_bin_edges(), [-0.5, 1.5, 3.5, 5.5])
    assert np.array_equal(hist.frequencies, [1, 2, 2])


def test_histogram_initialization_without_data():
    hist = Histogram()

    assert hist.bin_width == 1
    assert hist.zero_offset == 0
    assert hist.first_bin_no is None
    assert hist.last_bin_no is None
    assert hist.get_bin_edges() is None
    assert hist.frequencies is None


def test_add_histogram():
    data1 = np.array([1, 2, 3])
    data2 = np.array([3, 4, 5])
    hist1 = Histogram(data1, bin_width=1, zero_offset=0)
    hist2 = Histogram(data2, bin_width=1, zero_offset=0)

    hist1.add_histogram(hist2)

    assert hist1.first_bin_no == 1
    assert np.array_equal(hist1.frequencies, [1, 1, 2, 1, 1])


def test_add_histogram_with_empty_histogram():
    data = np.array([1, 2, 3])
    hist1 = Histogram(data, bin_width=1, zero_offset=0)
    hist2 = Histogram()

    hist1.add_histogram(hist2)

    assert hist1.first_bin_no == 1
    assert np.array_equal(hist1.frequencies, [1, 1, 1])


def test_add_histogram_to_empty_histogram():
    data = np.array([1, 2, 3])
    hist1 = Histogram()
    hist2 = Histogram(data, bin_width=1, zero_offset=0)

    hist1.add_histogram(hist2)

    assert hist1.first_bin_no == 1
    assert np.array_equal(hist1.frequencies, [1, 1, 1])


def test_add_two_empty_histograms():
    hist1 = Histogram()
    hist2 = Histogram()

    hist1.add_histogram(hist2)

    assert hist1.first_bin_no is None
    assert hist1.frequencies is None


def test_add_histogram_with_different_bin_widths():
    data1 = np.array([1, 2, 3])
    data2 = np.array([3, 4, 5])
    hist1 = Histogram(data1, bin_width=1, zero_offset=0)
    hist2 = Histogram(data2, bin_width=2, zero_offset=0)

    with pytest.raises(
        ValueError, match="Cannot add histogram data with different bin widths"
    ):
        hist1.add_histogram(hist2)


def test_add_histogram_with_different_zero_offsets():
    data1 = np.array([1, 2, 3])
    data2 = np.array([3, 4, 5])
    hist1 = Histogram(data1, bin_width=1, zero_offset=0)
    hist2 = Histogram(data2, bin_width=1, zero_offset=1)

    with pytest.raises(
        ValueError, match="Cannot add histogram data with different zero offsets"
    ):
        hist1.add_histogram(hist2)


def test_add_histogram_with_different_bin_widths_and_different_zero_offsets():
    data1 = np.array([1, 2, 3])
    data2 = np.array([3, 4, 5])
    hist1 = Histogram(data1, bin_width=1, zero_offset=0)
    hist2 = Histogram(data2, bin_width=2, zero_offset=1)

    with pytest.raises(
        ValueError, match="Cannot add histogram data with different bin widths"
    ):
        hist1.add_histogram(hist2)


def test_initialize_histogram_with_empty_data():
    data = np.array([])
    bin_width = 1.0
    zero_offset = 0.0
    hist = Histogram(data, bin_width, zero_offset)

    assert hist.bin_width == bin_width
    assert hist.zero_offset == zero_offset
    assert hist.first_bin_no is None
    assert hist.last_bin_no is None
    assert hist.get_bin_edges() is None
    assert hist.frequencies is None


def test_histogram_with_dask_data():
    data1 = da.from_array(np.array([1, 2, 3, 4, 5]), chunks=(2,))
    data2 = np.array([4, 5, 6, 7])
    bin_width = 1.0
    zero_offset = 0.0
    hist1 = Histogram(data1, bin_width, zero_offset)
    hist2 = Histogram(data2, bin_width, zero_offset)

    assert hist1.bin_width == bin_width
    assert hist1.zero_offset == zero_offset
    assert hist1.first_bin_no == 1
    assert hist1.last_bin_no == 5
    assert np.array_equal(hist1.get_bin_edges(), [1, 2, 3, 4, 5, 6])
    assert np.array_equal(hist1.frequencies.compute(), [1, 1, 1, 1, 1])

    hist1.add_histogram(hist2)
    assert hist1.first_bin_no == 1
    assert np.array_equal(hist1.frequencies.compute(), [1, 1, 1, 2, 2, 1, 1])


def test_pad_histogram():
    data = np.array([2, 3, 4])
    bin_width = 1.0
    zero_offset = 0.0
    hist = Histogram(data, bin_width, zero_offset)

    hist.pad_histogram(1, 6)

    assert hist.first_bin_no == 1
    assert hist.last_bin_no == 6
    assert np.array_equal(hist.frequencies, [0, 1, 1, 1, 0, 0])


def test_trim_histogram_no_trimming_needed():
    """Test trimming when no bins are empty at the edges."""
    data = np.array([1, 2, 3, 4, 5])
    hist = Histogram(data, bin_width=1, zero_offset=0)
    original_first_bin_no = hist.first_bin_no
    original_frequencies = hist.frequencies.copy()

    hist.trim_histogram()

    assert hist.first_bin_no == original_first_bin_no
    assert np.array_equal(hist.frequencies, original_frequencies)


def test_trim_histogram_with_empty_bins():
    """Test trimming when there are empty bins at the edges."""
    hist = Histogram(bin_width=1, zero_offset=0)
    hist.first_bin_no = 0
    hist.frequencies = np.array([0, 0, 3, 0, 4, 0, 0])

    hist.trim_histogram()

    assert hist.first_bin_no == 2
    assert np.array_equal(hist.frequencies, [3, 0, 4])


def test_trim_histogram_completely_empty():
    """Test trimming when the histogram is completely empty."""
    hist = Histogram(bin_width=1, zero_offset=0)
    hist.first_bin_no = 0
    hist.frequencies = np.array([0, 0, 0, 0])

    hist.trim_histogram()

    assert hist.first_bin_no is None
    assert hist.frequencies is None


def test_trim_histogram_with_no_frequencies():
    """Test trimming when the histogram has no frequencies (None)."""
    hist = Histogram(bin_width=1, zero_offset=0)

    hist.trim_histogram()

    assert hist.first_bin_no is None
    assert hist.frequencies is None


def test_get_quantiles():
    """Test the get_quantiles method of the Histogram class."""
    data = np.array([0, 2, 2, 3, 3, 3, 4, 4, 4, 5])
    hist = Histogram(data, bin_width=1, zero_offset=0)

    # Test valid quantiles
    quantiles = [0.0, 0.1, 0.2, 0.25, 0.3, 1.0]
    expected_quantiles = [0, 0, 2, 2, 2, 5]
    result = hist.get_quantiles(quantiles)
    assert np.array_equal(result, expected_quantiles)

    # Test invalid quantiles (outside [0, 1])
    with pytest.raises(ValueError, match="Quantiles must be between 0 and 1"):
        hist.get_quantiles([-0.1, 1.1])

    # Test empty histogram
    empty_hist = Histogram()
    with pytest.raises(ValueError, match="Histogram is empty"):
        empty_hist.get_quantiles([0.5])


def test_get_quantiles_dask():
    """Test the get_quantiles method of the Histogram class."""
    data = da.array([0, 2, 2, 3, 3, 3, 4, 4, 4, 5])
    hist = Histogram(data, bin_width=1, zero_offset=0)

    # Test valid quantiles
    quantiles = [0.0, 0.1, 0.2, 0.25, 0.3, 1.0]
    expected_quantiles = [0, 0, 2, 2, 2, 5]
    result = hist.get_quantiles(quantiles)
    assert np.array_equal(result, expected_quantiles)

    # Test invalid quantiles (outside [0, 1])
    with pytest.raises(ValueError, match="Quantiles must be between 0 and 1"):
        hist.get_quantiles([-0.1, 1.1])

    # Test empty histogram
    empty_hist = Histogram()
    with pytest.raises(ValueError, match="Histogram is empty"):
        empty_hist.get_quantiles([0.5])


def test_copy_histogram():
    data = np.array([1, 2, 3])
    bin_width = 1.0
    zero_offset = 0.0
    hist = Histogram(data, bin_width, zero_offset)

    hist_copy = hist.copy()

    assert hist_copy is not hist
    assert hist_copy.first_bin_no == hist.first_bin_no
    assert np.array_equal(hist_copy.frequencies, hist.frequencies)


def test_align_histograms():
    """Test aligning multiple histograms to the same bin range."""
    hist1 = Histogram(data=np.array([1, 2, 3]), bin_width=1, zero_offset=0)
    hist2 = Histogram(data=np.array([2, 3, 4]), bin_width=1, zero_offset=0)
    hist3 = Histogram(data=np.array([3, 4, 5]), bin_width=1, zero_offset=0)

    histograms = {"hist1": hist1, "hist2": hist2, "hist3": hist3}
    aligned_histograms = align_histograms(histograms)

    # Check that all histograms have the same first_bin_no and length
    first_bin_no = aligned_histograms["hist1"].first_bin_no

    assert np.array_equal(aligned_histograms["hist1"].frequencies, [1, 1, 1, 0, 0])
    assert np.array_equal(aligned_histograms["hist2"].frequencies, [0, 1, 1, 1, 0])
    assert np.array_equal(aligned_histograms["hist3"].frequencies, [0, 0, 1, 1, 1])

    for hist in aligned_histograms.values():
        assert hist.first_bin_no == first_bin_no


def test_histograms_to_anndata():
    """Test converting histograms to AnnData."""
    hist1 = Histogram(data=np.array([1, 2, 3]), bin_width=1, zero_offset=0)
    hist2 = Histogram(data=np.array([2, 3, 4]), bin_width=1, zero_offset=0)

    histograms = {"channel1": hist1, "channel2": hist2}
    adata = histograms_to_anndata(histograms)

    # Check AnnData structure
    assert adata.obs_names.tolist() == ["channel1", "channel2"]
    assert adata.var["bin_start"].tolist() == [1, 2, 3, 4]
    assert adata.uns["bin_width"] == hist1.bin_width
    assert adata.uns["zero_offset"] == hist1.zero_offset
    assert adata.uns["first_bin_no"] == 1

    # Check frequencies
    assert np.array_equal(adata.X[0], [1, 1, 1, 0])
    assert np.array_equal(adata.X[1], [0, 1, 1, 1])


def test_histograms_to_anndata_unequal():
    """Test converting histograms with unequal bin widths to AnnData."""
    hist1 = Histogram(data=np.array([1, 2, 3]), bin_width=1, zero_offset=0)
    hist2 = Histogram(data=np.array([2, 3, 4]), bin_width=2, zero_offset=0)

    histograms = {"channel1": hist1, "channel2": hist2}

    with pytest.raises(ValueError, match="All histograms must have the same bin_width"):
        histograms_to_anndata(histograms)

    hist1 = Histogram(data=np.array([1, 2, 3]), bin_width=1, zero_offset=0)
    hist2 = Histogram(data=np.array([2, 3, 4]), bin_width=1, zero_offset=1)

    histograms = {"channel1": hist1, "channel2": hist2}

    with pytest.raises(
        ValueError, match="All histograms must have the same zero_offset"
    ):
        histograms_to_anndata(histograms)


def test_anndata_to_histograms():
    """Test converting AnnData back to histograms."""
    hist1 = Histogram(data=np.array([1, 2, 3]), bin_width=1, zero_offset=0)
    hist2 = Histogram(data=np.array([2, 3, 4]), bin_width=1, zero_offset=0)

    histograms = {"channel1": hist1, "channel2": hist2}
    adata = histograms_to_anndata(histograms)
    histograms_converted = anndata_to_histograms(adata)
    # Check that the converted histograms have the same properties
    assert histograms_converted["channel1"].bin_width == hist1.bin_width
    assert histograms_converted["channel1"].zero_offset == hist1.zero_offset
    assert histograms_converted["channel1"].first_bin_no == hist1.first_bin_no
    assert np.array_equal(
        histograms_converted["channel1"].frequencies, hist1.frequencies
    )
    assert histograms_converted["channel2"].bin_width == hist2.bin_width
    assert histograms_converted["channel2"].zero_offset == hist2.zero_offset
    assert histograms_converted["channel2"].first_bin_no == hist2.first_bin_no
    assert np.array_equal(
        histograms_converted["channel2"].frequencies, hist2.frequencies
    )
