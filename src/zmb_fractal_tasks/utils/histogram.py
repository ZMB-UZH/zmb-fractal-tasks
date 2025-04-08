"""Histogram class for handling histogram data."""

from collections.abc import Sequence

import anndata as ad
import dask.array as da
import numpy as np


class Histogram:
    """Histogram class that can be updated with new data.

    The bins are defined by a bin width and a zero offset:
    bin_n = [zero_offset + n*bin_width, zero_offset + (n+1)*bin_width)

    self.frequencies is a 1D array of the number of data points in each bin.
    self.first_bin_no is the bin number (n) of the first bin in self.frequencies.
    """

    def __init__(
        self,
        data: np.ndarray = None,
        bin_width: float = 1,
        zero_offset: float = 0,
    ):
        """Initialize the histogram with data, bin width, and zero offset."""
        self.bin_width = bin_width
        self.zero_offset = zero_offset
        if data is not None:
            self.first_bin_no, self.frequencies = self._get_hist(data)
        else:
            self.first_bin_no = None
            self.frequencies = None

    def _get_hist(self, data: np.ndarray) -> tuple[int, np.ndarray]:
        """Compute histogram of data."""
        # check if data is empty
        if np.array(data).size == 0:
            return None, None

        first_bin_no = int((data.min() - self.zero_offset) // self.bin_width)
        last_bin_no = int((data.max() - self.zero_offset) // self.bin_width)
        bin_edges = np.arange(
            first_bin_no * self.bin_width + self.zero_offset,
            (last_bin_no + 2) * self.bin_width + self.zero_offset,
            self.bin_width,
        )
        freq = np.histogram(data, bin_edges)[0]

        return first_bin_no, freq

    @staticmethod
    def _combine_frequencies(
        frequencies1: np.ndarray,
        first_bin_no1: int,
        frequencies2: np.ndarray,
        first_bin_no2: int,
    ) -> tuple[int, np.ndarray]:
        """Combine two frequency arrays into one."""
        first_bin_no = min([first_bin_no1, first_bin_no2])
        last_bin_no = max(
            [
                first_bin_no1 + len(frequencies1) - 1,
                first_bin_no2 + len(frequencies2) - 1,
            ]
        )
        # TODO: check for overflow?
        if isinstance(frequencies1, da.core.Array) or isinstance(
            frequencies2, da.core.Array
        ):
            combined_frequencies = da.zeros(last_bin_no - first_bin_no + 1, dtype=int)
        else:
            combined_frequencies = np.zeros(last_bin_no - first_bin_no + 1, dtype=int)
        combined_frequencies[
            first_bin_no1 - first_bin_no : first_bin_no1
            - first_bin_no
            + len(frequencies1)
        ] += frequencies1
        combined_frequencies[
            first_bin_no2 - first_bin_no : first_bin_no2
            - first_bin_no
            + len(frequencies2)
        ] += frequencies2
        return first_bin_no, combined_frequencies

    @property
    def last_bin_no(self) -> int:
        """Get the last bin number."""
        if self.first_bin_no is None or self.frequencies is None:
            return None
        return self.first_bin_no + len(self.frequencies) - 1

    def get_bin_edges(self) -> np.ndarray:
        """Get bin edges for the histogram."""
        if self.first_bin_no is None or self.frequencies is None:
            return None
        bin_edges = np.arange(
            self.first_bin_no * self.bin_width + self.zero_offset,
            (self.last_bin_no + 2) * self.bin_width + self.zero_offset,
            self.bin_width,
        )
        return bin_edges

    def add_histogram(self, new_histogram: "Histogram"):
        """Add histogram data to the current histogram."""
        if self.bin_width != new_histogram.bin_width:
            raise ValueError("Cannot add histogram data with different bin widths")
        if self.zero_offset != new_histogram.zero_offset:
            raise ValueError("Cannot add histogram data with different zero offsets")
        if self.frequencies is None:
            # If the histogram is empty, initialize it with the new histogram data
            self.first_bin_no = new_histogram.first_bin_no
            self.frequencies = new_histogram.frequencies
        elif new_histogram.frequencies is None:
            # If the new histogram is empty, do nothing
            return
        else:
            # Update the histogram by merging the two histograms
            self.first_bin_no, self.frequencies = self._combine_frequencies(
                self.frequencies,
                self.first_bin_no,
                new_histogram.frequencies,
                new_histogram.first_bin_no,
            )

    def pad_histogram(self, new_first_bin_no: int, new_last_bin_no: int) -> None:
        """Pad the histogram to the new first and last bin numbers."""
        if self.first_bin_no is None or self.frequencies is None:
            self.first_bin_no = new_first_bin_no
            self.frequencies = np.zeros(
                new_last_bin_no - new_first_bin_no + 1, dtype=int
            )
        else:
            if new_first_bin_no > self.first_bin_no:
                raise ValueError("Cannot pad histogram to a higher first bin number")
            if new_last_bin_no < self.last_bin_no:
                raise ValueError("Cannot pad histogram to a lower last bin number")
            if isinstance(self.frequencies, da.core.Array):
                new_frequencies = da.zeros(
                    new_last_bin_no - new_first_bin_no + 1, dtype=int
                )
            else:
                new_frequencies = np.zeros(
                    new_last_bin_no - new_first_bin_no + 1, dtype=int
                )
            new_frequencies[
                self.first_bin_no - new_first_bin_no : self.first_bin_no
                - new_first_bin_no
                + len(self.frequencies)
            ] += self.frequencies
            self.first_bin_no = new_first_bin_no
            self.frequencies = new_frequencies

    def trim_histogram(self) -> None:
        """Trim the histogram to remove empty bins at the beginning and end."""
        if self.frequencies is None:
            return
        nonzero_indices = np.nonzero(self.frequencies)[0]
        if nonzero_indices.size == 0:
            self.first_bin_no = None
            self.frequencies = None
        else:
            start = nonzero_indices[0]
            end = nonzero_indices[-1] + 1
            self.frequencies = self.frequencies[start:end]
            self.first_bin_no += start

    def get_quantiles(self, quantiles: Sequence[float]) -> Sequence[float]:
        """Get the quantiles of the histogram."""
        if self.frequencies is None:
            raise ValueError("Histogram is empty")
        if not all(0 <= q <= 1 for q in quantiles):
            raise ValueError("Quantiles must be between 0 and 1")
        bin_edges = self.get_bin_edges()
        if isinstance(self.frequencies, da.core.Array):
            cumulative_frequencies = np.cumsum(self.frequencies.compute())
        else:
            cumulative_frequencies = np.cumsum(self.frequencies)
        total_count = cumulative_frequencies[-1]
        quantile_indices = [
            np.searchsorted(cumulative_frequencies, q * total_count, side="left")
            for q in quantiles
        ]
        quantile_values = bin_edges[quantile_indices]
        return quantile_values

    def copy(self) -> "Histogram":
        """Create a copy of the Histogram object."""
        new_histogram = Histogram(
            bin_width=self.bin_width,
            zero_offset=self.zero_offset,
        )
        new_histogram.first_bin_no = self.first_bin_no
        new_histogram.frequencies = (
            None if self.frequencies is None else self.frequencies.copy()
        )
        return new_histogram


def align_histograms(
    histo_dict: dict[str, Histogram],
) -> Sequence[Histogram]:
    """Align a dictionary of histograms to the same first_bin_no and length."""
    if len({h.bin_width for h in histo_dict.values()}) > 1:
        raise ValueError("All histograms must have the same bin_width")
    if len({h.zero_offset for h in histo_dict.values()}) > 1:
        raise ValueError("All histograms must have the same zero_offset")

    first_bin_no = min([h.first_bin_no for h in histo_dict.values()])
    last_bin_no = max([h.last_bin_no for h in histo_dict.values()])

    new_histo_dict = {}
    for name, h in histo_dict.items():
        h_new = h.copy()
        h_new.pad_histogram(first_bin_no, last_bin_no)
        new_histo_dict[name] = h_new

    return new_histo_dict


def histograms_to_anndata(histo_dict: dict[str, Histogram]) -> ad.AnnData:
    """Convert a dictionary of histograms to an AnnData object."""
    if len({h.bin_width for h in histo_dict.values()}) > 1:
        raise ValueError("All histograms must have the same bin_width")
    if len({h.zero_offset for h in histo_dict.values()}) > 1:
        raise ValueError("All histograms must have the same zero_offset")

    aligned_histo_dict = align_histograms(histo_dict)

    data = np.array(
        [h.frequencies for h in aligned_histo_dict.values()],
        dtype=int,
    )
    names = list(aligned_histo_dict.keys())
    bin_edges = aligned_histo_dict[names[0]].get_bin_edges()
    bin_width = aligned_histo_dict[names[0]].bin_width
    zero_offset = aligned_histo_dict[names[0]].zero_offset
    first_bin_no = aligned_histo_dict[names[0]].first_bin_no

    adata = ad.AnnData(data)
    adata.obs_names = names
    adata.var["bin_start"] = bin_edges[:-1]
    adata.uns["bin_width"] = bin_width
    adata.uns["zero_offset"] = zero_offset
    adata.uns["first_bin_no"] = first_bin_no

    return adata


def anndata_to_histograms(adata: ad.AnnData) -> dict[str, Histogram]:
    """Convert an AnnData object to a dictionary of histograms."""
    bin_width = adata.uns["bin_width"]
    zero_offset = adata.uns["zero_offset"]
    first_bin_no = adata.uns["first_bin_no"]

    histo_dict = {}
    for name in adata.obs_names:
        frequencies = adata[name].X[0]
        histo_dict[name] = Histogram(
            bin_width=bin_width,
            zero_offset=zero_offset,
        )
        histo_dict[name].frequencies = frequencies
        histo_dict[name].first_bin_no = first_bin_no
        histo_dict[name].trim_histogram()

    return histo_dict
