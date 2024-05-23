from __future__ import annotations

from typing import TYPE_CHECKING

from mne_bids import BIDSPath

from ..krios import read_EGI_ch_names, read_krios
from ..utils._checks import ensure_path, ensure_subject_int
from ..utils._docs import fill_doc

if TYPE_CHECKING:
    from pathlib import Path


@fill_doc
def export_krios_digitization(
    root_raw: Path | str, derivative: Path | str, subject: int
):
    """Export of the Krios digitization prior to scaling to match MNE's default.

    Parameters
    ----------
    %(bids_root_raw)s
    %(bids_derivative)s
    %(bids_subject)s
    """
    root_raw = ensure_path(root_raw, must_exist=True)
    derivative = ensure_path(derivative, must_exist=True)
    subject = ensure_subject_int(subject)
    bids_path_raw = BIDSPath(
        root=root_raw, subject=str(subject).zfill(2), datatype="eeg", suffix="eeg"
    )
    bids_path_derivative = BIDSPath(
        root=derivative, subject=str(subject).zfill(2), datatype="eeg"
    ).mkdir()
    fname = (bids_path_raw.directory / bids_path_raw.basename).with_suffix(".csv")
    elc, fid = read_krios(fname=fname)
    ch_names = read_EGI_ch_names()
    assert elc.shape[0] == len(ch_names)  # sanity-check
    fname = (
        bids_path_derivative.directory
        / f"{bids_path_derivative.basename}_elc_coords.xyz"
    )
    with open(fname, "w") as file:
        file.write(f"{len(ch_names) + 3}\t1\n")
        # write electrodes
        for coord, ch in zip(elc, ch_names, strict=True):
            file.write("\t".join([str(k) for k in coord]))
            file.write(f"\t{ch}\n")
        # write fiducials
        for coord, name in zip(fid, ["RPA", "LPA", "NZ"], strict=True):
            file.write("\t".join([str(k) for k in coord]))
            file.write(f"\t{name}\n")
