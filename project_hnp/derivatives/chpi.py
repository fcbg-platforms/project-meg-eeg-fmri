from __future__ import annotations

from typing import TYPE_CHECKING

from mne.chpi import (
    compute_chpi_amplitudes,
    compute_chpi_locs,
    compute_head_pos,
    write_head_pos,
)
from mne_bids import BIDSPath, read_raw_bids

from ..bids._constants import EXPECTED_MEG, OPTIONAL_MEG
from ..utils._checks import ensure_path, ensure_subject_int
from ..utils._docs import fill_doc

if TYPE_CHECKING:
    from pathlib import Path


@fill_doc
def compute_heas_pos(root: Path | str, derivative: Path | str, subject: int) -> None:
    """Compute the head position for the given subject.

    The HPI coil locations digitized through the Polhemus are not adjusted based on the
    initial HPI measurement of each individual recording.

    Parameters
    ----------
    %(bids_root)s
    %(bids_derivative)s
    %(bids_subject)s
    """
    root = ensure_path(root, must_exist=True)
    derivative = ensure_path(derivative, must_exist=True)
    subject = ensure_subject_int(subject)
    # compute head position
    bids_path = BIDSPath(root=root, subject=str(subject).zfill(2), datatype="meg")
    bids_path_derivative = BIDSPath(
        root=derivative, subject=str(subject).zfill(2), datatype="meg"
    ).mkdir()
    for task in EXPECTED_MEG.union(OPTIONAL_MEG):
        raw = read_raw_bids(bids_path.update(task=task))
        chpi_amplitudes = compute_chpi_amplitudes(raw)
        chpi_locs = compute_chpi_locs(raw.info, chpi_amplitudes)
        head_pos = compute_head_pos(raw.info, chpi_locs)
        # save output to disk
        fname = (
            bids_path_derivative.directory
            / f"{bids_path_derivative.basename}_head_pos.pos"
        )
        write_head_pos(fname, head_pos)
