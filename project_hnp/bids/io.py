from __future__ import annotations

from typing import TYPE_CHECKING

from mne_bids import BIDSPath

from ..utils._checks import ensure_int, ensure_path
from ._eeg import write_eeg_datasets
from ._meg import write_meg_datasets
from ._mri import write_mri_datasets

if TYPE_CHECKING:
    from pathlib import Path


def bidsification(
    root: Path | str,
    root_raw: Path | str,
    subject: int,
    data_eeg: Path | str,
    data_meg: Path | str,
    data_mri: Path | str,
):
    """Convert a dataset to BIDS.

    Parameters
    ----------
    root : Path | str
        Path to the root of the BIDS dataset.
    root_raw : Path | str
        Path to the root of the BIDS dataset containing raw/unconverted files.
    subject : int
        Subject number.
    data_eeg : Path | str
        Path to the EEG dataset.
    data_meg : Path | str
        Path to the MEG dataset.
    data_mri : Path | str
        Path to the MRI dataset.
    """
    root = ensure_path(root, must_exist=True)
    root_raw = ensure_path(root_raw, must_exist=True)
    subject = ensure_int(subject, "subject")
    if subject <= 0:
        raise ValueError(
            f"Argument 'subject' must be a positive integer, got {subject}."
        )
    data_eeg = ensure_path(data_eeg, must_exist=True)
    data_meg = ensure_path(data_meg, must_exist=True)
    data_mri = ensure_path(data_mri, must_exist=True)
    bids_path = BIDSPath(root=root, subject=str(subject).zfill(2))
    bids_path_raw = BIDSPath(root=root_raw, subject=str(subject).zfill(2))
    write_eeg_datasets(bids_path, bids_path_raw, data_eeg)
    write_mri_datasets(bids_path, bids_path_raw, data_mri)
    # MEG last to overwrite participant info
    write_meg_datasets(bids_path, bids_path_raw, data_meg)
