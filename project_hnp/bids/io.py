from __future__ import annotations

from typing import TYPE_CHECKING

from .eeg import write_eeg_datasets
from .meg import write_meg_datasets
from .mri import write_mri_datasets

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
    write_eeg_datasets(root, root_raw, subject, data_eeg)
    write_mri_datasets(root, root_raw, subject, data_mri)
    # MEG last to overwrite participant info
    write_meg_datasets(root, root_raw, subject, data_meg)
