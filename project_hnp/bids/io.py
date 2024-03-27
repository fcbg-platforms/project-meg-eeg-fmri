from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils._docs import fill_doc
from .eeg import write_eeg_datasets
from .meg import write_meg_datasets
from .mri import write_mri_datasets

if TYPE_CHECKING:
    from pathlib import Path


@fill_doc
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
    %(bids_root)s
    %(bids_root_raw)s
    %(bids_subject)s
    %(data_eeg)s
    %(data_meg)s
    %(data_mri)s
    """
    write_eeg_datasets(root, root_raw, subject, data_eeg)
    write_mri_datasets(root, root_raw, subject, data_mri)
    # MEG last to overwrite participant info
    write_meg_datasets(root, root_raw, subject, data_meg)
