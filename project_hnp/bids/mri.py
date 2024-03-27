from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

from mne_bids import BIDSPath, write_anat

from ..utils._checks import ensure_path
from ..utils._docs import fill_doc
from ._constants import EXPECTED_fMRI_T1
from ._utils import ensure_subject_int, validate_data_MRI

if TYPE_CHECKING:
    from pathlib import Path


@fill_doc
def write_mri_datasets(
    root: Path | str,
    root_raw: Path | str,
    subject: int,
    data_mri: Path | str,
) -> None:
    """Write MRI datasets.

    The MRI dataset should contain 2 folders: ``DICOM`` and ``NIfTI``.

    Parameters
    ----------
    %(bids_root)s
    %(bids_root_raw)s
    %(bids_subject)s
    %(data_mri)s
    """
    root = ensure_path(root, must_exist=True)
    root_raw = ensure_path(root_raw, must_exist=True)
    subject = ensure_subject_int(subject)
    data_mri = ensure_path(data_mri, must_exist=True)
    validate_data_MRI(data_mri)
    # create BIDS Path and folders
    bids_path = BIDSPath(
        root=root, subject=str(subject).zfill(2), datatype="anat", task=None
    )
    bids_path_raw = BIDSPath(
        root=root_raw,
        subject=str(subject).zfill(2),
        datatype="anat",
        suffix="T1w",
        task=None,
    )
    os.makedirs(bids_path_raw.fpath.parent, exist_ok=True)
    # find anatomical MRI and write it to BIDS dataset
    for file in (data_mri / "NIfTI").glob("*.nii"):
        if EXPECTED_fMRI_T1[0] in file.name.lower():
            write_anat(file, bids_path, overwrite=True)
            shutil.copy2(file, bids_path_raw.fpath.with_suffix("".join(file.suffixes)))
            break
