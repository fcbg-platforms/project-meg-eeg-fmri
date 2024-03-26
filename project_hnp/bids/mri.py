from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

from mne_bids import write_anat

from ..utils._docs import fill_doc
from ._constants import EXPECTED_fMRI_T1
from ._utils import validate_bids_paths, validate_data_MRI

if TYPE_CHECKING:
    from pathlib import Path

    from mne_bids import BIDSPath


@fill_doc
def write_mri_datasets(
    bids_path: BIDSPath,
    bids_path_raw: BIDSPath,
    data_mri: Path,
) -> None:
    """Write MRI datasets.

    Parameters
    ----------
    %(bids_path_root_sub)s
    %(bids_path_root_raw_sub)s
    %(data_mri)s
    """
    validate_bids_paths(bids_path, bids_path_raw)
    validate_data_MRI(data_mri)
    # find anatomical MRI and write it to BIDS dataset
    assert len(EXPECTED_fMRI_T1) == 1  # sanity-check
    bids_path.update(datatype="anat", task=None)
    bids_path_raw.update(datatype="anat", task=None, suffix="T1w")
    os.makedirs(bids_path_raw.fpath.parent, exist_ok=True)
    for file in (data_mri / "NIfTI").glob("*.nii"):
        if EXPECTED_fMRI_T1[0] in file.name.lower():
            write_anat(file, bids_path, overwrite=True)
            shutil.copy2(file, bids_path_raw.fpath.with_suffix("".join(file.suffixes)))
            break
