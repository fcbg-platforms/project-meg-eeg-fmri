from __future__ import annotations

from typing import TYPE_CHECKING

from mne_bids import BIDSPath, write_anat

from ..utils._docs import fill_doc
from ._constants import EXPECTED_MRI, EXPECTED_fMRI_T1
from ._utils import validate_bids_paths

if TYPE_CHECKING:
    from pathlib import Path


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
    data_mri : Path
        Path to the MRI dataset.
    """
    validate_bids_paths(bids_path, bids_path_raw)
    folders = [folder.name for folder in data_mri.iterdir() if folder.is_dir()]
    if set(folders) != EXPECTED_MRI:
        raise ValueError(f"Expected MRI folders {EXPECTED_MRI}, got {set(folders)}.")
    # find anatomical MRI and write it to BIDS dataset
    assert len(EXPECTED_fMRI_T1) == 1  # sanity-check
    bids_path.update(datatype="anat")
    for file in (data_mri / "NIfTI").glob("*.nii"):
        if EXPECTED_fMRI_T1[0] in file.name.lower():
            write_anat(file, bids_path, overwrite=True)
            break
