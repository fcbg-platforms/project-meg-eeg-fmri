from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from mne_bids import BIDSPath, write_anat

from ..utils._checks import ensure_path, ensure_subject_int
from ..utils._docs import fill_doc
from ._constants import EXPECTED_fMRI_NIFTI, EXPECTED_fMRI_T1, MAPPING_fMRI
from ._utils import (
    fetch_participant_information,
    validate_data_MRI,
    write_participant_information,
)

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
    bids_path = BIDSPath(root=root, subject=str(subject).zfill(2))
    bids_path_raw = BIDSPath(root=root_raw, subject=str(subject).zfill(2)).mkdir()
    # copy DICOM raw folder
    shutil.copytree(
        data_mri / "DICOM", bids_path_raw.directory / "dicom", dirs_exist_ok=True
    )
    # look for existing participant information
    participant_info = fetch_participant_information(bids_path)
    # write anatomical T1 MRI and fMRI dataset in .nii format
    _write_anat(bids_path, bids_path_raw, data_mri)
    _write_functional(bids_path, bids_path_raw, data_mri)
    # add back participant information if needed
    write_participant_information(bids_path, participant_info)


def _write_anat(bids_path: BIDSPath, bids_path_raw: BIDSPath, data_mri: Path) -> None:
    """Write anatomical T1 image.

    Parameters
    ----------
    bids_path : BIDSPath
        BIDS Path to the BIDS dataset with both root and subject set.
    bids_path_raw : BIDSPath
        BIDS Path to the raw BIDS dataset with both root and subject set.
    data_mri : Path
        Path to the MRI dataset.
    """
    assert bids_path.root is not None
    assert bids_path.subject is not None
    assert bids_path_raw.root is not None
    assert bids_path_raw.subject is not None
    bids_path.update(datatype="anat", task=None)
    bids_path_raw.update(datatype="anat", suffix="T1w", task=None).mkdir()
    # find anatomical MRI and write it to BIDS dataset
    for file in (data_mri / "NIfTI").glob("*.nii"):
        if EXPECTED_fMRI_T1[0] in file.name.lower():
            write_anat(file, bids_path, overwrite=True)
            shutil.copy2(
                file,
                (bids_path_raw.directory / bids_path_raw.basename).with_suffix(
                    "".join(file.suffixes)
                ),
            )
            shutil.copy2(
                file.with_suffix(".json"),
                (bids_path_raw.directory / bids_path_raw.basename).with_suffix(".json"),
            )
            break


def _write_functional(
    bids_path: BIDSPath, bids_path_raw: BIDSPath, data_mri: Path
) -> None:
    """Write the fMRI dataset.

    Parameters
    ----------
    bids_path : BIDSPath
        BIDS Path to the BIDS dataset with both root and subject set.
    bids_path_raw : BIDSPath
        BIDS Path to the raw BIDS dataset with both root and subject set.
    data_mri : Path
        Path to the fMRI dataset.
    """
    assert bids_path.root is not None
    assert bids_path.subject is not None
    assert bids_path_raw.root is not None
    assert bids_path_raw.subject is not None
    bids_path.update(datatype="func", suffix="bold", task=None, check=False).mkdir()
    bids_path_raw.update(datatype="func", suffix="bold", task=None, check=False).mkdir()
    for file in (data_mri / "NIfTI").glob("*.nii"):
        if not file.name[0].isdigit():
            continue  # processed file
        if "task" not in file.stem:
            continue
        task = file.stem.split("task-")[1].split("_")[0]
        if task not in EXPECTED_fMRI_NIFTI:
            raise ValueError(
                f"fMRI file name {file.name} does not match the expected task name."
            )
        task = MAPPING_fMRI[task]
        bids_path.update(task=task)
        bids_path_raw.update(task=task)
        for path in (bids_path, bids_path_raw):
            path.update(suffix="bold")
            shutil.copy2(
                file,
                (path.directory / path.basename).with_suffix("".join(file.suffixes)),
            )
            shutil.copy2(
                file.with_suffix(".json"),
                (path.directory / path.basename).with_suffix(".json"),
            )
