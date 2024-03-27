from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils._checks import ensure_int
from ..utils._docs import fill_doc
from ._constants import EXPECTED_EEG, EXPECTED_MEG, EXPECTED_MRI, OPTIONAL_MEG

if TYPE_CHECKING:
    from pathlib import Path


@fill_doc
def ensure_subject_int(subject: int) -> int:
    """Ensure that the subject number is a positive integer.

    Parameters
    ----------
    %(bids_subject)s

    Returns
    -------
    %(bids_subject)s
    """
    subject = ensure_int(subject, "subject")
    if subject <= 0:
        raise ValueError(
            f"Argument 'subject' must be a positive integer, got {subject}."
        )
    return subject


@fill_doc
def validate_data_MEG(data_meg: Path, subject: int) -> None:
    """Validate a folder containing MEG data.

    Parameters
    ----------
    data_meg : Path
        Path to the MEG dataset.
    %(bids_subject)s
    """
    for file in data_meg.glob("*.fif"):
        finfo = file.stem.split("_")
        assert finfo[0] == "sub"  # sanity-check
        if subject != int(finfo[1]):
            raise ValueError(
                f"The subject number in the filename ({int(finfo[1])}) does not match "
                f"the subject number requested in the BIDS path ({subject})."
            )
        assert finfo[2] == "task"  # sanity-check
        if finfo[3].lower() not in EXPECTED_MEG.union(OPTIONAL_MEG):
            raise ValueError(
                f"Unexpected task name '{finfo[3]}' in filename '{file.name}'."
            )


@fill_doc
def validate_data_EEG(data_eeg: Path, subject: int) -> None:
    """Validate a folder containing EEG data.

    Parameters
    ----------
    data_eeg : Path
        Path to the EEG dataset.
    %(bids_subject)s
    """
    for file in data_eeg.glob("*.mff"):
        finfo = file.stem.split("_")
        assert finfo[0].startswith("sub")  # sanity-check
        try:
            subj = int(finfo[0].split(sep="-")[1])
        except Exception:
            raise ValueError(
                f"The subject ID could not be parsed from the filename '{file.name}'."
            )
        if subject != subj:
            raise ValueError(
                f"The subject number in the filename ({subj}) does not match "
                f"the subject number requested in the BIDS path ({subject})."
            )
        assert finfo[1].startswith("task")  # sanity-check
        try:
            task = finfo[1].split("-")[1]
        except Exception:
            raise ValueError(
                f"The task name could not be parsed from the filename '{file.name}'."
            )
        if task not in EXPECTED_EEG:
            raise ValueError(
                f"Unexpected task name '{task}' in filename '{file.name}'."
            )


@fill_doc
def validate_data_MRI(data_mri: Path) -> None:
    """Validate a folder containing MRI data.

    Parameters
    ----------
    data_mri : Path
        Path to the MRI dataset.
    """
    folders = [folder.name for folder in data_mri.iterdir() if folder.is_dir()]
    if set(folders) != EXPECTED_MRI:
        raise ValueError(f"Expected MRI folders {EXPECTED_MRI}, got {set(folders)}.")
