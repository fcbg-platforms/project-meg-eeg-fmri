from __future__ import annotations

from typing import TYPE_CHECKING

from mne_bids.read import _from_tsv
from mne_bids.write import _write_tsv

from ..utils._checks import ensure_int
from ..utils._docs import fill_doc
from ._constants import EXPECTED_EEG, EXPECTED_MEG, EXPECTED_MRI, OPTIONAL_MEG

if TYPE_CHECKING:
    from pathlib import Path

    from mne_bids import BIDSPath


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


def fetch_participant_information(bids_path: BIDSPath) -> dict[str, str] | None:
    """Fetch participant information from the BIDS dataset.

    Parameters
    ----------
    bids_path : BIDSPath
        The BIDS path to the dataset, including root and the subject number.

    Returns
    -------
    dict | None
        The participant information if available, else None.
    """
    participants = _from_tsv(bids_path.root / "participants.tsv")
    if f"sub-{bids_path.subject}" in participants["participant_id"]:
        idx = participants["participant_id"].index(f"sub-{bids_path.subject}")
        return {key: elt[idx] for key, elt in participants.items()}
    return None


def write_participant_information(
    bids_path: BIDSPath, participant_info: dict[str, str] | None
) -> None:
    """Write participant information to the BIDS dataset.

    Parameters
    ----------
    bids_path : BIDSPath
        The BIDS path to the dataset, including root and the subject number.
    participant_info : dict | None
        Dictionary containing the participant information.
    """
    if participant_info is None:
        return  # nothing to do
    fname = bids_path.root / "participants.tsv"
    orig_data = _from_tsv(fname)
    assert f"sub-{bids_path.subject}" in orig_data["participant_id"]  # sanity-check
    idx = orig_data["participant_id"].index(f"sub-{bids_path.subject}")
    for key in orig_data:
        if key == "participant_id":
            continue
        orig_data[key][idx] = participant_info[key]
    _write_tsv(fname, orig_data, True)
