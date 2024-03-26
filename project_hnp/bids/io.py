from __future__ import annotations

import json
from importlib.resources import files
from typing import TYPE_CHECKING
from warnings import warn

from mne.io import read_raw_egi, read_raw_fif
from mne_bids import (
    BIDSPath,
    write_anat,
    write_meg_calibration,
    write_meg_crosstalk,
    write_raw_bids,
)
from mne_bids.utils import _write_json

from ..utils._checks import ensure_int, ensure_path
from ..utils._docs import fill_doc
from ._constants import (
    EXPECTED_EEG,
    EXPECTED_MEG,
    EXPECTED_MRI,
    OPTIONAL_MEG,
    EXPECTED_fMRI_T1,
)

if TYPE_CHECKING:
    from pathlib import Path


def bidsification(
    root: Path | str,
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
    subject = ensure_int(subject, "subject")
    if subject <= 0:
        raise ValueError(
            f"Argument 'subject' must be a positive integer, got {subject}."
        )
    data_eeg = ensure_path(data_eeg, must_exist=True)
    data_meg = ensure_path(data_meg, must_exist=True)
    data_mri = ensure_path(data_mri, must_exist=True)
    bids_path = BIDSPath(root=root, subject=str(subject).zfill(2))
    _write_eeg_datasets(bids_path, data_eeg)
    _write_mri_datasets(bids_path, data_mri)
    _write_meg_datasets(bids_path, data_meg)  # MEG last to overwrite participant info


@fill_doc
def _write_meg_datasets(
    bids_path: BIDSPath,
    data_meg: Path,
) -> None:
    """Write MEG datasets.

    Parameters
    ----------
    %(bids_path_root_sub)s
    data_meg : Path
        Path to the MEG dataset.
    """
    assert bids_path.root is not None
    assert bids_path.subject is not None
    empty_room = None
    for file in data_meg.glob("*.fif"):
        finfo = file.stem.split("_")
        assert finfo[0] == "sub"  # sanity-check
        if int(bids_path.subject) != int(finfo[1]):
            raise ValueError(
                f"The subject number in the filename ({int(finfo[1])}) does not match "
                "the subject number requested in the BIDS path "
                f"({int(bids_path.subject)})."
            )
        assert finfo[2] == "task"  # sanity-check
        if finfo[3].lower() not in EXPECTED_MEG.union(OPTIONAL_MEG):
            raise ValueError(
                f"Unexpected task name '{finfo[3]}' in filename '{file.name}'."
            )
        if finfo[3].lower() == "noise":
            empty_room = file
    if empty_room is None:
        warn(
            RuntimeWarning,
            f"The empty-room recording is missing in '{str(data_meg)}'.",
            stacklevel=2,
        )
    else:
        empty_room = read_raw_fif(empty_room)
    # now that the input is validated, we can update the BIDS dataset
    bids_path.update(datatype="meg")
    _write_meg_calibration_crosstalk(bids_path)
    for file in data_meg.glob("*.fif"):
        finfo = file.stem.split("_")
        task = finfo[3].lower()
        if task == "noise":
            continue
        bids_path.update(task=task)
        raw = read_raw_fif(file)
        write_raw_bids(
            raw,
            bids_path,
            events=None,  # TODO: extract and add events
            event_id=None,  # TODO: validate event IDs based on constants
            empty_room=empty_room,
            overwrite=True,
        )
        sidecar_fname = bids_path.copy().update(
            suffix=bids_path.datatype, extension=".json"
        )
        _write_dewar_position("68°", sidecar_fname.fpath)


def _write_meg_calibration_crosstalk(bids_path) -> None:
    """Write MEG calibration and crosstalk files.

    Parameters
    ----------
    bids_path : BIDSPath
        A :class:`~mne_bids.BIDSPath` with at least root amd subject set, and with
        datatype set to 'meg'.
    """
    assert bids_path.root is not None
    assert bids_path.subject is not None
    assert bids_path.datatype == "meg"
    fname = files("project_hnp.bids") / "assets" / "calibration" / "sss_cal.dat"
    assert fname.exists()  # sanity-check
    write_meg_calibration(fname, bids_path)
    fname = files("project_hnp.bids") / "assets" / "cross-talk" / "ct_sparse.fif"
    assert fname.exists()  # sanity-check
    write_meg_crosstalk(fname, bids_path)


def _write_dewar_position(position: str, sidecar_fname: Path):
    """Write the dewar position."""
    assert isinstance(position, str)
    with open(sidecar_fname, encoding="utf-8-sig") as fin:
        sidecar_json = json.load(fin)
    sidecar_json["DewarPosition"] = position
    _write_json(sidecar_fname, sidecar_json, True)


@fill_doc
def _write_eeg_datasets(
    bids_path: BIDSPath,
    data_eeg: Path,
) -> None:
    """Write EEG datasets.

    Parameters
    ----------
    %(bids_path_root_sub)s
    data_eeg : Path
        Path to the EEG dataset.
    """
    assert bids_path.root is not None
    assert bids_path.subject is not None
    for file in data_eeg.glob("*.mff"):
        finfo = file.stem.split("_")
        assert finfo[0].startswith("sub")  # sanity-check
        try:
            subject = int(finfo[0].split("-")[1])
        except Exception:
            raise ValueError(
                f"The subject ID could not be parsed from the filename '{file.name}'."
            )
        if int(bids_path.subject) != subject:
            raise ValueError(
                f"The subject number in the filename ({subject}) does not match "
                "the subject number requested in the BIDS path "
                f"({int(bids_path.subject)})."
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
    # now that the input is validated, we can update the BIDS dataset
    bids_path.update(datatype="eeg")
    for file in data_eeg.glob("*.mff"):
        finfo = file.stem.split("_")
        bids_path.update(task=finfo[1].split("-")[1])
        raw = read_raw_egi(file)
        write_raw_bids(
            raw,
            bids_path,
            events=None,  # TODO: extract and add events
            event_id=None,  # TODO: validate event IDs based on constants
            overwrite=True,
        )


@fill_doc
def _write_mri_datasets(
    bids_path: BIDSPath,
    data_mri: Path,
) -> None:
    """Write MRI datasets.

    Parameters
    ----------
    %(bids_path_root_sub)s
    data_mri : Path
        Path to the MRI dataset.
    """
    assert bids_path.root is not None
    assert bids_path.subject is not None
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
