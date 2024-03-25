from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING
from warnings import warn

from mne.io import read_raw_fif
from mne_bids import (
    BIDSPath,
    write_meg_calibration,
    write_meg_crosstalk,
    write_raw_bids,
)

from ..utils._checks import ensure_int, ensure_path
from ._constants import EXPECTED_MEG, OPTIONAL_MEG

if TYPE_CHECKING:
    from pathlib import Path


def bidsification(
    root: Path | str,
    subject: int,
    data_eeg: Path | str,
    data_meg: Path | str,
    data_mri: Path | str,
):
    """Convert a dataset to BIDS."""
    root = ensure_path(root, must_exist=True)
    subject = ensure_int(subject, "subject")
    if subject <= 0:
        raise ValueError(
            f"Argument 'subject' must be a positive integer, got {subject}."
        )
    data_eeg = ensure_path(data_eeg, must_exist=True)
    data_meg = ensure_path(data_meg, must_exist=True)
    data_mri = ensure_path(data_mri, must_exist=True)

    bids_path = BIDSPath(root=root, subject=subject)
    _write_meg_datasets(bids_path, data_meg)


def _write_meg_datasets(bids_path: BIDSPath, data_meg: Path) -> None:
    """Write MEG datasets.

    Parameters
    ----------
    bids_path : BIDSPath
        A :class:`~mne_bids.BIDSPath` with at least root amd subject set.
    """
    empty_room = None
    for file in data_meg.glob("*.fif"):
        finfo = file.stem.split("_")
        assert finfo[0] == "sub"  # sanity-check
        if bids_path.subject != int(finfo[1]):
            raise ValueError(
                f"The subject number in the filename ({int(finfo[1])}) does not match "
                "the subject number requested in the BIDS path ({bids_path.subject})."
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
        bids_path.update(task=finfo[3].lower())
        raw = read_raw_fif(file)
        write_raw_bids(
            raw,
            bids_path,
            events=None,
            event_id=None,
            empty_room=empty_room,
        )


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
    write_meg_calibration(fname, bids_path)
    fname = files("project_hnp.bids") / "assets" / "crosstalk" / "ct_sparse.fif"
    write_meg_crosstalk(fname, bids_path)
