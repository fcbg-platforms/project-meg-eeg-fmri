from __future__ import annotations

import json
import os
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
from mne_bids.utils import _write_json

from ..utils._docs import fill_doc
from ._constants import EXPECTED_MEG, OPTIONAL_MEG

if TYPE_CHECKING:
    from pathlib import Path


@fill_doc
def write_meg_datasets(
    bids_path: BIDSPath,
    bids_path_raw: BIDSPath,
    data_meg: Path,
) -> None:
    """Write MEG datasets.

    Parameters
    ----------
    %(bids_path_root_sub)s
    %(bids_path_root_raw_sub)s
    data_meg : Path
        Path to the MEG dataset.
    """
    assert bids_path.root is not None
    assert bids_path.subject is not None
    assert bids_path_raw.root is not None
    assert bids_path_raw.subject is not None
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
    bids_path_raw.update(datatype="meg", suffix="meg")
    _write_meg_calibration_crosstalk(bids_path)
    for file in data_meg.glob("*.fif"):
        finfo = file.stem.split("_")
        task = finfo[3].lower()
        bids_path_raw.update(task=task, extension=".fif")
        os.makedirs(bids_path_raw.fpath.parent, exist_ok=True)
        if task == "noise":  # only move RAW file
            raw = read_raw_fif(file)
            raw.save(bids_path_raw.fpath, overwrite=True)
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
        raw.save(bids_path_raw.fpath, overwrite=True)
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
