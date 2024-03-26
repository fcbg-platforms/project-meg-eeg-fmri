from __future__ import annotations

import json
import os
from importlib.resources import files
from typing import TYPE_CHECKING
from warnings import warn

from mne.io import read_raw_fif
from mne_bids import write_meg_calibration, write_meg_crosstalk, write_raw_bids
from mne_bids.utils import _write_json

from ..utils._docs import fill_doc
from ._utils import validate_bids_paths, validate_data_MEG

if TYPE_CHECKING:
    from pathlib import Path

    from mne_bids import BIDSPath


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
    %(data_meg)s
    """
    validate_bids_paths(bids_path, bids_path_raw)
    validate_data_MEG(data_meg, int(bids_path.subject))
    for file in data_meg.glob("*.fif"):
        task = file.stem.split("_")[3].lower()
        if task == "noise":
            empty_room = read_raw_fif(file)
            break
    else:
        warn(
            RuntimeWarning,
            f"The empty-room recording is missing in '{str(data_meg)}'.",
            stacklevel=2,
        )
    # update BIDSPath and create folders if necessary
    bids_path.update(datatype="meg")
    bids_path_raw.update(datatype="meg", suffix="meg", extension=".fif")
    os.makedirs(bids_path_raw.fpath.parent, exist_ok=True)
    # save BIDS and raw dataset
    _write_meg_calibration_crosstalk(bids_path)
    for file in data_meg.glob("*.fif"):
        task = file.stem.split("_")[3].lower()
        bids_path_raw.update(task=task)
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
        _write_dewar_position("68°", sidecar_fname.fpath)
        raw.save(bids_path_raw.fpath, overwrite=True)


def _write_meg_calibration_crosstalk(bids_path: BIDSPath) -> None:
    """Write MEG calibration and crosstalk files."""
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
    assert isinstance(sidecar_fname, Path)
    assert sidecar_fname.exists()
    with open(sidecar_fname, encoding="utf-8-sig") as fin:
        sidecar_json = json.load(fin)
    sidecar_json["DewarPosition"] = position
    _write_json(sidecar_fname, sidecar_json, True)
