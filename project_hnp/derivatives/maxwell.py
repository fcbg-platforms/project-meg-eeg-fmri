from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING

from mne.chpi import read_head_pos
from mne.preprocessing import maxwell_filter
from mne_bids import BIDSPath, read_raw_bids

from ..bids._constants import EXPECTED_MEG, OPTIONAL_MEG
from ..utils._checks import ensure_path, ensure_subject_int
from ..utils._docs import fill_doc
from ..utils.logs import warn

if TYPE_CHECKING:
    from pathlib import Path


_HEAD_DESTINATION: tuple[float, float, float] | str | Path = (0, 0, 0.04)
_CT_SPARSE: Path = files("project_hnp.derivatives") / "assets" / "ct_sparse.fif"
_SSS_CAL: Path = files("project_hnp.derivatives") / "assets" / "sss_cal.dat"


@fill_doc
def run_maxwell_filter(root: Path | str, derivative: Path | str, subject: int):
    """Run SSS on th MEG recording for the given subject.

    Parameters
    ----------
    %(bids_root)s
    %(bids_derivative)s
    %(bids_subject)s
    """
    root = ensure_path(root, must_exist=True)
    derivative = ensure_path(derivative, must_exist=True)
    subject = ensure_subject_int(subject)
    bids_path = BIDSPath(root=root, subject=str(subject).zfill(2), datatype="meg")
    bids_path_derivative = BIDSPath(
        root=derivative, subject=str(subject).zfill(2), datatype="meg"
    )
    for task in EXPECTED_MEG.union(OPTIONAL_MEG):
        raw = read_raw_bids(bids_path.update(task=task))
        if len(raw.info["bads"]) == 0:
            warn("No bad channels found. SSS might spread noise!")
        bids_path_derivative.update(task=task)
        fname = (
            bids_path_derivative.directory
            / f"{bids_path_derivative.basename}_head_pos.pos"
        )
        fname = ensure_path(fname, must_exist=True)
        head_pos = read_head_pos(fname)
        raw_sss = maxwell_filter(
            raw,
            calibration=_SSS_CAL,
            cross_talk=_CT_SPARSE,
            head_pos=head_pos,
            destination=_HEAD_DESTINATION,
            extended_proj=raw.info["projs"],
        )
        raw_sss.save(
            bids_path_derivative.directory / f"{bids_path.basename}_raw_sss.fif"
        )
