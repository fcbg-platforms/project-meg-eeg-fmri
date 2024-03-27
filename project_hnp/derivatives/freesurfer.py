from __future__ import annotations

from typing import TYPE_CHECKING

from mne.utils.misc import run_subprocess
from mne_bids import BIDSPath

from ..utils._checks import ensure_path, ensure_subject_int
from ..utils._docs import fill_doc

if TYPE_CHECKING:
    from pathlib import Path


@fill_doc
def recon_all(root: Path | str, derivative: Path | str, subject: int) -> None:
    """Run recon-all for the given subject.

    Parameters
    ----------
    %(bids_root)s
    %(bids_derivative)s
    %(bids_subject)s
    """
    root = ensure_path(root, must_exist=True)
    derivative = ensure_path(derivative, must_exist=True)
    subject = ensure_subject_int(subject)
    # run recon-all
    bids_path = BIDSPath(
        root=root,
        subject=str(subject).zfill(2),
        datatype="anat",
        suffix="T1w",
        task=None,
    )
    bids_path_derivative = BIDSPath(
        root=derivative, subject=str(subject).zfill(2)
    ).mkdir()
    # command: recon-all -s <subject> -sd <subjectsdir> -i <volume> -3T -all
    command = [
        "recon-all",
        "-s",
        str(subject).zfill(2),
        "-sd",
        str(bids_path_derivative.directory / "freesurfer"),
        "-i",
        str(bids_path.fpath),
        "-3T",
        "-all",
    ]
    run_subprocess(command)


@fill_doc
def mri_convert(derivative: Path | str, subject: int) -> None:
    """Run mri_convert.

    Parameters
    ----------
    %(bids_derivative)s
    %(bids_subject)s
    """
    derivative = ensure_path(derivative, must_exist=True)
    subject = ensure_subject_int(subject)
