from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

from mne.io import read_raw_egi
from mne_bids import BIDSPath, write_raw_bids

from ..krios import read_EGI_ch_names, read_krios_montage
from ..utils._checks import ensure_path
from ..utils._docs import fill_doc
from ._constants import EGI_CH_TO_DROP
from ._utils import ensure_subject_int, validate_data_EEG

if TYPE_CHECKING:
    from pathlib import Path

    from mne.channels import DigMontage
    from mne.io import BaseRaw


@fill_doc
def write_eeg_datasets(
    root: Path | str,
    root_raw: Path | str,
    subject: int,
    data_eeg: Path,
) -> None:
    """Write EEG datasets.

    Parameters
    ----------
    %(data_eeg)s
    """
    root = ensure_path(root, must_exist=True)
    root_raw = ensure_path(root_raw, must_exist=True)
    subject = ensure_subject_int(subject)
    bids_path = BIDSPath(root=root, subject=str(subject).zfill(2), datatype="eeg")
    bids_path_raw = BIDSPath(
        root=root_raw, subject=str(subject).zfill(2), datatype="eeg", suffix="eeg"
    )
    validate_data_EEG(data_eeg, int(bids_path.subject))
    os.makedirs(bids_path_raw.fpath.parent, exist_ok=True)
    # look for montage
    files = [file for file in data_eeg.glob("*.csv")]
    if len(files) == 0:
        montage = None
    elif len(files) == 1:
        montage = read_krios_montage(files[0])
        shutil.copy2(files[0], bids_path_raw.fpath.with_suffix(".csv"))
    else:
        raise ValueError(
            "Expected only one Krios digitization csv file, got "
            f"{[file.name for file in files]}."
        )
    # populate the BIDS dataset
    for file in data_eeg.glob("*.mff"):
        task = file.stem.split("_")[1].split("-")[1]
        bids_path.update(task=task)
        bids_path_raw.update(task=task)
        raw = read_raw_egi(file)
        _process_EGI_raw(raw, montage)
        write_raw_bids(
            raw,
            bids_path,
            events=None,  # TODO: extract and add events
            event_id=None,  # TODO: validate event IDs based on constants
            overwrite=True,
        )
        # copy original to bids_path_raw location, .mff are not handled equally between
        # win, linux and macOS
        if file.is_file():
            shutil.copy2(file, bids_path_raw.fpath.with_suffix(".mff"))
        elif file.is_dir():
            shutil.copytree(file, bids_path_raw.fpath.with_suffix(".mff"))


def _process_EGI_raw(raw: BaseRaw, montage: DigMontage) -> None:
    """Apply basic preprocessing steps to EGI raw data in-place."""
    ch_names = read_EGI_ch_names()
    ch_names2rename = [
        ch for ch in raw.ch_names if not (ch.startswith("DIN") or ch.startswith("STI"))
    ]
    raw.rename_channels(
        {ch1: ch2 for ch1, ch2 in zip(ch_names2rename, ch_names, strict=True)}
    )
    raw.set_montage(montage)
    raw.drop_channels(EGI_CH_TO_DROP)
    # TODO: Handle stim channels
