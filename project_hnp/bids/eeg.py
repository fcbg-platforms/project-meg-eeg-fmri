from __future__ import annotations

import shutil
import warnings
from typing import TYPE_CHECKING

from mne.io import read_raw_egi
from mne_bids import BIDSPath, write_raw_bids

from ..krios import read_EGI_ch_names, read_krios_montage
from ..utils._checks import ensure_path, ensure_subject_int
from ..utils._docs import fill_doc
from ..utils.logs import logger
from ._utils import (
    fetch_participant_information,
    find_events,
    validate_data_EEG,
    write_participant_information,
)

if TYPE_CHECKING:
    from pathlib import Path

    from mne.channels import DigMontage
    from mne.io import BaseRaw


@fill_doc
def write_eeg_datasets(
    root: Path | str,
    root_raw: Path | str,
    subject: int,
    data_eeg: Path | str,
) -> None:
    """Write EEG datasets.

    The EEG dataset should contain the recordings in .mff format and a digitization in
    .csv format.

    Parameters
    ----------
    %(bids_root)s
    %(bids_root_raw)s
    %(bids_subject)s
    %(data_eeg)s
    """
    root = ensure_path(root, must_exist=True)
    root_raw = ensure_path(root_raw, must_exist=True)
    subject = ensure_subject_int(subject)
    data_eeg = ensure_path(data_eeg, must_exist=True)
    validate_data_EEG(data_eeg, subject)
    # create BIDS Path and folders
    bids_path = BIDSPath(root=root, subject=str(subject).zfill(2), datatype="eeg")
    bids_path_raw = BIDSPath(
        root=root_raw, subject=str(subject).zfill(2), datatype="eeg", suffix="eeg"
    ).mkdir()
    # look for montage
    files = [file for file in data_eeg.glob("*.csv")]
    if len(files) == 0:
        montage = None
    elif len(files) == 1:
        try:
            montage = read_krios_montage(files[0])
        except Exception as error:
            logger.exception(error)
            montage = None
        if montage is not None:
            shutil.copy2(files[0], bids_path_raw.fpath.with_suffix(".csv"))
    else:
        logger.error(
            "Expected only one Krios digitization csv file, got %s",
            [file.name for file in files],
        )
        montage = None
    # look for existing participant information
    participant_info = fetch_participant_information(bids_path)
    # populate the BIDS dataset
    for file in data_eeg.glob("*.mff"):
        task = file.stem.split("_")[1].split("-")[1]
        bids_path.update(task=task)
        bids_path_raw.update(task=task)
        raw = read_raw_egi(file, events_as_annotations=False)
        _process_EGI_raw(raw, montage)
        events, event_id = find_events(raw, task)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Converting data files to BrainVision format",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Encountered unsupported non-voltage units",
                category=UserWarning,
            )
            if task == "rest":
                warnings.filterwarnings(
                    "ignore",
                    message="No events found or provided.",
                    category=RuntimeWarning,
                )
            write_raw_bids(
                raw,
                bids_path,
                events=events,
                event_id=event_id,
                overwrite=True,
            )
        # copy original to bids_path_raw location, .mff are not handled equally between
        # win, linux and macOS
        if file.is_file():
            shutil.copy2(file, bids_path_raw.fpath.with_suffix(".mff"))
        elif file.is_dir():
            shutil.copytree(
                file, bids_path_raw.fpath.with_suffix(".mff"), dirs_exist_ok=True
            )
    # add back participant information if needed
    write_participant_information(bids_path, participant_info)


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
