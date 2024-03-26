from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

from mne.io import read_raw_egi
from mne_bids import BIDSPath, write_raw_bids

from ..krios import read_EGI_ch_names, read_krios_montage
from ..utils._docs import fill_doc
from ._constants import EGI_CH_TO_DROP, EXPECTED_EEG
from ._utils import validate_bids_paths

if TYPE_CHECKING:
    from pathlib import Path


@fill_doc
def write_eeg_datasets(
    bids_path: BIDSPath,
    bids_path_raw: BIDSPath,
    data_eeg: Path,
) -> None:
    """Write EEG datasets.

    Parameters
    ----------
    %(bids_path_root_sub)s
    %(bids_path_root_raw_sub)s
    data_eeg : Path
        Path to the EEG dataset.
    """
    validate_bids_paths(bids_path, bids_path_raw)
    files = [file for file in data_eeg.glob("*.csv")]
    if len(files) == 0:
        montage = None
    elif len(files) == 1:
        montage = read_krios_montage(files[0])
        dst = (
            bids_path_raw.copy()
            .update(datatype="eeg", suffix="eeg", extension=".json")
            .fpath
        )
        os.makedirs(dst.parent, exist_ok=True)
        shutil.copy2(files[0], dst.with_suffix(".csv"))
    else:
        raise ValueError(
            "Expected only one Krios digitization file, got "
            f"{[file.name for file in files]}."
        )
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
    ch_names = read_EGI_ch_names()
    bids_path.update(datatype="eeg")
    bids_path_raw.update(datatype="eeg", suffix="eeg")
    for file in data_eeg.glob("*.mff"):
        finfo = file.stem.split("_")
        task = finfo[1].split("-")[1]
        bids_path.update(task=task)
        bids_path_raw.update(task=task)
        raw = read_raw_egi(file)
        ch_names2rename = [
            ch
            for ch in raw.ch_names
            if not (ch.startswith("DIN") or ch.startswith("STI"))
        ]
        raw.rename_channels(
            {ch1: ch2 for ch1, ch2 in zip(ch_names2rename, ch_names, strict=True)}
        )
        raw.set_montage(montage)
        raw.drop_channels(EGI_CH_TO_DROP)
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
            os.makedirs(bids_path_raw.fpath.parent, exist_ok=True)
            shutil.copy2(file, bids_path_raw.fpath.with_suffix(".mff"))
        elif file.is_dir():
            os.makedirs(bids_path_raw.fpath.parent, exist_ok=True)
            shutil.copytree(file, bids_path_raw.fpath.with_suffix(".mff"))
