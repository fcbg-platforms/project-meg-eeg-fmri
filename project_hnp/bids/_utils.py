from __future__ import annotations

from typing import TYPE_CHECKING

from ._constants import EXPECTED_MEG, OPTIONAL_MEG

if TYPE_CHECKING:
    from pathlib import Path

    from mne_bids import BIDSPath


def validate_bids_paths(bids_path: BIDSPath, bids_path_raw: BIDSPath):
    """Validate the provided BIDS path."""
    assert bids_path.root is not None
    assert bids_path.subject is not None
    assert bids_path_raw.root is not None
    assert bids_path_raw.subject is not None
    assert bids_path.subject == bids_path_raw.subject


def validate_data_MEG(data_meg: Path, subject: int) -> None:
    """Validate a folder containing MEG data."""
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
