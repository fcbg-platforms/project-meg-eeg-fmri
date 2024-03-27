from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

import pytest
from mne_bids import BIDSPath

from project_hnp.derivatives import export_krios_digitization
from project_hnp.krios import read_EGI_ch_names

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.filterwarnings("ignore:Some electrodes are missing*:RuntimeWarning")
def test_export_krios_digitization(tmp_path: Path, krios_file: Path):
    """Test exporting a Krios digitization file."""
    bids_path_raw = BIDSPath(
        root=tmp_path / "raw", subject="01", datatype="eeg", suffix="eeg"
    ).mkdir()
    os.makedirs(tmp_path / "derivatives")
    shutil.copy2(krios_file, bids_path_raw.fpath.with_suffix(".csv"))
    export_krios_digitization(tmp_path / "raw", tmp_path / "derivatives", subject=1)
    fname = tmp_path / "derivatives" / "sub-01" / "eeg" / "sub-01_elc_coords.xyz"
    assert fname.exists()
    with open(fname) as fid:
        lines = fid.readlines()
    ch_names = read_EGI_ch_names()
    assert int(lines[0].split("\t")[0]) == len(ch_names)
    assert len(lines[1:]) == len(ch_names)
    assert list(ch_names) == [elt.split("\t")[-1].rstrip("\n") for elt in lines[1:]]
    fname = tmp_path / "derivatives" / "sub-01" / "eeg" / "sub-01_fid_coords.xyz"
    assert fname.exists()
    with open(fname) as fid:
        lines = fid.readlines()
    assert int(lines[0].split("\t")[0]) == 3
    assert ["RPA", "LPA", "NZ"] == [
        elt.split("\t")[-1].rstrip("\n") for elt in lines[1:]
    ]
