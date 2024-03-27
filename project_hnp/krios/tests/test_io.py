from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from mne.channels import DigMontage
from mne.defaults import HEAD_SIZE_DEFAULT
from numpy.testing import assert_allclose
from pycpd import RigidRegistration

from project_hnp.krios.io import _TEMPLATE_FNAME, read_krios, read_krios_montage

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.filterwarnings("ignore:Some electrodes are missing*:RuntimeWarning")
def test_read_krios(krios_file: Path):
    """Test that read points are within a sphere defined by the fiducials."""
    elc, fid = read_krios(krios_file)
    assert np.allclose(fid[0, 1:], np.zeros(2))  # RPA
    assert np.allclose(fid[1, 1:], np.zeros(2))  # LPA
    assert np.allclose(fid[2, np.array([0, 2], dtype=np.int8)], np.zeros(2))  # NZ
    # check that all electrodes are within a sphere defined by the fiducials
    radius = np.average(np.linalg.norm(fid, axis=1)) * 2
    distances = np.linalg.norm(elc, axis=1)
    assert np.all(distances < radius)
    # the Z-axis induces larger deviation from a unit sphere, let's check the X/Y plane
    # separately with a tighter tolerance
    radius = np.average(np.linalg.norm(fid[:, :2], axis=1)) * 1.5
    distances = np.linalg.norm(elc[:, :2], axis=1)
    assert np.all(distances < radius)


@pytest.mark.filterwarnings("ignore:Some electrodes are missing*:RuntimeWarning")
def test_read_krios_rotation_to_template(krios_file: Path):
    """Test that the transformation from read points to the template is identity."""
    elc, _ = read_krios(krios_file)
    elc_template = np.loadtxt(
        _TEMPLATE_FNAME, skiprows=1, usecols=(0, 1, 2), max_rows=257, dtype=np.float64
    )
    # tolerance are flexible because the template and an actual head shape scan vary
    # greatly.
    _, (s_reg, R_reg, t_reg) = RigidRegistration(X=elc_template, Y=elc).register()
    assert_allclose(s_reg, 1.0, rtol=1e-3)
    assert_allclose(R_reg, np.eye(3, dtype=np.float64), atol=1e-2)
    assert_allclose(t_reg, np.zeros(3, dtype=np.float64), atol=1e-1)


def test_read_krios_missing_electrode():
    """Test reading files with missing electrodes."""
    elc1, fid1 = read_krios(files("project_hnp.krios") / "tests" / "data" / "2.csv")
    with pytest.warns(RuntimeWarning, match="Some electrodes are missing"):
        elc2, fid2 = read_krios(files("project_hnp.krios") / "tests" / "data" / "1.csv")
    assert elc1.shape == elc2.shape
    assert fid1.shape == fid2.shape


@pytest.mark.filterwarnings("ignore:Some electrodes are missing*:RuntimeWarning")
def test_read_krios_montage(krios_file: Path):
    """Test reading a Krios .csv info a DigMontage."""
    montage = read_krios_montage(krios_file)
    assert isinstance(montage, DigMontage)
    # check the coordinates on the XY plane
    radius = HEAD_SIZE_DEFAULT * 1.5
    elc = np.array([array for array in montage.get_positions()["ch_pos"].values()])
    distances = np.linalg.norm(elc[:, :2], axis=1)
    assert np.all(distances < radius)
