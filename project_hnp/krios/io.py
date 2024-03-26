from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from mne.channels import make_dig_montage, make_standard_montage
from mne.channels.montage import _get_montage_in_head
from mne.defaults import HEAD_SIZE_DEFAULT as _HEAD_SIZE_DEFAULT
from pycpd import RigidRegistration

from ..utils._checks import ensure_path
from ._transform import fit_matched_points, krios_to_head_coordinate, reorder_electrodes

if TYPE_CHECKING:
    from pathlib import Path

    from mne.channels import DigMontage
    from numpy.typing import NDArray


_LPA_LABEL: str = "LEFT_PP_MARKER"
_RPA_LABEL: str = "RIGHT_PP_MARKER"
_NZ_LABEL: str = "NASION_MARKER"
_TEMPLATE_FNAME: Path = (
    files("project_hnp.krios") / "assets" / "EGI 257.Geneva Average 13.10-10__.xyz"
)


def read_krios(fname: Path | str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read electrode locations from a Krios file.

    Parameters
    ----------
    fname : Path | str
        Path to the Krios file.

    Returns
    -------
    elc : array of shape (n_electrodes, 3)
        (x, y, z) coordinates of the EEG electrodes in the head coordinate system.
    fid : array of shape (3, 3)
        (x, y, z) coordinates of the fiducials in the head coordinate system, ordered as
        RPA, LPA, NZ.
    """
    fname = ensure_path(fname, must_exist=True)
    df = pd.read_csv(fname, sep=",", header=0)
    eeg_idx = (df["Special Sensor"] == "EEG").to_numpy()
    lpa_idx = (df["Special Sensor"] == _LPA_LABEL).to_numpy()
    rpa_idx = (df["Special Sensor"] == _RPA_LABEL).to_numpy()
    nz_idx = (df["Special Sensor"] == _NZ_LABEL).to_numpy()
    # extract (x, y, z) coordinates as array of shape (n_scanned, 3)
    elc = df.loc[eeg_idx, ["x(cm)", "y(cm)", "z(cm)"]].to_numpy()
    rpa = np.squeeze(df.loc[rpa_idx, ["x(cm)", "y(cm)", "z(cm)"]].to_numpy())
    lpa = np.squeeze(df.loc[lpa_idx, ["x(cm)", "y(cm)", "z(cm)"]].to_numpy())
    nz = np.squeeze(df.loc[nz_idx, ["x(cm)", "y(cm)", "z(cm)"]].to_numpy())
    del df
    elc, fid, _ = krios_to_head_coordinate(elc, rpa=rpa, lpa=lpa, nz=nz)
    # read template file
    df_template = pd.read_csv(_TEMPLATE_FNAME, sep=" ", header=0, skipinitialspace=True)
    elc_template = df_template.loc[:, ["x", "y", "z"]].to_numpy()
    # co-register with template
    # RigidRegistration(...).register() returns 2 elements:
    # - TY : array of shape (n_points, 3), the registered and transformed source points
    # - registration parameters : tuple of 3 elements
    #   - s_reg : float, the scale of the registration
    #   - R_reg : array of shape (3, 3), the rotation matrix of the registration
    #   - t_reg : array of shape (3,), the translation vector of the registration
    elc_TY, (s_reg, _, _) = RigidRegistration(X=elc_template, Y=elc).register()
    fid *= s_reg  # apply the same scaling to fiducials
    # reorder electrodes according to template
    elc_reordered = reorder_electrodes(elc_TY, elc_template)
    return elc_reordered, fid


def read_krios_montage(fname: Path | str) -> DigMontage:
    """Read electrode locations from a Krios file in a DigMontage.

    Parameters
    ----------
    fname : Path | str
        Path to the Krios file.

    Returns
    -------
    montage : DigMontage
        MNE channel montage for the EEG electrodes in the head coordinate frame.
    """
    elc, fid = read_krios(fname)
    # apply scaling to meters
    standard = make_standard_montage("standard_1020", head_size=_HEAD_SIZE_DEFAULT)
    standard = _get_montage_in_head(standard).get_positions()
    assert standard["coord_frame"] == "head"  # sanity-check
    fid_standard = np.array(
        [standard["rpa"], standard["lpa"], standard["nasion"]], dtype=np.float64
    )
    _, scaling = fit_matched_points(fid, fid_standard, scale=True)
    elc *= scaling
    fid *= scaling
    # figure out labels from template
    df_template = pd.read_csv(_TEMPLATE_FNAME, sep=" ", header=0, skipinitialspace=True)
    ch_names = df_template["labels"].values
    del df_template
    if elc.shape[0] != ch_names.size:
        raise ValueError(
            f"Number of electrodes ({elc.shape[0]}) does not match the "
            f"number of channel names ({ch_names.size}) in the template."
        )
    ch_pos = dict(zip(ch_names, elc, strict=True))
    return make_dig_montage(
        ch_pos=ch_pos,
        nasion=fid[2, :],
        lpa=fid[1, :],
        rpa=fid[0, :],
        coord_frame="head",
    )
