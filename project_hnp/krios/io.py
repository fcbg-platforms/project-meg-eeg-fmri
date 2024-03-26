from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pycpd import RigidRegistration

from ..utils._checks import ensure_path
from ._transform import krios_to_head_coordinate, reorder_electrodes

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


_LPA_LABEL: str = "LEFT_PP_MARKER"
_RPA_LABEL: str = "RIGHT_PP_MARKER"
_NZ_LABEL: str = "NASION_MARKER"


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
    template_fname = (
        files("project_hnp.krios") / "assets" / "EGI 257.Geneva Average 13.10-10__.xyz"
    )
    df_template = pd.read_csv(template_fname, sep=" ", header=0, skipinitialspace=True)
    elc_template = df_template.loc[:, ["x", "y", "z"]].to_numpy()
    # co-register with template
    # RigidRegistration(...).register() returns 4 elements:
    # - TY : array of shape (n_points, 3), the registered and transformed source points
    # - s_reg : float, the scale of the registration
    # - R_reg : array of shape (3, 3), the rotation matrix of the registration
    # - t_reg : array of shape (1, 3), the translation vector of the registration
    elc_TY, _ = RigidRegistration(X=elc_template, Y=elc).register()  # TODO: check
    # reorder electrodes according to template
    elc_reordered = reorder_electrodes(elc_TY, elc_template)
    return elc_reordered, fid
